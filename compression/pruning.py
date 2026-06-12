"""
Structured channel pruning for CenterPoint backbone (SECOND).

Uses torch-pruning for L1-norm channel pruning with automatic dependency
graph resolution across pts_backbone (SECOND) + pts_neck (SECONDFPN).

Flow per ratio: load FP32 → prune → fine-tune → eval → save checkpoint.
Run once per ratio for the sweep (25%, 40%, 55%).

Usage:
    python EdgeFusion-CenterPoint/compression/pruning.py \
        --config $CFG \
        --checkpoint $CKPT \
        --ratio 0.25 \
        --epochs 5 \
        --batch-size 16 \
        --lr 4e-4 \
        --num-workers 16

Sweep:
    for RATIO in 0.25 0.40 0.55; do
        python EdgeFusion-CenterPoint/compression/pruning.py \\
            --config $CFG --checkpoint $CKPT \\
            --ratio $RATIO --epochs 5 \\
            --batch-size 16 --lr 4e-4 --num-workers 16
    done
    # Add --use-cbgs for full CBGS dataset (~4.4x slower, better rare class recovery)

Eval only (after early stop):
    python EdgeFusion-CenterPoint/compression/pruning.py \
        --config $CFG \
        --eval-only \
        --model-path compression/results/pruning/ratio_25/pruned_25_epoch3.pth
"""

import argparse
import json
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch_pruning as tp
from mmdet3d.apis import init_model
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from tqdm import tqdm

RESULTS_DIR = Path(__file__).parent / 'results' / 'pruning'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def eval_checkpoint(args: argparse.Namespace) -> None:
    """Evaluate a saved pruned model checkpoint (.pt full model object)."""
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    print(f'[eval] Loading pruned model from {args.model_path}...')
    model = torch.load(args.model_path, map_location='cuda:0')
    model.cuda()
    _set_train_cfg_from_config(model, cfg)

    metrics = evaluate(model, cfg)
    print(f'[eval] mAP {metrics["mAP"]:.4f}  NDS {metrics["NDS"]:.4f}')

    out = Path(args.model_path).with_suffix('_metrics.json')
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'[eval] Saved to {out}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='CenterPoint structured pruning sweep')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', help='FP32 .pth checkpoint (not needed for --eval-only)')
    p.add_argument('--ratio', type=float, help='Channel pruning ratio')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--use-cbgs', action='store_true', default=False,
                   help='Use CBGS dataset (~4.4x more steps, better rare class sampling). '
                        'Default: raw dataset (~1.7 hrs/epoch vs ~7.5 hrs/epoch).')
    p.add_argument('--out', default=str(RESULTS_DIR))
    p.add_argument('--eval-only', action='store_true',
                   help='Evaluate a saved pruned model without training')
    p.add_argument('--model-path', type=str,
                   help='Path to pruned model checkpoint for --eval-only or --recalibrate')
    p.add_argument('--recalibrate', action='store_true',
                   help='Recover a checkpoint with stale EMA BN buffers (see '
                        '_merged_ema_state_dict). Requires --checkpoint (FP32), '
                        '--ratio, and --model-path (broken checkpoint state_dict).')
    p.add_argument('--recalib-batches', type=int, default=200,
                   help='Number of forward-only batches for BN recalibration')
    return p.parse_args()


def build_train_loader(cfg: Config, batch_size: int, num_workers: int = 8,
                       use_cbgs: bool = False):
    cfg = cfg.copy()
    if not use_cbgs and cfg.train_dataloader.dataset.type == 'CBGSDataset':
        # Unwrap CBGSDataset → raw NuScenesDataset
        # Raw: ~1,758 steps/epoch at batch=16 (~1.7 hrs)
        # CBGS: ~7,724 steps/epoch at batch=16 (~7.5 hrs)
        cfg.train_dataloader.dataset = cfg.train_dataloader.dataset.dataset
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = num_workers
    return Runner.build_dataloader(cfg.train_dataloader)


def build_val_loader(cfg: Config):
    return Runner.build_dataloader(cfg.test_dataloader)


def get_bev_example(model: nn.Module, batch: dict) -> torch.Tensor:
    """Get a BEV feature map for torch-pruning dependency tracing."""
    model.eval()
    with torch.no_grad():
        vd = model.data_preprocessor.voxelize(batch['inputs']['points'], batch)
        pf = model.pts_voxel_encoder(
            vd['voxels'].cuda(), vd['num_points'].cuda(), vd['coors'].cuda()
        ).squeeze()
        bs = int(vd['coors'][-1, 0].item()) + 1
        bev = model.pts_middle_encoder(pf, vd['coors'].cuda(), bs)
    return bev


def _set_train_cfg_from_config(model: nn.Module, cfg: Config) -> None:
    """Set pts_bbox_head.train_cfg from model config fields."""
    if model.pts_bbox_head.train_cfg is not None:
        return
    head_cfg = cfg.model.get('pts_bbox_head', {})
    bbox_coder = head_cfg.get('bbox_coder', {})
    voxel_layer = cfg.model.get('data_preprocessor', {}).get('voxel_layer', {})
    scatter = cfg.model.get('pts_middle_encoder', {})
    pc_range = list(voxel_layer.get(
        'point_cloud_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    ))
    voxel_size = list(voxel_layer.get('voxel_size', [0.2, 0.2, 8.0]))
    out_size_factor = bbox_coder.get('out_size_factor', 4)
    output_shape = scatter.get('output_shape', [512, 512])
    model.pts_bbox_head.train_cfg = dict(
        point_cloud_range=pc_range,
        voxel_size=voxel_size,
        grid_size=[output_shape[0], output_shape[1], 1],
        out_size_factor=out_size_factor,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
    )
    print(f'[pruning] train_cfg set: out_size_factor={out_size_factor}')


# ── pruning ───────────────────────────────────────────────────────────────────

class BackboneNeck(nn.Module):
    """Wrapper for backbone + neck dependency tracing."""
    def __init__(self, backbone, neck):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, x):
        from mmdet3d.models.necks.second_fpn import SECONDFPN as _S
        _orig = _S.forward

        def _traceable(self, x):
            outs = [self.deblocks[i](x[i]) for i in range(len(self.deblocks))]
            return torch.cat(outs, dim=1) if len(self.deblocks) > 1 else outs[0]

        _S.forward = _traceable
        try:
            feats = self.backbone(x)
            out = self.neck(feats)
        finally:
            _S.forward = _orig
        return out


def _rebuild_shared_conv(model: nn.Module, bev_example: torch.Tensor) -> None:
    """Rebuild shared_conv with correct in_channels after backbone+neck pruning.

    torch-pruning adjusts backbone+neck channels but shared_conv still expects
    the original channel count. We detect the actual pruned neck output via a
    forward pass and reinitialize shared_conv.conv with the correct in_channels.
    Fine-tuning recovers the reinitialized weights quickly (single 3x3 conv).
    """
    model.eval()
    with torch.no_grad():
        feats = model.pts_backbone(bev_example)
        neck_out = model.pts_neck(feats)
        new_in_ch = neck_out[0].shape[1]

    old_conv = model.pts_bbox_head.shared_conv.conv
    if old_conv.in_channels == new_in_ch:
        return  # No change needed

    new_conv = nn.Conv2d(
        new_in_ch, old_conv.out_channels,
        old_conv.kernel_size, old_conv.stride, old_conv.padding,
        bias=old_conv.bias is not None,
    ).to(bev_example.device)
    nn.init.kaiming_normal_(new_conv.weight)
    if new_conv.bias is not None:
        nn.init.zeros_(new_conv.bias)

    model.pts_bbox_head.shared_conv.conv = new_conv
    print(f'[pruning] shared_conv rebuilt: {old_conv.in_channels} → {new_in_ch} in_channels')


def prune_model(model: nn.Module, bev_example: torch.Tensor, ratio: float) -> None:
    """Apply L1-norm structured channel pruning to backbone + neck."""
    combined = BackboneNeck(model.pts_backbone, model.pts_neck)
    combined.eval()

    # Param count before
    before = sum(p.numel() for p in model.pts_backbone.parameters())
    before += sum(p.numel() for p in model.pts_neck.parameters())
    print(f'[pruning] backbone+neck params before: {before/1e6:.2f}M')

    importance = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        combined,
        example_inputs=bev_example,
        importance=importance,
        pruning_ratio=ratio,
        ignored_layers=[],
        global_pruning=False,
    )
    pruner.step()

    model.pts_backbone = combined.backbone
    model.pts_neck = combined.neck

    # Rebuild shared_conv to accept pruned neck output channels
    _rebuild_shared_conv(model, bev_example)

    total = sum(p.numel() for p in model.pts_backbone.parameters())
    total += sum(p.numel() for p in model.pts_neck.parameters())
    print(f'[pruning] backbone+neck params after:  {total/1e6:.2f}M '
          f'({100*total/before:.1f}% of original)')


# ── fine-tuning ───────────────────────────────────────────────────────────────

def _merged_ema_state_dict(model: nn.Module, ema_model: nn.Module) -> dict:
    """Combine EMA-smoothed parameters with the live model's BN buffers.

    ema_model.parameters() are tracked via EMA, but BatchNorm running_mean/
    running_var are buffers — never touched by a parameter-only EMA loop.
    ema_model's buffers stay frozen at their post-pruning, pre-training values
    (especially wrong for the reinitialized shared_conv's BN). Using
    ema_model.state_dict() directly overwrites model's properly-trained BN
    stats with these stale values, producing garbage at eval time
    (model.eval() relies on running_mean/running_var).

    Fix: take parameter values from ema_model, buffer values from model.
    """
    param_names = set(dict(model.named_parameters()).keys())
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    merged = {}
    for k, v in model_state.items():
        merged[k] = ema_state[k] if k in param_names else v
    return merged


def fine_tune(
    model: nn.Module,
    train_loader,
    epochs: int,
    lr: float,
    out_dir: Path,
    ratio: float,
    ema_decay: float = 0.999,
) -> None:
    import copy
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    ratio_str = f'{int(ratio * 100):02d}'

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Fine-tune epoch {epoch}/{epochs}')
        for batch in pbar:
            data = model.data_preprocessor(batch, training=True)
            losses = model(**data, mode='loss')
            loss = sum(
                v for k, v in losses.items()
                if 'loss' in k and isinstance(v, torch.Tensor)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 35)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             lr=f'{scheduler.get_last_lr()[0]:.2e}')

        avg_loss = total_loss / len(train_loader)
        print(f'[pruning] epoch {epoch}/{epochs}  loss {avg_loss:.4f}  '
              f'lr {scheduler.get_last_lr()[0]:.2e}')

        # Save checkpoint after every epoch — allows early stopping at any point
        # Merge EMA params with live BN buffers (see _merged_ema_state_dict)
        merged_state = _merged_ema_state_dict(model, ema_model)
        ckpt_path = out_dir / f'pruned_{ratio_str}_epoch{epoch}.pth'
        torch.save({
            'state_dict': merged_state,
            'ratio': ratio,
            'epoch': epoch,
            'loss': avg_loss,
        }, ckpt_path)
        print(f'[pruning] checkpoint saved: {ckpt_path.name}')

    model.load_state_dict(_merged_ema_state_dict(model, ema_model))
    print('[pruning] EMA params + live BN buffers loaded for evaluation.')


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, cfg: Config) -> dict:
    """Evaluate using mmdet3d Runner evaluator — reads all paths from config."""
    from mmengine.evaluator import Evaluator as MMEval
    val_loader = Runner.build_dataloader(cfg.test_dataloader)
    evaluator = MMEval(cfg.test_evaluator)

    model.eval()
    evaluator.dataset_meta = val_loader.dataset.metainfo
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating pruned model'):
            data = model.data_preprocessor(batch, training=False)
            outputs = model(**data, mode='predict')
            evaluator.process(data_samples=outputs, data_batch=batch)

    metrics = evaluator.evaluate(len(val_loader.dataset))
    map_val = metrics.get('NuScenes metric/pred_instances_3d_NuScenes/mAP', 0.0)
    nds_val = metrics.get('NuScenes metric/pred_instances_3d_NuScenes/NDS', 0.0)
    return {'mAP': map_val, 'NDS': nds_val}


def recalibrate_bn(model: nn.Module, train_loader, n_batches: int = 200) -> None:
    """Recompute BatchNorm running stats from scratch via cumulative averaging.

    Used to recover a model whose BN buffers are stale/mismatched relative to
    its trained parameters (e.g. EMA buffer bug — see _merged_ema_state_dict).
    Resets running_mean/running_var/num_batches_tracked, sets momentum=None
    (cumulative average over all batches seen), then runs forward-only passes
    in train mode so BN updates its buffers from real activation statistics.
    No backward pass, no optimizer — only ~150-200 batches needed since the
    conv/linear weights are already trained.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average

    model.train()
    it = iter(train_loader)
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc='Recalibrating BN'):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            data = model.data_preprocessor(batch, training=True)
            model(**data, mode='loss')

    model.eval()
    print(f'[pruning] BN recalibrated over {n_batches} batches.')


def recalibrate_checkpoint(args: argparse.Namespace) -> None:
    """Recover a checkpoint with stale EMA BN buffers via BN recalibration.

    Rebuilds the pruned architecture (deterministic L1 pruning given the same
    FP32 checkpoint + ratio), loads trained parameters from the broken
    checkpoint (buffer values are ignored — ratio's shapes only matter for
    rebuild), recalibrates BN, then evaluates and saves a corrected checkpoint.
    """
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    print('[recalib] Rebuilding pruned architecture from FP32...')
    model = init_model(cfg, checkpoint=args.checkpoint, device='cuda:0')
    _set_train_cfg_from_config(model, cfg)

    train_loader = build_train_loader(cfg, args.batch_size, args.num_workers, args.use_cbgs)
    sample_batch = next(iter(train_loader))
    bev_example = get_bev_example(model, sample_batch)
    prune_model(model, bev_example, args.ratio)

    print(f'[recalib] Loading trained weights from {args.model_path}...')
    ckpt = torch.load(args.model_path, map_location='cuda:0')
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)

    print(f'[recalib] Recalibrating BN over {args.recalib_batches} batches...')
    recalibrate_bn(model, train_loader, args.recalib_batches)

    print('[recalib] Evaluating...')
    metrics = evaluate(model, cfg)
    print(f'[recalib] ratio={args.ratio:.0%}  mAP {metrics["mAP"]:.4f}  '
          f'NDS {metrics["NDS"]:.4f}')

    ratio_str = f'{int(args.ratio * 100):02d}'
    out_dir = Path(args.out) / f'ratio_{ratio_str}'
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(out_dir / f'pruned_{ratio_str}_recalib.pth')
    torch.save({'state_dict': model.state_dict(), 'ratio': args.ratio}, ckpt_path)
    model_path = str(out_dir / f'pruned_model_{ratio_str}_recalib.pt')
    torch.save(model, model_path)
    metrics_path = str(out_dir / f'pruned_{ratio_str}_recalib_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({'ratio': args.ratio, **metrics}, f, indent=2)

    print(f'[recalib] Saved checkpoint: {ckpt_path}')
    print(f'[recalib] Saved full model: {model_path}')
    print(f'[recalib] Metrics saved to {metrics_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.eval_only:
        if not args.model_path:
            raise ValueError('--model-path required with --eval-only')
        eval_checkpoint(args)
        return

    if args.recalibrate:
        if not (args.checkpoint and args.ratio is not None and args.model_path):
            raise ValueError(
                '--checkpoint, --ratio, and --model-path required with --recalibrate'
            )
        recalibrate_checkpoint(args)
        return

    if not args.checkpoint or args.ratio is None:
        raise ValueError('--checkpoint and --ratio required for training')
    ratio_str = f'{int(args.ratio * 100):02d}'
    out_dir = Path(args.out) / f'ratio_{ratio_str}'
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    print('[pruning] Loading FP32 model...')
    model = init_model(cfg, checkpoint=args.checkpoint, device='cuda:0')
    _set_train_cfg_from_config(model, cfg)

    # Get BEV example for tracing
    train_loader = build_train_loader(cfg, args.batch_size, args.num_workers, args.use_cbgs)
    mode = 'CBGS' if args.use_cbgs else 'raw'
    print(f'[pruning] Dataset: {mode} ({len(train_loader)} steps/epoch)')
    sample_batch = next(iter(train_loader))
    bev_example = get_bev_example(model, sample_batch)

    run_name = f'pruning_ratio_{ratio_str}'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'ratio': args.ratio,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
        })

        # ── 1. Prune ─────────────────────────────────────────────────────────
        print(f'\n[pruning] Pruning at ratio {args.ratio:.0%}...')
        prune_model(model, bev_example, args.ratio)

        # ── 2. Fine-tune ──────────────────────────────────────────────────────
        print(f'\n[pruning] Fine-tuning for {args.epochs} epochs...')
        fine_tune(model, train_loader, args.epochs, args.lr, out_dir, args.ratio)

        # ── 3. Evaluate ───────────────────────────────────────────────────────
        print('\n[pruning] Evaluating...')
        metrics = evaluate(model, cfg)
        map_val = metrics['mAP']
        nds_val = metrics['NDS']

        print(f'\n[pruning] ratio={args.ratio:.0%}  mAP {map_val:.4f}  NDS {nds_val:.4f}')
        mlflow.log_metrics({'mAP': map_val, 'NDS': nds_val})

        # ── 4. Save ───────────────────────────────────────────────────────────
        # Save state_dict for PTQ/reference
        ckpt_path = str(out_dir / f'pruned_{ratio_str}.pth')
        torch.save({'state_dict': model.state_dict(), 'ratio': args.ratio}, ckpt_path)

        # Save full model object — required for loading without knowing
        # pruned channel dimensions (state_dict alone causes size mismatch)
        model_path = str(out_dir / f'pruned_model_{ratio_str}.pt')
        torch.save(model, model_path)

        metrics_path = str(out_dir / f'pruned_{ratio_str}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({'ratio': args.ratio, 'mAP': map_val, 'NDS': nds_val}, f, indent=2)

        print(f'[pruning] Saved checkpoint: {ckpt_path}')
        print(f'[pruning] Saved full model: {model_path}')
        print(f'[pruning] Metrics saved to {metrics_path}')


if __name__ == '__main__':
    main()
