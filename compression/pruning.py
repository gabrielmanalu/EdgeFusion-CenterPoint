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
        --batch-size 4

Sweep:
    for RATIO in 0.25 0.40 0.55; do
        python EdgeFusion-CenterPoint/compression/pruning.py \\
            --config $CFG --checkpoint $CKPT \\
            --ratio $RATIO --epochs 5 --batch-size 4
    done
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='CenterPoint structured pruning sweep')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True, help='FP32 .pth checkpoint')
    p.add_argument('--ratio', type=float, required=True,
                   help='Channel pruning ratio (e.g. 0.25 removes 25% channels)')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--out', default=str(RESULTS_DIR))
    return p.parse_args()


def build_train_loader(cfg: Config, batch_size: int):
    cfg = cfg.copy()
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = 4
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
    """Wrapper for joint backbone+neck pruning dependency tracing."""
    def __init__(self, backbone, neck):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, x):
        from mmdet3d.models.necks.second_fpn import SECONDFPN as _S
        _orig = _S.forward

        def _traceable(self, x):
            outs = [self.deblocks[i](x[i]) for i in range(len(self.deblocks))]
            return [torch.cat(outs, dim=1) if len(self.deblocks) > 1 else outs[0]]

        _S.forward = _traceable
        try:
            feats = self.backbone(x)
            out = self.neck(feats)
        finally:
            _S.forward = _orig
        return out


def prune_model(model: nn.Module, bev_example: torch.Tensor, ratio: float) -> None:
    """Apply L1-norm structured channel pruning to backbone + neck."""
    combined = BackboneNeck(model.pts_backbone, model.pts_neck)
    combined.eval()

    # Ignore head — not connected to backbone/neck in this tracing context
    ignored = []

    importance = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        combined,
        example_inputs=bev_example,
        importance=importance,
        pruning_ratio=ratio,
        ignored_layers=ignored,
        global_pruning=False,
    )

    pruner.step()

    # Write back pruned modules
    model.pts_backbone = combined.backbone
    model.pts_neck = combined.neck

    # Count remaining params
    total = sum(p.numel() for p in model.pts_backbone.parameters())
    total += sum(p.numel() for p in model.pts_neck.parameters())
    print(f'[pruning] ratio={ratio:.0%} — backbone+neck params: {total/1e6:.2f}M')


# ── fine-tuning ───────────────────────────────────────────────────────────────

def fine_tune(
    model: nn.Module,
    train_loader,
    epochs: int,
    lr: float,
    ema_decay: float = 0.999,
) -> None:
    import copy
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # EMA model — CenterPoint uses EMA weights for final mAP accuracy
    ema_model = copy.deepcopy(model)
    ema_model.eval()

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

            # EMA update
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

            total_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             lr=f'{scheduler.get_last_lr()[0]:.2e}')

        avg_loss = total_loss / len(train_loader)
        print(f'[pruning] epoch {epoch}/{epochs}  loss {avg_loss:.4f}  '
              f'lr {scheduler.get_last_lr()[0]:.2e}')

    # Load EMA weights before evaluation
    model.load_state_dict(ema_model.state_dict())
    print('[pruning] EMA weights loaded for evaluation.')


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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    ratio_str = f'{int(args.ratio * 100):02d}'
    out_dir = Path(args.out) / f'ratio_{ratio_str}'
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    print('[pruning] Loading FP32 model...')
    model = init_model(cfg, checkpoint=args.checkpoint, device='cuda:0')
    _set_train_cfg_from_config(model, cfg)

    # Get BEV example for tracing
    train_loader = build_train_loader(cfg, args.batch_size)
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
        fine_tune(model, train_loader, args.epochs, args.lr)

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
