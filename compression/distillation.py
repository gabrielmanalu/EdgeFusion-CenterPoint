"""
Knowledge distillation: FP32 teacher -> 25%-pruned-architecture student.

The student is initialized via the same L1-magnitude channel selection as
pruning.py's ratio=0.25 (deterministic given the FP32 checkpoint), giving it
FP32-inherited weights for kept channels — identical architecture, init, and
compute budget as the pruning.py ratio_25 run (0.4081 mAP, task-loss only).

Training combines:
    L_total = L_task + alpha * L_heatmap_distill + beta * L_reg_distill

L_heatmap_distill: MSE between teacher and student sigmoid heatmaps (dense,
    all spatial locations and classes). Carries "dark knowledge" about object
    confidence — including rare classes the student would otherwise miss.

L_reg_distill: teacher-confidence-weighted L1 between teacher and student
    reg/height/dim/rot/vel outputs. Weighting by the teacher's heatmap
    confidence focuses regression distillation on locations where the
    teacher believes an object exists, without needing GT-based masks.

Teacher and student share identical task_head architecture (pruning only
touches backbone+neck; shared_conv is rebuilt to the same 64 output channels),
so teacher/student outputs have matching shapes — no adapter needed.

Direct comparison target: pruning.py ratio_25 -> 0.4081 mAP, same arch/init/
epochs/dataset, task-loss-only.

Usage:
    python EdgeFusion-CenterPoint/compression/distillation.py \
        --config $CFG \
        --checkpoint $CKPT \
        --ratio 0.25 \
        --epochs 5 \
        --batch-size 16 \
        --lr 4e-4 \
        --num-workers 16 \
        --alpha 1.0 \
        --beta 1.0

Eval only (after early stop):
    python EdgeFusion-CenterPoint/compression/distillation.py \
        --config $CFG \
        --eval-only \
        --model-path compression/results/distillation/ratio_25/distilled_25_epoch3.pth
"""

import argparse
import copy
import json
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from mmdet3d.apis import init_model
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from tqdm import tqdm

RESULTS_DIR = Path(__file__).parent / 'results' / 'distillation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── shared helpers (mirrors pruning.py) ────────────────────────────────────────

def _set_train_cfg_from_config(model: nn.Module, cfg: Config) -> None:
    """Construct pts_bbox_head.train_cfg from model config fields."""
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
    print(f'[distill] train_cfg set: out_size_factor={out_size_factor}')


def build_train_loader(cfg: Config, batch_size: int, num_workers: int = 8,
                       use_cbgs: bool = False):
    cfg = cfg.copy()
    if not use_cbgs and cfg.train_dataloader.dataset.type == 'CBGSDataset':
        cfg.train_dataloader.dataset = cfg.train_dataloader.dataset.dataset
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = num_workers
    return Runner.build_dataloader(cfg.train_dataloader)


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

    Weight VALUES from this rebuild are discarded — the student is trained
    from this point, so a random init for shared_conv is fine (it learns
    quickly, same as observed in pruning.py).
    """
    model.eval()
    with torch.no_grad():
        feats = model.pts_backbone(bev_example)
        neck_out = model.pts_neck(feats)
        new_in_ch = neck_out[0].shape[1]

    old_conv = model.pts_bbox_head.shared_conv.conv
    if old_conv.in_channels == new_in_ch:
        return

    new_conv = nn.Conv2d(
        new_in_ch, old_conv.out_channels,
        old_conv.kernel_size, old_conv.stride, old_conv.padding,
        bias=old_conv.bias is not None,
    ).to(bev_example.device)
    nn.init.kaiming_normal_(new_conv.weight)
    if new_conv.bias is not None:
        nn.init.zeros_(new_conv.bias)

    model.pts_bbox_head.shared_conv.conv = new_conv
    print(f'[distill] shared_conv rebuilt: {old_conv.in_channels} -> {new_in_ch} in_channels')


def prune_model(model: nn.Module, bev_example: torch.Tensor, ratio: float) -> None:
    """Apply L1-norm structured channel pruning to backbone + neck.

    Used here purely as a deterministic channel-selection + weight-inheritance
    INIT for the student — same selection as pruning.py ratio_25, giving the
    student FP32-inherited weights for kept channels before distillation
    training begins.
    """
    combined = BackboneNeck(model.pts_backbone, model.pts_neck)
    combined.eval()

    before = sum(p.numel() for p in model.pts_backbone.parameters())
    before += sum(p.numel() for p in model.pts_neck.parameters())
    print(f'[distill] student backbone+neck params before: {before/1e6:.2f}M')

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
    _rebuild_shared_conv(model, bev_example)

    total = sum(p.numel() for p in model.pts_backbone.parameters())
    total += sum(p.numel() for p in model.pts_neck.parameters())
    print(f'[distill] student backbone+neck params after:  {total/1e6:.2f}M '
          f'({100*total/before:.1f}% of original)')


def _merged_ema_state_dict(model: nn.Module, ema_model: nn.Module) -> dict:
    """Combine EMA-smoothed parameters with the live model's BN buffers.

    See pruning.py for the full explanation — ema_model.parameters() are
    EMA-tracked but BatchNorm running_mean/running_var are buffers and stay
    frozen at init. Using ema_model.state_dict() directly would overwrite
    model's properly-trained BN stats with stale values.
    """
    param_names = set(dict(model.named_parameters()).keys())
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    merged = {}
    for k, v in model_state.items():
        merged[k] = ema_state[k] if k in param_names else v
    return merged


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, cfg: Config) -> dict:
    """Evaluate using mmdet3d Runner evaluator — reads all paths from config."""
    from mmengine.evaluator import Evaluator as MMEval
    val_loader = Runner.build_dataloader(cfg.test_dataloader)
    evaluator = MMEval(cfg.test_evaluator)

    model.eval()
    evaluator.dataset_meta = val_loader.dataset.metainfo
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating distilled model'):
            data = model.data_preprocessor(batch, training=False)
            outputs = model(**data, mode='predict')
            evaluator.process(data_samples=outputs, data_batch=batch)

    metrics = evaluator.evaluate(len(val_loader.dataset))
    map_val = metrics.get('NuScenes metric/pred_instances_3d_NuScenes/mAP', 0.0)
    nds_val = metrics.get('NuScenes metric/pred_instances_3d_NuScenes/NDS', 0.0)
    return {'mAP': map_val, 'NDS': nds_val}


# ── distillation ────────────────────────────────────────────────────────────────

def build_teacher(cfg: Config, checkpoint: str) -> nn.Module:
    """Load frozen FP32 teacher."""
    teacher = init_model(cfg, checkpoint=checkpoint, device='cuda:0')
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def build_student(cfg: Config, checkpoint: str, ratio: float,
                  bev_example: torch.Tensor) -> nn.Module:
    """Build student: FP32 init -> L1-channel-selected pruning (architecture
    + weight inheritance for kept channels, same as pruning.py ratio_25)."""
    student = init_model(cfg, checkpoint=checkpoint, device='cuda:0')
    _set_train_cfg_from_config(student, cfg)
    prune_model(student, bev_example, ratio)
    return student


def distillation_loss(teacher_outs, student_outs, alpha: float, beta: float):
    """Heatmap + confidence-weighted regression distillation losses.

    teacher_outs / student_outs: output of pts_bbox_head.forward(pts_feats),
    a tuple of per-task results, each a 1-element list containing a dict with
    keys 'heatmap', 'reg', 'height', 'dim', 'rot', 'vel'. Teacher and student
    share identical task_head shapes (only backbone+neck were pruned), so no
    adapter is needed.
    """
    hm_loss = 0.0
    reg_loss = 0.0
    n_tasks = len(teacher_outs)
    reg_keys = ('reg', 'height', 'dim', 'rot', 'vel')

    for t_task, s_task in zip(teacher_outs, student_outs):
        t_dict = t_task[0]
        s_dict = s_task[0]

        t_hm = torch.sigmoid(t_dict['heatmap']).clamp(1e-4, 1 - 1e-4)
        s_hm = torch.sigmoid(s_dict['heatmap']).clamp(1e-4, 1 - 1e-4)
        hm_loss = hm_loss + F.mse_loss(s_hm, t_hm)

        # Teacher confidence (max over classes) — soft attention for reg distill
        weight = t_hm.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]

        for key in reg_keys:
            if key in t_dict and key in s_dict:
                diff = (s_dict[key] - t_dict[key]).abs()
                reg_loss = reg_loss + (diff * weight).mean()

    hm_loss = hm_loss / n_tasks
    reg_loss = reg_loss / n_tasks
    total = alpha * hm_loss + beta * reg_loss
    return total, hm_loss, reg_loss


def distill_train(
    student: nn.Module,
    teacher: nn.Module,
    train_loader,
    epochs: int,
    lr: float,
    alpha: float,
    beta: float,
    out_dir: Path,
    ratio: float,
    ema_decay: float = 0.999,
) -> None:
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    ema_model = copy.deepcopy(student)
    ema_model.eval()
    ratio_str = f'{int(ratio * 100):02d}'

    for epoch in range(1, epochs + 1):
        student.train()
        sums = {'task': 0.0, 'hm': 0.0, 'reg': 0.0, 'total': 0.0}
        pbar = tqdm(train_loader, desc=f'Distill epoch {epoch}/{epochs}')

        for batch in pbar:
            # Single preprocessing pass — teacher and student must see the
            # same (possibly augmented) input for distillation to be valid.
            data = student.data_preprocessor(batch, training=True)
            inputs = data['inputs']
            data_samples = data['data_samples']
            batch_input_metas = [ds.metainfo for ds in data_samples]

            with torch.no_grad():
                _, t_feats = teacher.extract_feat(inputs, batch_input_metas)
                t_outs = teacher.pts_bbox_head(t_feats)

            _, s_feats = student.extract_feat(inputs, batch_input_metas)
            s_outs = student.pts_bbox_head(s_feats)

            task_losses = student.pts_bbox_head.loss(s_feats, data_samples)
            L_task = sum(
                v for k, v in task_losses.items()
                if 'loss' in k and isinstance(v, torch.Tensor)
            )

            L_distill, L_hm, L_reg = distillation_loss(t_outs, s_outs, alpha, beta)
            loss = L_task + L_distill

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 35)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), student.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

            sums['task'] += L_task.item()
            sums['hm'] += L_hm.item()
            sums['reg'] += L_reg.item()
            sums['total'] += loss.item()
            pbar.set_postfix(
                task=f'{L_task.item():.3f}',
                hm=f'{L_hm.item():.4f}',
                reg=f'{L_reg.item():.3f}',
                lr=f'{scheduler.get_last_lr()[0]:.2e}',
            )

        n = len(train_loader)
        print(f'[distill] epoch {epoch}/{epochs}  '
              f'task {sums["task"]/n:.4f}  hm {sums["hm"]/n:.4f}  '
              f'reg {sums["reg"]/n:.4f}  total {sums["total"]/n:.4f}  '
              f'lr {scheduler.get_last_lr()[0]:.2e}')

        merged_state = _merged_ema_state_dict(student, ema_model)
        ckpt_path = out_dir / f'distilled_{ratio_str}_epoch{epoch}.pth'
        torch.save({
            'state_dict': merged_state,
            'ratio': ratio,
            'epoch': epoch,
            'task_loss': sums['task'] / n,
            'hm_loss': sums['hm'] / n,
            'reg_loss': sums['reg'] / n,
        }, ckpt_path)
        print(f'[distill] checkpoint saved: {ckpt_path.name}')

    student.load_state_dict(_merged_ema_state_dict(student, ema_model))
    print('[distill] EMA params + live BN buffers loaded for evaluation.')


# ── eval-only / main ────────────────────────────────────────────────────────────

def eval_checkpoint(args: argparse.Namespace) -> None:
    """Evaluate a saved distilled student checkpoint (.pt full model object)."""
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    print(f'[eval] Loading distilled model from {args.model_path}...')
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
    p = argparse.ArgumentParser(description='CenterPoint knowledge distillation')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', help='FP32 .pth checkpoint (teacher + student init)')
    p.add_argument('--ratio', type=float, default=0.25,
                   help='Student channel ratio — same architecture as pruning.py '
                        '(default 0.25 for direct comparison)')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--alpha', type=float, default=1.0, help='Heatmap distill weight')
    p.add_argument('--beta', type=float, default=1.0, help='Regression distill weight')
    p.add_argument('--use-cbgs', action='store_true', default=False,
                   help='Use CBGS dataset (default: raw, ~1.7 hrs/epoch at batch=16)')
    p.add_argument('--out', default=str(RESULTS_DIR))
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--model-path', type=str,
                   help='Path to distilled model checkpoint for --eval-only')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.eval_only:
        if not args.model_path:
            raise ValueError('--model-path required with --eval-only')
        eval_checkpoint(args)
        return

    if not args.checkpoint:
        raise ValueError('--checkpoint required for training')

    ratio_str = f'{int(args.ratio * 100):02d}'
    out_dir = Path(args.out) / f'ratio_{ratio_str}'
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))

    print('[distill] Loading frozen FP32 teacher...')
    teacher = build_teacher(cfg, args.checkpoint)

    train_loader = build_train_loader(cfg, args.batch_size, args.num_workers, args.use_cbgs)
    mode = 'CBGS' if args.use_cbgs else 'raw'
    print(f'[distill] Dataset: {mode} ({len(train_loader)} steps/epoch)')
    sample_batch = next(iter(train_loader))
    bev_example = get_bev_example(teacher, sample_batch)

    print('[distill] Building student (FP32 init -> L1 channel-selected pruning)...')
    student = build_student(cfg, args.checkpoint, args.ratio, bev_example)

    run_name = f'distillation_ratio_{ratio_str}'
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'ratio': args.ratio,
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'alpha': args.alpha,
            'beta': args.beta,
        })

        print(f'\n[distill] Training for {args.epochs} epochs '
              f'(alpha={args.alpha}, beta={args.beta})...')
        distill_train(student, teacher, train_loader, args.epochs, args.lr,
                      args.alpha, args.beta, out_dir, args.ratio)

        print('\n[distill] Evaluating...')
        metrics = evaluate(student, cfg)
        map_val = metrics['mAP']
        nds_val = metrics['NDS']

        print(f'\n[distill] ratio={args.ratio:.0%}  mAP {map_val:.4f}  NDS {nds_val:.4f}')
        print('[distill] (pruning.py ratio_25 baseline: mAP 0.4081, NDS 0.5382)')
        mlflow.log_metrics({'mAP': map_val, 'NDS': nds_val})

        ckpt_path = str(out_dir / f'distilled_{ratio_str}.pth')
        torch.save({'state_dict': student.state_dict(), 'ratio': args.ratio}, ckpt_path)

        model_path = str(out_dir / f'distilled_model_{ratio_str}.pt')
        torch.save(student, model_path)

        metrics_path = str(out_dir / f'distilled_{ratio_str}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({'ratio': args.ratio, 'mAP': map_val, 'NDS': nds_val,
                       'alpha': args.alpha, 'beta': args.beta}, f, indent=2)

        print(f'[distill] Saved checkpoint: {ckpt_path}')
        print(f'[distill] Saved full model: {model_path}')
        print(f'[distill] Metrics saved to {metrics_path}')


if __name__ == '__main__':
    main()
