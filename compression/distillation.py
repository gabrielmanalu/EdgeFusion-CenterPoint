"""
Knowledge distillation — full CenterPoint (teacher) to compact student.

Teacher:  open-mmlab FP32 CenterPoint (frozen).
Student:  smaller-capacity CenterPoint (centerpoint_tiny architecture class).

Loss = alpha * task_loss(student)
     + beta  * logit_distillation(teacher, student)
     + gamma * feature_distillation(teacher_neck, student_neck)

Usage:
    python compression/distillation.py \
        --teacher-cfg  configs/centerpoint_pillar02_circlenms_nus.py \
        --teacher-ckpt /workspace/data/centerpoint/centerpoint_nuscenes.pth \
        --student-cfg  configs/centerpoint_tiny_nus.py \
        --epochs 20
"""

import argparse

import mlflow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-cfg", required=True)
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--student-cfg", required=True)
    p.add_argument(
        "--student-ckpt",
        default=None,
        help="Optional warm-start checkpoint for the student",
    )
    p.add_argument("--alpha", type=float, default=1.0, help="Task loss weight")
    p.add_argument("--beta", type=float, default=2.0, help="Logit distill weight")
    p.add_argument("--gamma", type=float, default=1.0, help="Feature distill weight")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", default="compression/results/distillation/")
    return p.parse_args()


def build_teacher(cfg_path: str, ckpt_path: str):
    """Load and freeze the teacher model (no gradient updates)."""
    # TODO: load via mmdet3d API; set requires_grad=False for all params
    raise NotImplementedError


def build_student(cfg_path: str, ckpt_path):
    """Load or randomly initialise the student model."""
    # TODO: load via mmdet3d API; if ckpt_path is None, random init
    raise NotImplementedError


def distillation_loss(
    teacher_out, student_out, alpha: float, beta: float, gamma: float
):
    """Combined task + logit + feature distillation loss."""
    # TODO: implement each component
    raise NotImplementedError


def main() -> None:
    args = parse_args()

    with mlflow.start_run(run_name=f"distill_ep{args.epochs}"):
        mlflow.log_params(
            {
                "method": "distillation",
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
                "epochs": args.epochs,
            }
        )
        # TODO: build_teacher → build_student → training loop → eval → save
        raise NotImplementedError


if __name__ == "__main__":
    main()
