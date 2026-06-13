# MLflow Experiment Tracking

All compression experiments log to a local MLflow tracking server.

## Start the UI

```bash
cd /workspace/EdgeFusion-CenterPoint
mlflow ui --port 5000
# open http://localhost:5000
```

## Run naming convention

| Pattern                   | Script                        | Example                 |
| ------------------------- | ----------------------------- | ----------------------- |
| `fp32_baseline`           | `baseline/eval.py`            | `fp32_baseline`         |
| `ptq_int8_calib{N}`       | `compression/ptq.py`          | `ptq_int8_calib512`     |
| `sensitivity_{N}samp`     | `compression/sensitivity.py`  | `sensitivity_200samp`   |
| `qat_ep{E}_bs{B}`         | `compression/qat.py`          | `qat_ep5_bs4`           |
| `pruning_ratio_{NN}`      | `compression/pruning.py`      | `pruning_ratio_25`      |
| `distillation_ratio_{NN}` | `compression/distillation.py` | `distillation_ratio_25` |

`pareto.py` does not log to MLflow — it's a pure data/plotting script reading the
metrics JSONs each stage above writes to `compression/results/*/`.

## Key metrics (logged per script)

| Script          | Per-step (step=epoch) | Final                                                   |
| --------------- | --------------------- | ------------------------------------------------------- |
| ptq.py          | —                     | `ptq_mAP`, `ptq_NDS`, `mAP_drop`, `NDS_drop`            |
| sensitivity.py  | —                     | `fp32_ref_loss`, `n_sensitive_nodes`                    |
| qat.py          | `loss`, `lr`          | `qat_mAP`, `qat_NDS`, `recovery_vs_ptq`, `drop_vs_fp32` |
| pruning.py      | — (console only)      | `mAP`, `NDS`                                            |
| distillation.py | — (console only)      | `mAP`, `NDS`                                            |

pruning.py and distillation.py print per-epoch loss breakdowns to console
(`task`/`hm`/`reg` for distillation) but do not log them to MLflow — only the final
eval `mAP`/`NDS` are tracked as MLflow metrics for these two scripts.

Jetson benchmark metrics (`latency_p99_ms`, `power_W`, `mj_per_frame`) are not yet
implemented — to be added post-hoc once the P3 TRT benchmark script exists.
