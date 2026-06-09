# MLflow Experiment Tracking

All compression experiments log to a local MLflow tracking server.

## Start the UI

```bash
cd /workspace/EdgeFusion-CenterPoint
mlflow ui --port 5000
# open http://localhost:5000
```

## Run naming convention

| Pattern | Script |
|---|---|
| `fp32_baseline` | `baseline/eval.py` |
| `ptq_int8` | `compression/ptq.py` |
| `sensitivity_analysis` | `compression/sensitivity.py` |
| `qat_bs{B}_ep{E}` | `compression/qat.py` |
| `prune_r{sparsity%}` | `compression/pruning.py` |
| `distill_ep{E}` | `compression/distillation.py` |
| `pareto` | `compression/pareto.py` |

## Key metrics

`mAP`, `NDS`, `loss`, `lr`, `epoch`,
`latency_p99_ms` *(added from Jetson benchmarks post-hoc)*,
`power_W`, `mj_per_frame`.
