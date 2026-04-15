# growt-nemo

**Growt structural audit callback for [NVIDIA NeMo](https://docs.nvidia.com/nemo-framework/) / PyTorch Lightning** — know when to stop training.

[![License: MPL-2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

> "Loss goes down. Growt shows you if structure goes up."

## What is this?

PyTorch Lightning callback that tracks structural quality across training epochs. Logs to TensorBoard and WandB with matplotlib visualizations.

## Install

```bash
pip install growt-nemo
```

## Quick Start

```python
from growt_nemo import GrowtAuditCallback

callback = GrowtAuditCallback(
    api_url="http://your-growt-api:8000",
    api_key="your-key",
    audit_every_n_epochs=5,  # Periodic audit during training
)

trainer = pl.Trainer(callbacks=[callback])
trainer.fit(model, train_dl)

# After training:
print(callback.last_audit.diagnosis)  # SAFE / RED_FLAG
```

## What You See

**TensorBoard:** growt/transfer_oracle, growt/coverage_pct, growt/sqnr_db as line charts + per-class coverage bar chart + training trajectory plot

**WandB:** All above + per-class risk table + full HTML report

**Console:** Rich formatted report with trajectory summary

## License

[MPL-2.0](LICENSE)
