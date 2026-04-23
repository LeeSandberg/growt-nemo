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

## Status & Contributing

This is an early release to get the integration started. The code works but is not battle-tested in production yet. We welcome contributions:

- Bug fixes and improvements — PRs welcome
- New features and endpoint integrations
- Better error handling and edge cases
- Documentation improvements
- Test coverage

Open an issue or submit a PR on GitHub. All contributions must be compatible with the MPL-2.0 license.


## Related

- [Documentation](https://transferoracle.ai/growt/docs) — API reference, all plugins, tiers
- [growt-client](https://github.com/LeeSandberg/growt-client) — Python client (shared by all plugins)
- [growt-modelopt](https://github.com/LeeSandberg/growt-modelopt) — NVIDIA ModelOpt
- [growt-quark](https://github.com/LeeSandberg/growt-quark) — AMD Quark
- [growt-nemo](https://github.com/LeeSandberg/growt-nemo) — NeMo / PyTorch Lightning
- [growt-vllm](https://github.com/LeeSandberg/growt-vllm) — vLLM (NVIDIA + AMD)
- [growt-triton](https://github.com/LeeSandberg/growt-triton) — Triton Inference Server
- [growt-trt-validator](https://github.com/LeeSandberg/growt-trt-validator) — TensorRT validator
- [growt-tao](https://github.com/LeeSandberg/growt-tao) — TAO Toolkit
- [growt-huggingface](https://github.com/LeeSandberg/growt-huggingface) — HuggingFace TrainerCallback + Model Card
- [growt-wandb](https://github.com/LeeSandberg/growt-wandb) — W&B callback, artifact metadata, registry gate

