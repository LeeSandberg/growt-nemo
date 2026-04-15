#!/usr/bin/env python3
"""Example: Adding Growt transfer audit to a NeMo / PyTorch Lightning training run.

Features demonstrated:
  - Automatic transfer audit at end of training with rich console report
  - Periodic audits during validation (every 2 epochs)
  - Training trajectory tracking across epochs
  - TensorBoard scalar logging (oracle, coverage, SQNR, flags)
  - WandB logging (per-class risk table, HTML report, trajectory table)
  - Programmatic access to last_audit and last_metrics after training
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from growt_nemo import GrowtAuditCallback


# ------------------------------------------------------------------ #
# 1.  Define a toy model (replace with your NeMo model)              #
# ------------------------------------------------------------------ #
class TinyClassifier(pl.LightningModule):
    """Minimal two-layer classifier for demonstration."""

    def __init__(
        self, input_dim: int = 128, hidden_dim: int = 64, num_classes: int = 10
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.head(self.encoder(x))

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:  # type: ignore[override]
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:  # type: ignore[override]
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ------------------------------------------------------------------ #
# 2.  Create dummy data (replace with your actual datasets)          #
# ------------------------------------------------------------------ #
def make_dataloaders(
    n_train: int = 500,
    n_val: int = 100,
    input_dim: int = 128,
    num_classes: int = 10,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    x_train = torch.randn(n_train, input_dim)
    y_train = torch.randint(0, num_classes, (n_train,))
    x_val = torch.randn(n_val, input_dim)
    y_val = torch.randint(0, num_classes, (n_val,))

    train_dl = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)
    return train_dl, val_dl


# ------------------------------------------------------------------ #
# 3.  Train with Growt audit callback                                #
# ------------------------------------------------------------------ #
def main() -> None:
    train_dl, val_dl = make_dataloaders()
    model = TinyClassifier()

    # --- The key addition: attach GrowtAuditCallback ---
    growt_callback = GrowtAuditCallback(
        api_url="https://api.transferoracle.ai",  # or "http://localhost:8000"
        api_key=None,                              # set your API key here or via env
        fail_on_red_flag=True,
        extract_layer=None,                        # auto-detect penultimate layer
        max_samples=5000,
        # NEW: periodic audits during validation
        periodic_audit=True,
        audit_every_n_epochs=2,                    # audit every 2nd validation epoch
        # deploy_dataloader=deploy_dl,             # optional: real deployment data
    )

    # --- Option A: TensorBoard logger (default) ---
    tb_logger = pl.loggers.TensorBoardLogger("tb_logs", name="growt_demo")

    # --- Option B: WandB logger (install wandb, pip install growt-nemo[wandb]) ---
    # wandb_logger = pl.loggers.WandbLogger(project="growt-demo", name="nemo-run")

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[growt_callback],
        enable_checkpointing=False,
        logger=tb_logger,  # swap to wandb_logger for WandB
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # ------------------------------------------------------------------ #
    # 4.  Inspect results after training                                 #
    # ------------------------------------------------------------------ #
    # The callback prints a rich report automatically. For programmatic access:
    audit = growt_callback.last_audit
    metrics = growt_callback.last_metrics

    if audit is not None:
        print(f"\n--- Programmatic Access ---")
        print(f"Diagnosis:       {audit.diagnosis}")
        print(f"Safe to deploy:  {audit.safe_to_deploy}")
        print(f"Transfer oracle: {audit.transfer_oracle}")
        print(f"Coverage:        {audit.coverage_pct}")

    if metrics is not None:
        print(f"SQNR:            {metrics.sqnr_db} dB")
        print(f"Cosine mean:     {metrics.cosine_mean}")
        print(f"Rank corr.:      {metrics.rank_correlation}")

    # Access full audit history (epoch, AuditResult) tuples
    print(f"\nAudit history: {len(growt_callback._audit_history)} periodic audits recorded")
    for epoch, result in growt_callback._audit_history:
        print(f"  Epoch {epoch}: {result.diagnosis} (oracle={result.transfer_oracle})")


if __name__ == "__main__":
    main()
