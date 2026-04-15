"""PyTorch Lightning callback for Growt transfer auditing — V2."""

from __future__ import annotations

import logging
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from growt_client import (
    AuditResult,
    GrowtClient,
    MetricsResult,
    format_audit_report,
    format_training_trajectory,
)
from growt_nemo.extractor import extract_features

logger = logging.getLogger("growt_nemo")


class GrowtAuditCallback(pl.Callback):
    """Run Growt structural audit during and after training.

    Tracks audit results across epochs for trajectory analysis.
    Logs to TensorBoard and WandB when available.
    """

    def __init__(
        self,
        train_dataloader: Optional[DataLoader] = None,
        deploy_dataloader: Optional[DataLoader] = None,
        api_url: str = "https://api.transferoracle.ai",
        api_key: Optional[str] = None,
        layer_name: Optional[str] = None,
        fail_on_red_flag: bool = True,
        max_samples: int = 5000,
        audit_every_n_epochs: int = 0,
    ) -> None:
        super().__init__()
        self._train_dataloader = train_dataloader
        self._deploy_dataloader = deploy_dataloader
        self._client = GrowtClient(api_url=api_url, api_key=api_key)
        self._layer_name = layer_name
        self._fail_on_red_flag = fail_on_red_flag
        self._max_samples = max_samples
        self._audit_every_n = audit_every_n_epochs
        self._audit_history: list[tuple[int, AuditResult]] = []
        self._metrics_history: list[tuple[int, MetricsResult]] = []
        self._last_audit: Optional[AuditResult] = None
        self._last_metrics: Optional[MetricsResult] = None

    @property
    def last_audit(self) -> Optional[AuditResult]:
        return self._last_audit

    @property
    def last_metrics(self) -> Optional[MetricsResult]:
        return self._last_metrics

    @property
    def audit_history(self) -> list[tuple[int, AuditResult]]:
        return self._audit_history

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._train_dataloader is None:
            self._train_dataloader = trainer.train_dataloader

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._audit_every_n <= 0:
            return
        epoch = trainer.current_epoch
        if epoch > 0 and epoch % self._audit_every_n == 0:
            logger.info("[Growt] Periodic audit at epoch %d...", epoch)
            audit, metrics = self._run_audit(trainer, pl_module)
            self._audit_history.append((epoch, audit))
            if metrics:
                self._metrics_history.append((epoch, metrics))
            self._log_metrics(trainer, audit, metrics)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        logger.info("[Growt] Running final transfer audit...")
        audit, metrics = self._run_audit(trainer, pl_module)
        self._last_audit = audit
        self._last_metrics = metrics
        self._audit_history.append((trainer.current_epoch, audit))
        if metrics:
            self._metrics_history.append((trainer.current_epoch, metrics))

        # Rich console output
        print(format_audit_report(audit, metrics, title="GROWT TRAINING AUDIT"))
        if len(self._audit_history) > 1:
            print(format_training_trajectory(self._audit_history))

        self._log_metrics(trainer, audit, metrics)
        self._log_wandb(trainer, audit, metrics)

        if self._fail_on_red_flag and audit.diagnosis == "RED_FLAG":
            raise RuntimeError(
                f"[Growt] Model flagged as RED_FLAG — unsafe to deploy.\n{audit.report}"
            )

    def _run_audit(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
    ) -> tuple[AuditResult, Optional[MetricsResult]]:
        train_dl = self._train_dataloader
        deploy_dl = self._deploy_dataloader

        if deploy_dl is None:
            logger.info("[Growt] No deploy_dataloader — using validation set as proxy.")
            val_dls = trainer.val_dataloaders
            if val_dls:
                deploy_dl = val_dls[0] if isinstance(val_dls, list) else val_dls
            else:
                deploy_dl = train_dl

        train_feats, train_labels = extract_features(
            pl_module, train_dl, layer_name=self._layer_name, max_samples=self._max_samples,
        )
        deploy_feats, deploy_labels = extract_features(
            pl_module, deploy_dl, layer_name=self._layer_name, max_samples=self._max_samples,
        )

        val_acc = trainer.callback_metrics.get("val_accuracy") or trainer.callback_metrics.get("val_acc")
        val_accuracy = float(val_acc) if val_acc is not None else None

        audit = self._client.audit_transfer(
            features_train=train_feats.tolist(),
            labels_train=train_labels.tolist(),
            features_deploy=deploy_feats.tolist(),
            labels_deploy=deploy_labels.tolist(),
            val_accuracy=val_accuracy,
        )

        metrics = None
        if len(train_feats) == len(deploy_feats):
            metrics = self._client.metrics_compare(
                features_reference=train_feats.tolist(),
                features_compare=deploy_feats.tolist(),
                labels_reference=train_labels.tolist(),
            )

        return audit, metrics

    def _log_metrics(
        self, trainer: pl.Trainer, audit: AuditResult, metrics: Optional[MetricsResult],
    ) -> None:
        if not trainer.logger:
            return
        m = {
            "growt/transfer_oracle": audit.transfer_oracle or 0.0,
            "growt/coverage_pct": audit.coverage_pct or 0.0,
            "growt/safe_to_deploy": 1.0 if audit.safe_to_deploy else 0.0,
            "growt/n_flagged_samples": float(audit.n_flagged_samples),
        }
        if metrics:
            m["growt/sqnr_db"] = metrics.sqnr_db or 0.0
            m["growt/cosine_mean"] = metrics.cosine_mean or 0.0
            if metrics.rank_correlation is not None:
                m["growt/rank_correlation"] = metrics.rank_correlation
        trainer.logger.log_metrics(m, step=trainer.global_step)

        # Log matplotlib figures to TensorBoard
        self._log_figures(trainer, audit)

    def _log_figures(
        self, trainer: pl.Trainer, audit: AuditResult,
    ) -> None:
        """Log matplotlib figures to TensorBoard (per-class coverage, trajectory)."""
        if not trainer.logger:
            return
        try:
            from growt_client.visualizations import plot_per_class_coverage, plot_training_trajectory

            # Per-class coverage bar chart
            fig_coverage = plot_per_class_coverage(audit, title="Growt Per-Class Coverage")
            if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "add_figure"):
                trainer.logger.experiment.add_figure("growt/per_class_coverage", fig_coverage, trainer.global_step)
            import matplotlib.pyplot as plt
            plt.close(fig_coverage)

            # Training trajectory (if we have history)
            if len(self._audit_history) > 1:
                fig_trajectory = plot_training_trajectory(
                    self._audit_history,
                    self._metrics_history if self._metrics_history else None,
                    title="Growt Training Trajectory",
                )
                if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "add_figure"):
                    trainer.logger.experiment.add_figure("growt/training_trajectory", fig_trajectory, trainer.global_step)
                plt.close(fig_trajectory)

        except ImportError:
            logger.debug("[Growt] matplotlib not available — skipping figure logging")
        except Exception as e:
            logger.warning("[Growt] Figure logging failed: %s", e)

    def _log_wandb(
        self, trainer: pl.Trainer, audit: AuditResult, metrics: Optional[MetricsResult],
    ) -> None:
        if not trainer.logger or not hasattr(trainer.logger, "experiment"):
            return
        exp = trainer.logger.experiment
        if not hasattr(exp, "log"):
            return

        try:
            import wandb
            if audit.classes_at_risk:
                table = wandb.Table(
                    columns=["class", "at_risk"],
                    data=[[c, True] for c in audit.classes_at_risk],
                )
                exp.log({"growt/classes_at_risk": table})
            if audit.report:
                exp.log({"growt/report": wandb.Html(f"<pre>{audit.report}</pre>")})
        except ImportError:
            pass
