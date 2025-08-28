from typing import Any
from lightning import LightningModule

import monai
import torch
import torchmetrics


class SCModel(LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SCModel")
        
        parser.add_argument("--learn-rate", help="Learn rate", type=float, default=1e-3)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"])
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument(
            "--model", type=str, choices=["densenet", "resnet", "luna", "medmnist18", "medmnist50", "dino"]
        )
        parser.add_argument("--tag", required=False, default="", help="Tag string for logging")
        return parent_parser

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()

        self.net = monai.networks.nets.densenet121(
                dropout_prob=self.hparams.dropout, 
                spatial_dims=3, 
                in_channels=1, 
                out_channels=2
            )
        
        self.loss_func = monai.losses.CrossEntropyLoss()

        self.metrics_dicts = torch.nn.ModuleDict({
            "train": self._init_metrics(),
            "val": self._init_metrics(),
        })

    def _init_metrics(self) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict({
            "acc_overall": torchmetrics.Accuracy(task="binary"),
            "acc_per_class": torchmetrics.Accuracy(task="multiclass", num_classes=2, average="none"),
            "precision": torchmetrics.Precision(task="binary"),
            "recall": torchmetrics.Recall(task="binary"),
            "f1": torchmetrics.F1Score(task="binary"),
            "auc": torchmetrics.AUROC(task="binary"),
        })
    
    def _update_metrics(self, stage: str, preds_probs: torch.Tensor, targets: torch.Tensor):
        """Update metrics for a given stage."""
        metrics = self.metrics_dicts[stage]
        preds_pos = preds_probs[:, 1]

        metrics["acc_overall"].update(preds_pos, targets)
        metrics["acc_per_class"].update(preds_probs, targets)
        metrics["precision"].update(preds_pos, targets)
        metrics["recall"].update(preds_pos, targets)
        metrics["f1"].update(preds_pos, targets)
        metrics["auc"].update(preds_pos, targets)
    
    def configure_optimizers(self):
        return {
            "optimizer": torch.optim.Adam(
                params=self.parameters(),
                lr=self.hparams.learn_rate,
                weight_decay=self.hparams.weight_decay,
            )
        }
    
    def _compute_and_log_metrics(self, stage: str):
        """Compute and log all metrics for a stage at epoch end."""
        metrics = self.metrics_dicts[stage]
        metrics_dict = {}

        # Compute each metric
        for name, metric in metrics.items():
            result = metric.compute()
            if name == "acc_per_class":
                metrics_dict[f"{stage}/correct/neg"] = result[0].item()
                metrics_dict[f"{stage}/correct/pos"] = result[1].item()
            else:
                metrics_dict[f"{stage}/{name}"] = result.item()

        # Log metrics
        self.log_dict(metrics_dict, prog_bar=True, sync_dist=True)

        # Reset metrics for next epoch
        for metric in metrics.values():
            metric.reset()

    def forward(self, x: torch.Tensor):
        return self.net(x) 

    def training_step(self, batch, batch_idx):
        x, y = batch 
        logits = self.forward(x)
        preds = torch.softmax(logits, dim=1)
        loss = self.loss_func(logits, y)

        self._update_metrics("train", preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def training_epoch_end(self, outputs):
        self._compute_and_log_metrics("train")


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        preds_probs = torch.softmax(logits, dim=1)
        loss = self.loss_func(logits, y)

        self._update_metrics("val", preds_probs, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        self._compute_and_log_metrics("val")
        
    

    