from typing import Any
from lightning import LightningModule

import monai
import torch
import torchmetrics


def get_model(model_name: str, hparams: Any):
    model_constructors = {
        "densenet121": monai.networks.nets.densenet121,
        "densenet201": monai.networks.nets.densenet201,
    }
    
    return model_constructors[model_name](
        dropout_prob=hparams.dropout,
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
    )

def get_loss_function(loss_name: str):
    if loss_name == "crossentropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_name == "focal":
        return monai.losses.FocalLoss(to_onehot_y=True, alpha=0.75)
    


class SCModel(LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SCModel")
        
        parser.add_argument("--learn-rate", help="Learn rate", type=float, default=1e-3)
        parser.add_argument("--dropout", type=float, default=0.4)
        parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"])
        parser.add_argument("--model", choices=["densenet121", "densenet201"])
        parser.add_argument("--loss", choices=["focal", "crossentropy"])
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        # parser.add_argument(
        #     "--model", type=str, choices=["densenet", "resnet", "luna", "medmnist18", "medmnist50", "dino"]
        # )
        return parent_parser

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()

        self.net = get_model(self.hparams.model, self.hparams)
        
        self.loss_func = get_loss_function(self.hparams.loss)
        self.warmup_epochs = 200

        self.metrics_dicts = torch.nn.ModuleDict({
            "train_metrics": self._init_metrics(),
            "val_metrics": self._init_metrics(),
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
        metrics = self.metrics_dicts[f"{stage}_metrics"]
        preds_pos = preds_probs[:, 1]

        metrics["acc_overall"].update(preds_pos, targets)
        metrics["acc_per_class"].update(preds_probs, targets)
        metrics["precision"].update(preds_pos, targets)
        metrics["recall"].update(preds_pos, targets)
        metrics["f1"].update(preds_pos, targets)
        metrics["auc"].update(preds_pos, targets)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learn_rate, 
            weight_decay=self.hparams.weight_decay
        )

        if not self.trainer or not hasattr(self.trainer, "train_dataloader"):
            raise ValueError("Trainer must be attached to compute scheduler properly.")

        # train_loader = self.trainer.train_dataloader
        steps_per_epoch = 13
        total_steps = self.hparams.max_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.05,   # start from 0
            end_factor=1.0,     # reach base_lr
            total_iters=warmup_steps,
        )

        # Cosine decay: base_lr → eta_min
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6,
        )
        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=200 * steps_per_epoch,
        #     eta_min=1e-6
        # )

        # Chain warmup then cosine
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
    
    def _compute_and_log_metrics(self, stage: str):
        """Compute and log all metrics for a stage at epoch end."""
        metrics = self.metrics_dicts[f"{stage}_metrics"]
        metrics_dict = {"step": self.current_epoch}

        # Compute each metric
        for name, metric in metrics.items():
            result = metric.compute()
            if name == "acc_per_class":
                metrics_dict[f"{stage}/correct/neg"] = result[0].item()
                metrics_dict[f"{stage}/correct/pos"] = result[1].item()
            else:
                metrics_dict[f"{stage}/{name}"] = result.item()

        # Log metrics
        self.log_dict(metrics_dict, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

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

        lr = self.lr_schedulers().get_last_lr()[0]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self, *args, **kwargs):
        self._compute_and_log_metrics("train")


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        preds_probs = torch.softmax(logits, dim=1)
        loss = self.loss_func(logits, y)

        self._update_metrics("val", preds_probs, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_end(self, *args, **kwargs):
        self._compute_and_log_metrics("val")
        
    

    