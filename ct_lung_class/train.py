import os
from datetime import datetime
import pickle
import random
import sys
from typing import List, Optional, Tuple

from sklearn.metrics import auc, roc_curve
from datasets import DatasetItem
import monai

import numpy as np
import torch
import torch.nn as nn
from constants import (
    METRICS_LABEL_NDX,
    METRICS_LOSS_NDX,
    METRICS_PRED_NDX,
    METRICS_SIZE,
)
from image import NoduleInfoTuple
from datasets import NoduleDataset

# from medmnist_model import load_mendminst_resnet50, load_mendminst_resnet18
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, WeightedRandomSampler
from ct_lung_class.pretrained.medicalnet import create_pretrained_medical_resnet
from pretrained.luna_model import create_pretrained_luna
from dino_classifier import VolumeClassifier

# from luna_model import create_pretrained_luna
from util.logconf import logging
from util.util import enumerateWithEstimate  # importstr
from torch.utils.tensorboard import SummaryWriter
from conf import SETTINGS

# from monai.data import TestTimeAugmentation
from monai.losses import FocalLoss


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
world_size = torch.cuda.device_count()


# EPOCH_MAP = [2800, 2450, 3800, 2000, 3400] FOR ABSTRACT 
EPOCH_MAP = [700, 2000, 1600, 1400, 1800]

class NoduleTrainingApp:
    def __init__(self, cli_args):
        self.cli_args = cli_args
        self.time_str = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {
            augmentation: getattr(self.cli_args, augmentation)
            for augmentation in ["affine_prob", "translate", "scale", "padding"]
        }

        self.tag = self.cli_args.tag
        self.use_cuda = torch.cuda.is_available()
        self.logger = logger
        self.device = cli_args.device

    def init_model(self, model_path: str) -> torch.nn.Module:
        if model_path == "densenet":
            model = monai.networks.nets.densenet201(
                dropout_prob=self.cli_args.dropout, spatial_dims=3, in_channels=1, out_channels=2
            )
        elif model_path == "resnet":
            model, _ = create_pretrained_medical_resnet("/data/kaplinsp/resnet50_medicalnet.tar")
        elif model_path == "luna":
            model = create_pretrained_luna()
        elif model_path == "dino":
            model = VolumeClassifier()
        # elif model_path == "medmnist18":
        #     model = load_mendminst_resnet18(self.cli_args.finetune_depth)
        # elif model_path == "medmnist50":
        #     model = load_mendminst_resnet50(self.cli_args.finetune_depth)
        else:
            raise ValueError("Invalid model choice")

        if self.use_cuda:
            model = model.to(self.device)
        return model

    def init_optimizer(
        self, optimizer_str: str, learn_rate: float, momentum: float, l2_reg: float
    ) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        if optimizer_str == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=learn_rate,
                momentum=momentum,
                nesterov=True,
                weight_decay=l2_reg,
            )
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 1, 1e-6)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        elif optimizer_str == "adam":
            optimizer = Adam(self.model.parameters(), lr=learn_rate, weight_decay=l2_reg)
            scheduler = None
        else:
            raise ValueError("Invalid optimizer")

        return optimizer, scheduler

    def init_train_dataloader(self, nodule_list: List[NoduleInfoTuple]) -> DataLoader:
        train_ds = NoduleDataset(
            nodule_info_list=nodule_list,
            isValSet_bool=False,
            augmentation_dict=self.augmentation_dict,
            dilate=self.cli_args.dilate,
            resample=self.cli_args.resample,
            box_size=self.cli_args.box_size,
            fixed_size=self.cli_args.fixed_size,
        )
        labels = np.array([s.is_nodule for s in nodule_list])
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        self.class_weights = class_weights
        sample_weights = class_weights[labels]
        sample_weights
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        if self.cli_args.oversample:
            kwargs = {"sampler": sampler}
        else:
            kwargs = {"shuffle": True}

        train_dl = DataLoader(
            train_ds,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            drop_last=False,
            **kwargs,
        )

        return train_dl

    def init_val_dataloader(self, nodule_list: List[NoduleInfoTuple]) -> DataLoader:

        val_ds = NoduleDataset(
            nodule_info_list=nodule_list,
            isValSet_bool=True,
            dilate=self.cli_args.dilate,
            resample=self.cli_args.resample,
            box_size=self.cli_args.box_size,
            fixed_size=self.cli_args.fixed_size,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            drop_last=False,
            shuffle=True,
        )

        return val_dl

    def init_logs_outputs(self):
        model_output_dir = os.path.join(
            SETTINGS["model_dir"], f"{self.run_dir}-{self.tag}-{self.local_rank}"
        )
        self.model_output_dir = model_output_dir
        if not os.path.exists(model_output_dir):
            os.mkdir(model_output_dir)

        if not os.path.exists(SETTINGS["log_dir"]):
            os.mkdir(SETTINGS["log_dir"])

        log_file = os.path.join(SETTINGS["log_dir"], self.run_dir)
        file_handler = logging.FileHandler(log_file)
        self.logger.addHandler(file_handler)

    def assert_no_leak(
        self, train_dl: DataLoader, val_dl: DataLoader, test_set: List[NoduleInfoTuple], dataset: List[NoduleInfoTuple]
    ):
        trains = set(
            f"{nod.file_path}{nod.center_lps}" for nod in train_dl.dataset.noduleInfo_list
        )
        vals = set(f"{nod.file_path}{nod.center_lps}" for nod in val_dl.dataset.noduleInfo_list)
        tests = set(f"{nod.file_path}{nod.center_lps}" for nod in test_set)
        assert len(vals.intersection(trains)) == 0, "Data leak, overlapping train and val samples"
        assert len(vals.intersection(tests)) == 0, "Data leak, val overlapping tests"
        assert len(val_dl.dataset.noduleInfo_list) + len(train_dl.dataset.noduleInfo_list) + len(test_set) == len(
            dataset
        ), "Using all samples in dataset"

    def save_datasets(self, train_set: List[NoduleInfoTuple], val_set: List[NoduleInfoTuple], test_set: List[NoduleInfoTuple]):
        train_ids = [(nod.file_path, nod.center_lps) for nod in train_set]
        with open(os.path.join(self.model_output_dir, "train.pkl"), "wb") as f:
            pickle.dump(train_ids, f)

        val_ids = [(nod.file_path, nod.center_lps) for nod in val_set]
        with open(os.path.join(self.model_output_dir, "val.pkl"), "wb") as f:
            pickle.dump(val_ids, f)
            
        test_ids = [(nod.file_path, nod.center_lps) for nod in test_set]
        with open(os.path.join(self.model_output_dir, "test.pkl"), "wb") as f:
            pickle.dump(test_ids, f)

    def main(
        self,
        local_rank: int,
        dataset: List[Tuple[List[NoduleInfoTuple], List[NoduleInfoTuple]]],
        device_count: int,
    ):
        # self.cli_args.epochs = EPOCH_MAP[local_rank]
        print(self.cli_args.epochs)
        self.logger.info(f"Starting {type(self).__name__}, {self.cli_args}")
        self.device = self.device if self.device is not None else local_rank % device_count
        self.model = self.init_model(self.cli_args.model)
        self.optimizer, self.scheduler = self.init_optimizer(
            self.cli_args.optimizer,
            self.cli_args.learn_rate,
            self.cli_args.momentum,
            self.cli_args.weight_decay,
        )

        self.run_dir = f"log_{self.model._get_name()}_{self.time_str}.log"
        self.local_rank = local_rank
        self.init_logs_outputs()
        train_set, val_set, test_set = dataset[local_rank] if self.cli_args.k_folds > 1 else dataset[0]
        self.save_datasets(train_set, val_set, test_set)

        # print(train_set, test_set)
        train_dl = self.init_train_dataloader(train_set)
        val_dl = self.init_val_dataloader(val_set)
        self.assert_no_leak(train_dl, val_dl, test_set, train_set + val_set + test_set)
        executed = " ".join(sys.argv)
        self.writer = SummaryWriter(
            os.path.join(
                f"{SETTINGS['tensorboard_dir']}",
                f"{self.time_str}_{self.tag}_{local_rank}",
            )
        )
        self.train(train_dl, val_dl)

    def train(self, train_dl, val_dl):
        # best_f1 = -1
        # best_accuracy = -1
        # best_loss = float("inf")
        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            self.logger.info(
                f"Epoch {epoch_ndx} of {self.cli_args.epochs}, {len(train_dl)}/{len(val_dl)} "
                f"batches of size {self.cli_args.batch_size}, lr = {self.scheduler.get_last_lr() if self.scheduler else self.cli_args.learn_rate}"
            )

            trnMetrics_t = self.train_epoch(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, "train", trnMetrics_t)

            valMetrics_t = self.validate(epoch_ndx, val_dl)
            self.log_metrics(epoch_ndx, "val", valMetrics_t)
            if self.scheduler:
                self.scheduler.step(epoch_ndx)

            it_val_metric = self.log_metrics(epoch_ndx, "val", valMetrics_t)

            val_f1 = it_val_metric["pr/f1_score"]
            val_accuracy = it_val_metric["correct/all"]
            val_loss = it_val_metric["loss/all"]
            val_roc = it_val_metric["pr/auc_roc"]

            best_f1 = 0
            best_accuracy = (0,)
            best_loss = float("inf")
            best_roc = 0

            # if val_roc > best_roc:
            #     best_roc = val_roc
            #     torch.save(
            #         {
            #             "epoch": epoch_ndx,
            #             "model_state_dict": self.model.state_dict(),
            #             "optimizer_state_dict": self.optimizer.state_dict(),
            #         },
            #         os.path.join(self.model_output_dir, "best_auc_model.pth"),
            #     )

            # if val_f1 > best_f1:  # best val F1
            #     best_f1 = val_f1
            #     torch.save(
            #         {
            #             "epoch": epoch_ndx,
            #             "model_state_dict": self.model.state_dict(),
            #             "optimizer_state_dict": self.optimizer.state_dict(),
            #         },
            #         os.path.join(self.model_output_dir, "best_f1_model.pth"),
            #     )
            # if val_accuracy > best_accuracy:  # best val accuracy
            #     best_accuracy = val_accuracy
            #     torch.save(
            #         {
            #             "epoch": epoch_ndx,
            #             "model_state_dict": self.model.state_dict(),
            #             "optimizer_state_dict": self.optimizer.state_dict(),
            #         },
            #         os.path.join(self.model_output_dir, "best_accuracy_model.pth"),
            #     )
            # if val_loss < best_loss:  # best val loss
            #     best_loss = val_loss
            #     torch.save(
            #         {
            #             "epoch": epoch_ndx,
            #             "model_state_dict": self.model.state_dict(),
            #             "optimizer_state_dict": self.optimizer.state_dict(),
            #         },
            #         os.path.join(self.model_output_dir, "best_loss_model.pth"),
            #     )

            if epoch_ndx >= self.cli_args.epochs - 10:  # best val F1
                best_f1 = val_f1
                torch.save(
                    {
                        "epoch": epoch_ndx,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    os.path.join(self.model_output_dir, f"best_f1_model_{epoch_ndx}.pth"),
                )
            if epoch_ndx >= self.cli_args.epochs - 10:  # best val accuracy
                best_accuracy = val_accuracy
                torch.save(
                    {
                        "epoch": epoch_ndx,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    os.path.join(self.model_output_dir, f"best_accuracy_model_{epoch_ndx}.pth"),
                )
            if epoch_ndx >= self.cli_args.epochs - 10:  # best val loss
                best_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch_ndx,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    os.path.join(self.model_output_dir, f"best_loss_model_{epoch_ndx}.pth"),
                )

    def train_epoch(self, epoch_ndx: int, train_dl: DataLoader) -> torch.Tensor:
        self.model.train()
        train_dl.dataset.shuffleSamples()
        train_metrics = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            f"E{epoch_ndx} Training",
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            torch.autograd.set_detect_anomaly(True)
            loss_var = self.compute_batch_loss(
                batch_ndx, batch_tup, train_dl.batch_size, train_metrics, "train", epoch_ndx
            )
            # loss_var.clip(1e-6)
            loss_var.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-6)
            self.optimizer.step()
            # if self.scheduler:
            #     self.scheduler.step(epoch_ndx + batch_ndx / len(train_dl))

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return train_metrics.to("cpu")

    def validate(self, epoch_ndx: int, val_dl: DataLoader) -> torch.Tensor:
        with torch.no_grad():
            self.model.eval()
            val_metrics = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                f"E{epoch_ndx} Validation",
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, val_metrics, "val", epoch_ndx
                )

        return val_metrics.to("cpu")

    def compute_batch_loss(
        self,
        batch_ndx: int,
        batch_tup: DatasetItem,
        batch_size: int,
        metrics_g: torch.Tensor,
        mode_str: str,
        epoch_ndx,
    ) -> torch.Tensor:
        (
            input_t,
            label_t,
        ) = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g = self.model(input_g)
        probability_g = torch.softmax(logits_g, dim=1)
        loss_func = nn.CrossEntropyLoss()
        # loss_func = FocalLoss(to_onehot_y=True, alpha=0.75)
        loss_g = loss_func(logits_g, label_g)

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1]
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g

        return loss_g.mean()

    def log_metrics(
        self,
        epoch_ndx: int,
        mode_str: str,
        metrics: torch.Tensor,
        classification_threshold=0.5,
    ) -> dict:

        negative_label_mask = metrics[METRICS_LABEL_NDX] <= classification_threshold
        negative_predicted_mask = metrics[METRICS_PRED_NDX] <= classification_threshold

        positive_label_mask = ~negative_label_mask
        positive_predicted_mask = ~negative_predicted_mask

        neg_count = int(negative_label_mask.sum())
        pos_count = int(positive_label_mask.sum())

        neg_correct = int((negative_label_mask & negative_predicted_mask).sum())
        true_positive_count = pos_correct = int(
            (positive_label_mask & positive_predicted_mask).sum()
        )

        false_positive_count = neg_count - neg_correct
        false_negative_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict["loss/all"] = metrics[METRICS_LOSS_NDX].mean()
        metrics_dict["loss/neg"] = metrics[METRICS_LOSS_NDX, negative_label_mask].mean()
        metrics_dict["loss/pos"] = metrics[METRICS_LOSS_NDX, positive_label_mask].mean()

        metrics_dict["correct/all"] = np.divide(
            (pos_correct + neg_correct) * 100, metrics.shape[1], where=metrics.shape[1] != 0
        )
        metrics_dict["correct/neg"] = np.divide(neg_correct * 100, neg_count, where=neg_count != 0)
        metrics_dict["correct/pos"] = np.divide(pos_correct * 100, pos_count, where=pos_count != 0)

        precision = np.divide(
            true_positive_count,
            true_positive_count + false_positive_count,
            where=(true_positive_count + false_positive_count) != 0,
        )
        metrics_dict["pr/precision"] = precision

        recall = np.divide(
            true_positive_count,
            true_positive_count + false_negative_count,
            where=(true_positive_count + false_negative_count) != 0,
        )
        metrics_dict["pr/recall"] = recall

        metrics_dict["pr/f1_score"] = np.divide(
            2 * (precision * recall), precision + recall, where=(precision + recall) != 0
        )

        fpr, tpr, _ = roc_curve(
            metrics[METRICS_LABEL_NDX].detach().numpy(), metrics[METRICS_PRED_NDX].detach().numpy()
        )
        roc_auc = auc(fpr, tpr)
        metrics_dict["pr/auc_roc"] = roc_auc

        self.logger.info(
            (
                "E{} {:8} {loss/all:.8f} loss, "
                + "{correct/all:-5.1f}% correct, "
                + "{pr/precision:.4f} precision, "
                + "{pr/recall:.4f} recall, "
                + "{pr/f1_score:.4f} f1 score "
                + "{pr/auc_roc:.2f} auc-roc "
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        self.logger.info(
            ("E{} {:8}" + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})").format(
                epoch_ndx,
                mode_str + "_neg",
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        self.logger.info(
            ("E{} {:8}" + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})").format(
                epoch_ndx,
                mode_str + "_pos",
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )
        # total_norm = 0.0
        # for p in self.model.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # metrics_dict["total_norm"] = total_norm**0.5
        
        # if epoch_ndx % 5 == 0:
        for name, value in metrics_dict.items():
            self.writer.add_scalar(f"{name}/{mode_str}", value, epoch_ndx)

        return metrics_dict
