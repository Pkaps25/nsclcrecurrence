import argparse
import itertools
import os
from datetime import datetime
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from constants import (
    METRICS_LABEL_NDX,
    METRICS_LOSS_NDX,
    METRICS_PRED_NDX,
    METRICS_SIZE,
)
from image import NoduleInfoTuple
from datasets import DatasetItem, NoduleDataset, getNoduleInfoList
from model import NoduleRecurrenceClassifier
from torch.optim import SGD
from torch.utils.data import DataLoader
from util.logconf import logging
from util.util import enumerateWithEstimate
from sklearn.model_selection import KFold, train_test_split
from torch.utils.tensorboard import SummaryWriter


LOG_DIR = "/home/kaplinsp/ct_lung_class/logs"
OUTPUT_PATH = "/home/kaplinsp/ct_lung_class/ct_lung_class/models/"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
world_size = torch.cuda.device_count()


class NoduleTrainingApp:
    def __init__(self, cli_args):
        self.cli_args = cli_args
        self.time_str = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict["flip"] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict["offset"] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict["scale"] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict["rotate"] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict["noise"] = 25.0

        self.tag = self.cli_args.tag
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:3" if self.use_cuda else "cpu")
        self.logger = logging.getLogger(__name__)
        self.cross_validate = True if self.cli_args.k_folds > 1 else False
        self.nodule_info_list = getNoduleInfoList()
        random.shuffle(self.nodule_info_list)
        if self.cross_validate:
            self.k_fold_splits = self.generate_k_folds(self.cli_args.k_folds)
        
    def generate_k_folds(self, n_splits: int):
        nodules = [[nod] for nod in self.nodule_info_list]
        kfold = KFold(n_splits=n_splits, shuffle=True)
        return kfold.split(nodules)
    
    def _train_test_split(
        self, 
        nodules: List[NoduleInfoTuple], 
        test_ratio: float,
    ) -> Tuple[List[NoduleInfoTuple], List[NoduleInfoTuple]]:
        nods = [[nod] for nod in nodules]
        labels = [nod.is_nodule for nod in nodules]
        x_train, x_test = train_test_split(nods, test_size=test_ratio)
        return list(itertools.chain(*x_train)), list(itertools.chain(*x_test))
            
        
    def init_model(self) -> torch.nn.Module:
        # model = monai.networks.nets.DenseNet(dropout_prob=0.5,spatial_dims=3,in_channels=1,out_channels=2, block_config=(3, 4, 8, 6))
        model = NoduleRecurrenceClassifier(
           dropout_prob=0.4, spatial_dims=3, in_channels=1, out_channels=2
        )
        # model = monai.networks.nets.ResNet(block="basic", layers=(3,4,6,3), block_inplanes=(64, 32, 16, 8), num_classes=2, n_input_channels=1)

        if self.cli_args.finetune:
            d = torch.load(self.cli_args.finetune, map_location="cpu")
            model_blocks = [
                n for n, subm in model.named_children() if len(list(subm.parameters())) > 0
            ]
            finetune_blocks = model_blocks[-self.cli_args.finetune_depth :]
            model.load_state_dict(
                {
                    k: v
                    for k, v in d["model_state"].items()
                    if k.split(".")[0] not in model_blocks[-1]
                },
                strict=False,
            )
            for n, p in model.named_parameters():
                if n.split(".")[0] not in finetune_blocks:
                    p.requires_grad_(False)

        if self.use_cuda:
            model = model.to(self.device)

        return model

    def init_optimizer(self) -> torch.optim.Optimizer:
        # return SGD(self.model.parameters(),lr=lr, weight_decay=1e-4)
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99, weight_decay=1e-4)

    def init_train_dataloader(self, nodule_list: List[NoduleInfoTuple]) -> DataLoader:
        train_ds = NoduleDataset(
            nodule_info_list=nodule_list,
            isValSet_bool=False,
            augmentation_dict=self.augmentation_dict,
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            drop_last=True,
            shuffle=True
        )

        return train_dl

    def init_val_dataloader(self, nodule_list: List[NoduleInfoTuple]) -> DataLoader:
        
        val_ds = NoduleDataset(
            nodule_info_list=nodule_list,
            isValSet_bool=True,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            drop_last=True,
            shuffle=True
        )

        return val_dl

    def init_logs_outputs(self):
        model_output_dir = os.path.join(OUTPUT_PATH, self.run_dir)
        if not os.path.exists(model_output_dir):
            os.mkdir(model_output_dir)

        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)

        log_file = os.path.join(LOG_DIR, self.run_dir)
        file_handler = logging.FileHandler(log_file)
        self.logger.addHandler(file_handler)
        
    def assert_no_leak(self, train_dl, val_dl):
        trains = set(f"{nod.file_path}{nod.center_lps}" for nod in train_dl.dataset.noduleInfo_list)
        vals = set(f"{nod.file_path}{nod.center_lps}" for nod in val_dl.dataset.noduleInfo_list)
        assert len(vals.intersection(trains)) == 0, "Data leak, overlapping train and val samples"
        assert len(val_dl.dataset.noduleInfo_list) + len(train_dl.dataset.noduleInfo_list) == len(self.nodule_info_list), "Using all samples in dataset"

    def main(self, *args):
        self.logger.info(f"Starting {type(self).__name__}, {self.cli_args}")

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

        self.run_dir = f"log_{self.model._get_name()}_{self.time_str}.log"
        self.writer = SummaryWriter(f"/data/kaplinsp/runs/{self.time_str}_{self.tag}")
        self.init_logs_outputs()

        train_set, test_set = self._train_test_split(self.nodule_info_list, self.cli_args.val_ratio)
        if not self.cross_validate:
            train_dl = self.init_train_dataloader(train_set)
            val_dl = self.init_val_dataloader(test_set)
            self.assert_no_leak(train_dl, val_dl)
            self.train(train_dl, val_dl)
            return 
        
        # """ K FOLD VALIDATION"""
        # for split_idx, (train_idx, val_idx) in enumerate(self.k_fold_splits):
        #     self.run_dir = f"log_{self.model._get_name()}_{self.time_str}.log/split_{split_idx}"
        #     if not os.path.exists(os.path.join(OUTPUT_PATH, self.run_dir)):
        #         os.mkdir(os.path.join(OUTPUT_PATH, self.run_dir))
            
        #     self.logger.info(f"Starting cross validation split {split_idx}")
        #     self.model = self.init_model()
        #     self.optimizer = self.init_optimizer()
            
        #     train_dl = self.init_train_dataloader(train_idx=train_idx)
        #     val_dl = self.init_val_dataloader(val_idx=val_idx)
        #     self.train(train_dl, val_dl)

    def train(self, train_dl, val_dl):
        best_f1 = -1
        best_accuracy = -1
        best_loss = float("inf")
        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            self.logger.info(
                f"Epoch {epoch_ndx} of {self.cli_args.epochs}, {len(train_dl)}/{len(val_dl)} batches of size {self.cli_args.batch_size}"
            )

            trnMetrics_t = self.train_epoch(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, "train", trnMetrics_t)

            valMetrics_t = self.validate(epoch_ndx, val_dl)
            it_val_metric = self.log_metrics(epoch_ndx, "val", valMetrics_t)

            val_f1 = it_val_metric["pr/f1_score"]
            val_accuracy = it_val_metric["correct/all"]
            val_loss = it_val_metric["loss/all"]
            
            # if val_f1 > best_f1:  # best val F1
            #     best_f1 = val_f1
            #     torch.save(
            #         {
            #             "epoch": epoch_ndx,
            #             "model_state_dict": self.model.state_dict(),
            #             "optimizer_state_dict": self.optimizer.state_dict(),
            #         },
            #         os.path.join(OUTPUT_PATH, self.run_dir, "best_f1_model.pth"),
            #     )
            # if val_accuracy > best_accuracy:  # best val accuracy
            #     best_accuracy = val_accuracy
            #     torch.save(
            #         {
            #             "epoch": epoch_ndx,
            #             "model_state_dict": self.model.state_dict(),
            #             "optimizer_state_dict": self.optimizer.state_dict(),
            #         },
            #         os.path.join(OUTPUT_PATH, self.run_dir, "best_accuracy_model.pth"),
            #     )
            # if val_loss < best_loss:  # best val loss
            #     best_loss = val_loss
            #     torch.save(
            #         {
            #             "epoch": epoch_ndx,
            #             "model_state_dict": self.model.state_dict(),
            #             "optimizer_state_dict": self.optimizer.state_dict(),
            #         },
            #         os.path.join(OUTPUT_PATH, self.run_dir, "best_loss_model.pth"),
            #     )
                
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
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                train_metrics,
            )

            loss_var.backward()
            self.optimizer.step()

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
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    val_metrics,
                )

        return val_metrics.to("cpu")

    def compute_batch_loss(
        self, batch_ndx: int, batch_tup: DatasetItem, batch_size: int, metrics_g: torch.Tensor
    ) -> torch.Tensor:
        input_t, label_t, _,  = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g = self.model(input_g)
        foo = (
            np.exp(logits_g.detach().cpu().numpy())
            / np.exp(logits_g.detach().cpu().numpy()).sum(1)[:, None]
        )
        probability_g = torch.tensor(foo)
        # probability_g=torch.exp(logits_g)/torch.sum(torch.exp(logits_g), 1)[:, None]

        # HINGE LOSS
        # loss_func=nn.HingeEmbeddingLoss()
        # loss_g=loss_func(logits_g,label_g)

        # CROSS ENTROPY LOSS
        loss_func = nn.CrossEntropyLoss()
        loss_g = loss_func(logits_g, label_g[:, 1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1]
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

        metrics_dict["correct/all"] = (pos_correct + neg_correct) / metrics.shape[1] * 100
        metrics_dict["correct/neg"] = (neg_correct) / neg_count * 100
        metrics_dict["correct/pos"] = (pos_correct) / pos_count * 100

        precision = metrics_dict["pr/precision"] = true_positive_count / np.float32(
            true_positive_count + false_positive_count
        )
        recall = metrics_dict["pr/recall"] = true_positive_count / np.float32(
            true_positive_count + false_negative_count
        )

        metrics_dict["pr/f1_score"] = 2 * (precision * recall) / (precision + recall)

        self.logger.info(
            (
                "E{} {:8} {loss/all:.8f} loss, "
                + "{correct/all:-5.1f}% correct, "
                + "{pr/precision:.4f} precision, "
                + "{pr/recall:.4f} recall, "
                + "{pr/f1_score:.4f} f1 score"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        self.logger.info(
            (
                "E{} {:8} {loss/neg:.8f} loss, "
                + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + "_neg",
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        self.logger.info(
            (
                "E{} {:8} {loss/pos:.8f} loss, "
                + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + "_pos",
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )
        for name, value in metrics_dict.items():
            self.writer.add_scalar(f"{name}/{mode_str}", value, epoch_ndx)

        return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mp.set_start_method('spawn')
    parser.add_argument(
        "--batch-size",
        help="Batch size to use for training",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--num-workers",
        help="Number of worker processes for background data loading",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--epochs",
        help="Number of epochs to train for",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--balanced",
        help="Balance the training data to half positive, half negative.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--augmented",
        help="Augment the training data.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--augment-flip",
        help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--augment-offset",
        help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--augment-scale",
        help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--augment-rotate",
        help="Augment the training data by randomly rotating the data around the head-foot axis.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--augment-noise",
        help="Augment the training data by randomly adding noise to the data.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--finetune",
        help="Start finetuning from this model.",
        default="",
    )

    parser.add_argument(
        "--finetune-depth",
        help="Number of blocks (counted from head) to include in finetuning",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--k-folds",
        help="Number of cross-validation folds.",
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Tag string for logging"
    )
    parser.add_argument(
        "--val-ratio",
        required=False,
        type=float,
        default=0.25
    )
    cli_args = parser.parse_args()
    NoduleTrainingApp(cli_args).main()
