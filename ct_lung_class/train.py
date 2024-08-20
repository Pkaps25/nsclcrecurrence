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
from datatsets_peter import NoduleInfoTuple
from datasets import DatasetItem, NoduleDataset, getNoduleInfoList
from model import NoduleRecurrenceClassifier
from torch.optim import SGD
from torch.utils.data import DataLoader
from util.logconf import logging
from util.util import enumerateWithEstimate
from sklearn.model_selection import KFold, train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


LOG_DIR = "/home/kaplinsp/ct_lung_class/logs"
OUTPUT_PATH = "/home/kaplinsp/ct_lung_class/ct_lung_class/models/"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
world_size = torch.cuda.device_count()


class NoduleTrainingApp:
    def __init__(self, rank, world_size, train_data, test_data, cli_args):
        self.cli_args = cli_args
        self.time_str = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        self.totalTrainingSamples_count = 0
        self.tag = self.cli_args.tag
        self.rank = rank 
        self.world_size = world_size
        self.device = rank
        self.logger = logging.getLogger(__name__)
    
        self.train_dl = train_data
        self.val_dl = test_data
        
        self.logger.info(f"Starting training on device {self.rank}")

        self.logger.info(f"Init model on {self.rank}")
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

        self.run_dir = f"log_{self.model._get_name()}_{self.time_str}.log"
        self.writer = SummaryWriter(f"/data/kaplinsp/ddp_runs/")
        self.init_logs_outputs()
        self.logger.info(f"Done with init on {self.rank}")

        # self.assert_no_leak(self.train_dl, self.val_dl)
        # self.train(self.train_dl, self.val_dl)
    
    
    def init_model(self) -> torch.nn.Module:
        model = NoduleRecurrenceClassifier(
           dropout_prob=0.4, spatial_dims=3, in_channels=1, out_channels=2
        )
        model = DDP(model.to(self.rank), device_ids=[self.rank])

        return model

    def init_optimizer(self) -> torch.optim.Optimizer:
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99, weight_decay=1e-4)

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
        self.logger.info(f"Starting training on device {self.rank}")

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

        self.run_dir = f"log_{self.model._get_name()}_{self.time_str}.log"
        self.writer = SummaryWriter(f"/data/kaplinsp/ddp_runs/")
        self.init_logs_outputs()

        self.assert_no_leak(self.train_dl, self.val_dl)
        self.train(self.train_dl, self.val_dl)
         
        

    def train(self, train_dl, val_dl):
        self.logger.info("In train fn")
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
        self.train_dl.sampler.set_epoch(epoch_ndx)
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
        self.val_dl.sampler.set_epoch(epoch_ndx)
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