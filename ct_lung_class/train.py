import os
from datetime import datetime
import sys
from typing import List, Tuple

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
from datasets import DatasetItem, NoduleDataset, getNoduleInfoList
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
from pretrained import create_pretrained_medical_resnet
from util.logconf import logging
from util.util import enumerateWithEstimate # importstr
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

        self.augmentation_dict = {
            augmentation: getattr(self.cli_args, augmentation)
            for augmentation in ["affine_prob", "translate", "scale", "padding"]
        }

        self.tag = self.cli_args.tag
        self.use_cuda = torch.cuda.is_available()
        self.logger = logger
        self.device = cli_args.device

    def init_model(self, model_path: str) -> torch.nn.Module:
        # model_cls = importstr(model_path)
        # model = model_cls(dropout_prob=0.4, spatial_dims=3, in_channels=1, out_channels=2)
        model, _ = create_pretrained_medical_resnet("/data/kaplinsp/resnet50_medicalnet.tar")
        if self.use_cuda:
            model = model.to(self.device)
        return model

    def init_optimizer(
        self, learn_rate: float, momentum: float, l2_reg: float
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        # optimizer = SGD(self.model.parameters(), lr=learn_rate, momentum=momentum, nesterov=True)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # return optimizer, scheduler
        return Adam(self.model.parameters(), lr=learn_rate, weight_decay=l2_reg)

    def init_train_dataloader(self, nodule_list: List[NoduleInfoTuple]) -> DataLoader:
        train_ds = NoduleDataset(
            nodule_info_list=nodule_list,
            isValSet_bool=False,
            augmentation_dict=self.augmentation_dict,
            dilate=self.cli_args.dilate,
            resample=self.cli_args.resample,
        )
        labels = np.array([s.is_nodule for s in nodule_list])
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sample_weights
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            sampler=sampler,
            drop_last=False,
        )

        return train_dl

    def init_val_dataloader(self, nodule_list: List[NoduleInfoTuple]) -> DataLoader:

        val_ds = NoduleDataset(
            nodule_info_list=nodule_list,
            isValSet_bool=True,
            dilate=self.cli_args.dilate,
            resample=self.cli_args.resample,
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
        model_output_dir = os.path.join(OUTPUT_PATH, self.run_dir)
        if not os.path.exists(model_output_dir):
            os.mkdir(model_output_dir)

        if not os.path.exists(LOG_DIR):
            os.mkdir(LOG_DIR)

        log_file = os.path.join(LOG_DIR, self.run_dir)
        file_handler = logging.FileHandler(log_file)
        self.logger.addHandler(file_handler)

    def assert_no_leak(
        self, train_dl: DataLoader, val_dl: DataLoader, dataset: List[NoduleInfoTuple]
    ):
        trains = set(
            f"{nod.file_path}{nod.center_lps}" for nod in train_dl.dataset.noduleInfo_list
        )
        vals = set(f"{nod.file_path}{nod.center_lps}" for nod in val_dl.dataset.noduleInfo_list)
        assert len(vals.intersection(trains)) == 0, "Data leak, overlapping train and val samples"
        assert len(val_dl.dataset.noduleInfo_list) + len(train_dl.dataset.noduleInfo_list) == len(
            dataset
        ), "Using all samples in dataset"

    def main(
        self,
        local_rank: int,
        dataset: List[Tuple[List[NoduleInfoTuple], List[NoduleInfoTuple]]],
        device_count: int,
    ):
        self.logger.info(f"Starting {type(self).__name__}, {self.cli_args}")
        self.device = self.device if self.device is not None else local_rank % device_count
        self.model = self.init_model(self.cli_args.model)
        # self.optimizer, self.scheduler = self.init_optimizer(
        #     self.cli_args.learn_rate, self.cli_args.momentum, self.cli_args.weight_decay
        # )
        self.optimizer = self.init_optimizer(
            self.cli_args.learn_rate, self.cli_args.momentum, self.cli_args.weight_decay
        )

        self.run_dir = f"log_{self.model._get_name()}_{self.time_str}.log"
        self.init_logs_outputs()
        train_set, test_set = dataset[local_rank]
        # print(train_set, test_set)
        train_dl = self.init_train_dataloader(train_set)
        val_dl = self.init_val_dataloader(test_set)
        self.assert_no_leak(train_dl, val_dl, train_set + test_set)
        executed = " ".join(sys.argv)
        self.writer = SummaryWriter(
            f"/data/kaplinsp/tensorboard/full-dataset/{self.time_str}_{self.tag}_{executed}_{local_rank}"
        )
        self.train(train_dl, val_dl)

    def train(self, train_dl, val_dl):
        # best_f1 = -1
        # best_accuracy = -1
        # best_loss = float("inf")
        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            self.logger.info(
                f"Epoch {epoch_ndx} of {self.cli_args.epochs}, {len(train_dl)}/{len(val_dl)} "
                f"batches of size {self.cli_args.batch_size}"
            )

            trnMetrics_t = self.train_epoch(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, "train", trnMetrics_t)

            valMetrics_t = self.validate(epoch_ndx, val_dl)
            self.log_metrics(epoch_ndx, "val", valMetrics_t)
            # self.scheduler.step()
            # it_val_metric = self.log_metrics(epoch_ndx, "val", valMetrics_t)

            # val_f1 = it_val_metric["pr/f1_score"]
            # val_accuracy = it_val_metric["correct/all"]
            # val_loss = it_val_metric["loss/all"]

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
            # loss_var.clip(1e-6)
            loss_var.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-6)
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
        (
            input_t,
            label_t,
            _,
        ) = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g = self.model(input_g)
        probability_g = torch.softmax(logits_g, dim=1)
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
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        metrics_dict["total_norm"] = total_norm**0.5
        for name, value in metrics_dict.items():
            self.writer.add_scalar(f"{name}/{mode_str}", value, epoch_ndx)

        return metrics_dict


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     mp.set_start_method("spawn")
#     parser.add_argument(
#         "--batch-size",
#         help="Batch size to use for training",
#         default=32,
#         type=int,
#     )
#     parser.add_argument(
#         "--num-workers",
#         help="Number of worker processes for background data loading",
#         default=4,
#         type=int,
#     )
#     parser.add_argument(
#         "--epochs",
#         help="Number of epochs to train for",
#         default=50,
#         type=int,
#     )
#     parser.add_argument(
#         "--balanced",
#         help="Balance the training data to half positive, half negative.",
#         action="store_true",
#         default=False,
#     )
#     parser.add_argument(
#         "--affine-prob", help="Probability of affine transform", type=float, default=0.75
#     )
#     parser.add_argument("--translate", help="translation amount", type=int, default=15)
#     parser.add_argument("--scale", help="scale amount", type=float, default=0.15)
#     parser.add_argument("--padding", help="augmentation padding mode", default="border")
#     parser.add_argument(
#         "--dilate",
#         type=int,
#         help="Dilation in MM",
#     )
#     parser.add_argument("--resample", type=int, help="resample size")
#     parser.add_argument(
#         "--finetune",
#         help="Start finetuning from this model.",
#         default="",
#     )

#     parser.add_argument(
#         "--finetune-depth",
#         help="Number of blocks (counted from head) to include in finetuning",
#         type=int,
#         default=1,
#     )
#     parser.add_argument(
#         "--k-folds", help="Number of cross-validation folds.", type=int, default=1, required=False
#     )
#     parser.add_argument("--learn-rate", help="Learn rate", type=float, default=1e-3)
#     parser.add_argument("--momentum", type=float, default=0.99)
#     parser.add_argument("--weight-decay", type=float, default=1e-4)
#     parser.add_argument("--model", type=str, required=True)
#     parser.add_argument("--device", required=True, type=int)
#     parser.add_argument("--tag", required=False, default="", help="Tag string for logging")
#     parser.add_argument("--val-ratio", required=False, type=float, default=0.2)
#     cli_args = parser.parse_args()
#     NoduleTrainingApp(cli_args).main()
