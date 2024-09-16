import argparse
from functools import cached_property
import itertools
from typing import Generator, List, Tuple
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.multiprocessing as mp

from datasets import getNoduleInfoList
from image import NoduleInfoTuple
from train import NoduleTrainingApp
from concurrent.futures import ProcessPoolExecutor, as_completed


def create_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        help="Batch size to use for training",
        default=16,
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
        "--affine-prob", help="Probability of affine transform", type=float, default=0.75
    )
    parser.add_argument("--translate", help="translation amount", type=int, default=15)
    parser.add_argument("--scale", help="scale amount", type=float, default=0.10)
    parser.add_argument("--padding", help="augmentation padding mode", default="border")
    parser.add_argument(
        "--dilate",
        type=int,
        help="Dilation in MM",
    )
    parser.add_argument("--resample", type=int, help="resample size")
    parser.add_argument(
        "--k-folds", help="Number of cross-validation folds.", type=int, default=1, required=False
    )
    parser.add_argument("--learn-rate", help="Learn rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="monai.networks.nets.densenet121")
    parser.add_argument("--tag", required=False, default="", help="Tag string for logging")
    parser.add_argument("--val-ratio", required=False, type=float, default=0.2)
    parser.add_argument("--device", required=False, type=int)
    return parser


class DataManager:

    def __init__(self, k_folds: int, stratify: bool, val_ratio: float) -> None:
        self.k_folds = k_folds
        self.val_ratio = val_ratio
        self.stratify = stratify

    @cached_property
    def nodule_info_list(self) -> List[NoduleInfoTuple]:
        return getNoduleInfoList()

    def split(self) -> Generator[Tuple[List[NoduleInfoTuple], List[NoduleInfoTuple]], None, None]:
        nods = [[nod] for nod in self.nodule_info_list]
        if self.k_folds == 1:
            kwargs = {"test_size": self.val_ratio}
            if self.stratify:
                labels = [nod.is_nodule for nod in self.nodule_info_list]
                kwargs["stratify"] = labels
            x_train, x_test = train_test_split(nods, **kwargs)
            yield list(itertools.chain(*x_train)), list(itertools.chain(*x_test))
        else:
            kfold = KFold(n_splits=self.k_folds, shuffle=True)
            splits = kfold.split(nods)
            for train_index, test_index in splits:
                yield [nods[i][0] for i in train_index], [nods[i][0] for i in test_index]


def main():
    mp.set_start_method("spawn")
    parser = create_argument_parser()
    cli_args = parser.parse_args()
    run_data = list(
        DataManager(cli_args.k_folds, False, cli_args.val_ratio).split()
    )  # stratify currently set to False
    num_devices = torch.cuda.device_count()

    mp.spawn(
        NoduleTrainingApp(cli_args).main,
        args=(
            run_data,
            num_devices,
        ),
        nprocs=len(run_data),
    )


if __name__ == "__main__":
    main()
