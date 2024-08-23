import argparse
import itertools
import os
from typing import Tuple 
from sklearn.model_selection import train_test_split
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from datasets import NoduleDataset, getNoduleInfoList
from train import NoduleTrainingApp


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",
        help="Batch size to use for training",
        default=32,
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
    return parser

def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def ddp_cleanup():
    dist.destroy_process_group()

class MainRunner:
    def __init__(self, cli_args):
        self.cli_args = cli_args
        self.use_cuda = torch.cuda.is_available()
    
    def main(self, rank: int, world_size: int, cli_args: dict, train_data: NoduleDataset, test_data: NoduleDataset):
        ddp_setup(rank, world_size)
        train_dl = DataLoader(
            train_data,
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=False,
            sampler=DistributedSampler(train_data, rank=rank)
        )
        
        val_dl = DataLoader(
                test_data,
                batch_size=self.cli_args.batch_size,
                num_workers=self.cli_args.num_workers,
                pin_memory=True,
                drop_last=False,
                shuffle=False,
                sampler=DistributedSampler(test_data, rank=rank)
            )
        trainer = NoduleTrainingApp(rank, world_size, train_dl, val_dl, cli_args)
        trainer.train(train_dl, val_dl)
        ddp_cleanup
    
class DataManager:
    def __init__(self, cli_args) -> None:
        self.cli_args = cli_args
        self.use_cuda = torch.cuda.is_available()
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
    

    def create_train_test_split(self, test_ratio: float) -> Tuple[DataLoader, DataLoader]:
        nodules = getNoduleInfoList()
        nods = [[nod] for nod in nodules]
        labels = [nod.is_nodule for nod in nodules]
        x_train, x_test = train_test_split(nods, test_size=test_ratio)
        train_ds = NoduleDataset(
                nodule_info_list=list(itertools.chain(*x_train)),
                isValSet_bool=False,
                augmentation_dict=self.augmentation_dict,
            )
        
        val_ds = NoduleDataset(
            nodule_info_list=list(itertools.chain(*x_test)),
            isValSet_bool=True,
        )

        
        
        return train_ds, val_ds



if __name__ == "__main__":
    parser = create_argument_parser()
    cli_args = parser.parse_args()
    data_manager = DataManager(cli_args)
    train_data, test_data = data_manager.create_train_test_split(cli_args.val_ratio)
    world_size = 2
    mp.spawn(MainRunner(cli_args).main, args=(world_size, cli_args, train_data, test_data), nprocs=world_size)