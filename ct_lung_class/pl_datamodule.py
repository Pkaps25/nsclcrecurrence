from typing import List, Optional
from lightning import LightningDataModule

from datasets import NoduleDataset
from torch.utils.data import DataLoader


class SCLCDataModule(LightningDataModule):
    
    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        parser.add_argument(
            "--batch-size",
            default=16,
            type=int,
        )
        parser.add_argument(
            "--dataset", 
            help="Datasets to use in training", 
            nargs="+", 
            choices=["prasad", "r17", "sclc", "zara"]
        )
        parser.add_argument("--val-ratio", required=False, type=float, default=0.15)
        parser.add_argument("--test-ratio", type=float)
        parser.add_argument(
            "--box-size", help="Box size for fixed size boxes fallback", type=int, nargs="+"
        )
        parser.add_argument("--fixed-size", help="Use a fixed size box", action="store_true")
        parser.add_argument(
            "--dilate",
            type=int,
            help="Dilation in MM",
        )
        
        parser.add_argument(
            "--affine-prob", help="Probability of affine transform", type=float, default=0.75
        )
        parser.add_argument("--translate", help="translation amount", type=int, default=15)
        parser.add_argument("--scale", help="scale amount", type=float, default=0.10)
        parser.add_argument("--padding", help="augmentation padding mode", default="border")
        
        parser.add_argument(
            "--k-folds", help="Number of cross-validation folds.", type=int, default=1, required=False
        )
        return parent_parser
    
    def __init__(
            self, 
            train_set: List,
            val_set: List,
            test_set: Optional[List] = None,
        ):
        super().__init__()
        self.save_hyperparameters()

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.augmentation_dict = {
            augmentation: getattr(self.hparams, augmentation)
            for augmentation in ["affine_prob", "translate", "scale", "padding"]
        }
        

    def setup(self, stage):

        self.train_ds = NoduleDataset(
            nodule_info_list=self.train_set,
            isValSet_bool=False,
            augmentation_dict=self.augmentation_dict,
            dilate=self.hparams.dilate,
            resample=self.hparams.resample,
            box_size=self.hparams.box_size,
            fixed_size=self.hparams.fixed_size,
        )

        self.val_ds = NoduleDataset(
            nodule_info_list=self.val_set,
            isValSet_bool=True,
            dilate=self.hparams.dilate,
            resample=self.hparams.resample,
            box_size=self.hparams.box_size,
            fixed_size=self.hparams.fixed_size,
        )

        if self.test_set:
            self.test_ds = NoduleDataset(
                nodule_info_list=self.test_ds,
                isValSet_bool=True,
                dilate=self.hparams.dilate,
                resample=self.hparams.resample,
                box_size=self.hparams.box_size,
                fixed_size=self.hparams.fixed_size,
            )

        if self.hparams.oversample:
            import numpy as np
            from torch.utils.data import WeightedRandomSampler
            labels = np.array([n.is_nodule for n in self.train_set])
            class_weights = 1.0 / np.bincount(labels)
            sample_weights = class_weights[labels]
            self.train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

        
    def train_dataloader(self):
        kwargs = {"sampler": self.train_sampler} if self.train_sampler else {"shuffle": True}
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=self.use_cuda,
            drop_last=False,
            **kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=self.use_cuda,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        if not self.test_ds:
            return None
        
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            pin_memory=self.use_cuda,
            drop_last=False,
            shuffle=False,
        )