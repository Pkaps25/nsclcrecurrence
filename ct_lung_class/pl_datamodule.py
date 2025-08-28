from typing import Any
from lightning import LightningDataModule


class SCLCDataset(LightningDataModule):
    
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
    
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        