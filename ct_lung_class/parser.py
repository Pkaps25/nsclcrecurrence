import argparse


def create_argument_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--resample", type=int, nargs=3, help="resample size")
    parser.add_argument(
        "--k-folds", help="Number of cross-validation folds.", type=int, default=1, required=False
    )
    parser.add_argument("--learn-rate", help="Learn rate", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--model", type=str, choices=["densenet", "resnet", "luna", "medmnist18", "medmnist50"]
    )
    parser.add_argument("--tag", required=False, default="", help="Tag string for logging")
    parser.add_argument("--val-ratio", required=False, type=float, default=0.2)
    parser.add_argument("--device", required=False, type=int)
    parser.add_argument(
        "--dataset", help="dataset choices to add", nargs="+", choices=["prasad", "r17", "sclc"]
    )
    parser.add_argument("--oversample", help="Oversample during training", action="store_true")
    parser.add_argument(
        "--finetune-depth", help="Number of blocks from head to finetune", default=0, type=int
    )
    parser.add_argument("--box-size", help="Box size for fixed size boxes fallback", type=int)
    return parser
