import torch
import torch.multiprocessing as mp

from train import NoduleTrainingApp
from parser import create_argument_parser
from data import DataManager


def main():
    mp.set_start_method("spawn")
    parser = create_argument_parser()
    cli_args = parser.parse_args()
    run_data = list(DataManager(cli_args.k_folds, True, cli_args.val_ratio).split())
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
