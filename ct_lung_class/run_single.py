import torch.multiprocessing as mp

from train import NoduleTrainingApp
from parser import create_argument_parser
from data import DataManager


def main():
    mp.set_start_method("spawn")
    parser = create_argument_parser()
    cli_args = parser.parse_args()
    run_data = list(DataManager(cli_args.k_folds, cli_args.val_ratio, cli_args.dataset).split())
    NoduleTrainingApp(cli_args).main(cli_args.device, run_data, 1)


if __name__ == "__main__":
    main()
