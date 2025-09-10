import os
import pickle
import torch
import torch.multiprocessing as mp

from train import NoduleTrainingApp
from parser import create_argument_parser
from data import DataManager
from datasets import getNoduleInfoList


def main():
    mp.set_start_method("spawn")
    parser = create_argument_parser()
    cli_args = parser.parse_args()
    

    
    # base_path = f"{cli_args.base_path}-" + "{fold_index}"
    # base_path = "/data/kaplinsp/models/log_DenseNet121_2025-04-15_22.17.11.log-fixedsize-no-resample-{fold_index}"
    # base_path = "/data/kaplinsp/models/log_DenseNet121_2025-05-05_10.25.50.log-dilate-find-epochs-{fold_index}" FROM ABSTRACT
    base_path =  "/data/kaplinsp/models/log_DenseNet121_2025-08-29_18.12.12.log-cross-val-zara-dataset-box-65-l2-.001-lr-1.5e-4-{fold_index}"
    run_data = []
    for i in range(5):
        path = base_path.format(fold_index=i)
        with open(os.path.join(path, "val.pkl"), 'rb') as f:
            val_ids = pickle.load(f)
        with open(os.path.join(path, "train.pkl"), 'rb') as f:
            train_ids = pickle.load(f)
            
        with open(os.path.join(path, "test.pkl"), 'rb') as f:
            test_ids = pickle.load(f)
        
        nodules = getNoduleInfoList(['zara'])
        train_nodules = [nod for nod in nodules if (nod.file_path, nod.center_lps) in val_ids + train_ids]
        test_nodules = [nod for nod in nodules if (nod.file_path, nod.center_lps)  in test_ids]
        run_data.append([train_nodules, test_nodules, []]) # both the val and test sets are dummies to work with the rest of the pipeline
    
    
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
