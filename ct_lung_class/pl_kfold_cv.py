import argparse
import os
from datetime import datetime
import logging
import torch
import torch.multiprocessing as mp

from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pl_datamodule import SCLCDataModule
from pl_train import SCModel
from data import DataManager
from conf import SETTINGS


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel K-Fold CV across devices")

    # Data + model args
    parser = SCLCDataModule.add_dataset_specific_args(parser)
    parser = SCModel.add_model_specific_args(parser)

    # Training args
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", required=True, help="Experiment tag")

    # CV args
    # parser.add_argument("--k-folds", type=int, default=5)
    # parser.add_argument("--val-ratio", type=float, default=0.15)
    # parser.add_argument("--test-ratio", type=float, default=0.15)

    # Hardware
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--devices", type=int, default=torch.cuda.device_count())

    return parser.parse_args()


def get_version_string(run_time, args, fold_idx):
    return (
        f"{run_time}_lr{args.learn_rate}_bs{args.batch_size}_"
        f"l2{args.weight_decay}_size{args.box_size[0]}_model{args.model}_"
        f"loss{args.loss}_dp{args.dropout}_fold{fold_idx}"
    )


def run_fold(local_rank, splits, args, run_time, num_devices):
    fold_idx = local_rank
    train_set, val_set, test_set = splits[fold_idx]
    device_id = fold_idx % num_devices  # round-robin assignment

    # Pin to GPU
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(device_id)

    seed_everything(args.seed + fold_idx)
    version_string = get_version_string(run_time, args, fold_idx)


    logger = logging.getLogger(f"lightning.fold{fold_idx}")
    log_file_path = os.path.join(SETTINGS["log_dir"], args.tag, f"{version_string}.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger.propagate = False        
    logger.handlers.clear()
    logger.addHandler(logging.FileHandler(log_file_path))

    model_output_dir = os.path.join(
        SETTINGS["model_dir"], args.tag, version_string
    )
    os.makedirs(model_output_dir, exist_ok=True)

    # DataModule
    dm = SCLCDataModule(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        **vars(args),
    )
    dm.setup()
    dm.save_datasets(model_output_dir)

    model = SCModel(**vars(args))

    checkpoint_loss = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        dirpath=model_output_dir,
        filename="{epoch:02d}-val_loss-{val/loss:.2f}",
        save_last=True,
    )
    checkpoint_acc = ModelCheckpoint(
        monitor="val/acc_overall",
        mode="max",  
        save_top_k=1,
        dirpath=model_output_dir,
        filename="{epoch:02d}-val_acc-{val/acc_overall:.3f}",
    )

    tb_logger = TensorBoardLogger(
        save_dir="/data/kaplinsp/tensorboard/test-cv/",
        name=args.tag,
        version=version_string,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=[device_id],
        # callbacks=[checkpoint_acc, checkpoint_loss],
        logger=tb_logger,
        log_every_n_steps=1,
    )

    trainer.logger.log_hyperparams({
        "dataset": args.dataset,
        "seed": args.seed,
        "lr": args.learn_rate,
        "batch_size": args.batch_size,
        "box_size": args.box_size,
        "l2": args.weight_decay,
        "model": args.model,
        "loss": args.loss,
        "dropout": args.dropout,
        "fold": fold_idx,
    })

    print(f"=== Training fold {fold_idx + 1}/{args.k_folds} on device {device_id} ===")
    trainer.fit(model, datamodule=dm)


def main():
    args = parse_args()
    run_time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    dm_data = DataManager(
        k_folds=args.k_folds,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        dataset_names=args.dataset,
    )
    splits = list(dm_data.split())
    num_devices = args.devices

    # mp.spawn will call run_fold(fold_idx, ...)
    mp.spawn(
        run_fold,
        args=(splits, args, run_time, num_devices),
        nprocs=args.k_folds,
    )


if __name__ == "__main__":
    main()
