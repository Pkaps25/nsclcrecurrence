import argparse
from datetime import datetime
import logging
import os
from pl_datamodule import SCLCDataModule
from pl_train import SCModel

from data import DataManager
from conf import SETTINGS

from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer, seed_everything

def parse_args():
    parser = argparse.ArgumentParser(description="Train SCModel on Nodule dataset")

    # Add data module args
    parser = SCLCDataModule.add_dataset_specific_args(parser)
    # Add model args
    parser = SCModel.add_model_specific_args(parser)

    # Other training args
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", nargs="+", default=[0], type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", required=True, help="Experiment tag")

    return parser.parse_args()

def get_version_string(run_time, args):
    return (
        f"{run_time}_lr{args.learn_rate}_bs{args.batch_size}_"
        f"l2{args.weight_decay}_size{args.box_size}_model{args.model}_"
        f"loss{args.loss}_dp{args.dropout}_bs{args.batch_size}"
    )


def main():
    args = parse_args()
    seed_everything(args.seed)

    run_time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    version_string = get_version_string(run_time, args)
    
    
    logger = logging.getLogger("lightning.pytorch")
    log_file_path = os.path.join(SETTINGS["log_dir"], args.tag, f"{version_string}.log")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logger.addHandler(logging.FileHandler(log_file_path))

    # --- Step 1: Prepare dataset splits ---
    dm_data = DataManager(
        k_folds=1,  # single split
        val_ratio=0.15,
        test_ratio=0.15,
        dataset_names=args.dataset
    )
    train_set, val_set, test_set = next(dm_data.split())

    dm = SCLCDataModule(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,  # optional, used for later evaluation
        **vars(args),
    )
    dm.setup()
    print(vars(args))
    model = SCModel(**vars(args))

    checkpoint_cb = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-model"
    )
    # early_stop_cb = EarlyStopping(
    #     monitor="val/loss",
    #     patience=10,
    #     mode="min"
    # )

    logger = TensorBoardLogger(
        save_dir="/data/kaplinsp/tensorboard/lightning-test-2/", # TODO move to config file
        name=args.tag, 
        version=version_string
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_cb],
        log_every_n_steps=1,
        logger=logger,
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
    })

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()