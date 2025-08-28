import argparse
from pl_datamodule import SCLCDataModule
from pl_train import SCModel

from data import DataManager

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def parse_args():
    parser = argparse.ArgumentParser(description="Train SCModel on Nodule dataset")

    # Add data module args
    parser = SCLCDataModule.add_data_specific_args(parser)
    # Add model args
    parser = SCModel.add_model_specific_args(parser)

    # Other training args
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # --- Step 1: Prepare dataset splits ---
    dm_data = DataManager(
        k_folds=1,  # single split
        val_ratio=0.15,
        test_ratio=0.15,
        dataset_names=[args.dataset_name]
    )
    train_set, val_set, test_set = next(dm_data.split())

    # --- Step 2: Initialize datamodule ---
    dm = SCLCDataModule(
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,  # optional, used for later evaluation
        cli_args=args
    )
    dm.setup()

    model = SCModel(**vars(args))

    checkpoint_cb = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-model"
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=1,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()