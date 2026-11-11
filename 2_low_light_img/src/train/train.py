from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from src.train.dataset import LowLightDataModule
from src.train.lit_module import HVICIDNetLitModule


def main() -> None:
    config_path = Path("src/train/config.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    L.seed_everything(cfg["seed"], workers=True)
    torch.set_float32_matmul_precision("high")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    datamodule = LowLightDataModule(
        train_dir=Path(data_cfg["train_dir"]),
        batch_size=data_cfg["batch_size"],
        patch_size=data_cfg["patch_size"],
        repeat_factor=data_cfg["repeat_factor"],
        val_split=data_cfg["val_split"],
        num_workers=data_cfg["num_workers"],
        seed=cfg["seed"],
    )

    lit_model = HVICIDNetLitModule(
        channels=model_cfg["channels"],
        heads=model_cfg["heads"],
        lr=train_cfg["lr"],
        max_epochs=train_cfg["epochs"],
    )

    ckpt_dir = Path(model_cfg["output_dir"]) / model_cfg["name"]
    loggers = [
        CSVLogger(save_dir=ckpt_dir, name="logs"),
        TensorBoardLogger(save_dir=ckpt_dir, name="tb_logs"),
    ]

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{val_psnr:.2f}",
            monitor="val_psnr",
            mode="max",
            save_top_k=1,
            save_last=True
        ),
        EarlyStopping(
            monitor="val_psnr",
            mode="max",
            patience=train_cfg["early_stopping_patience"],
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        max_epochs=train_cfg["epochs"],
        accelerator="auto",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    trainer.fit(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
