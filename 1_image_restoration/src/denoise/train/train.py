import argparse
import logging
from pathlib import Path
from typing import Any
import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from src.denoise.train.dataset import DenoiseDataModule
from src.denoise.train.lit_module import NAFNetLitModule

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_config(config_path: Path) -> dict[str, Any]:
    """Loads training configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    """Loads YAML configuration, sets random seeds, and runs the Lightning Trainer."""
    parser = argparse.ArgumentParser(description="Train NAFNet using a YAML configuration.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/denoise/train/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--input_dir", type=str, default=None, help="Optional override for input_dir.")
    parser.add_argument("--model_name", type=str, default=None, help="Optional override for model name.")

    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    seed = cfg["seed"]
    input_dir = Path(args.input_dir or cfg["data"]["input_dir"])
    gt_dir = Path(cfg["data"]["gt_dir"])
    model_name = args.model_name or cfg["model"]["name"]
    output_dir = Path(cfg["model"]["output_dir"])

    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    datamodule = DenoiseDataModule(
        input_dir=input_dir,
        gt_dir=gt_dir,
        batch_size=cfg["data"]["batch_size"],
        val_split=cfg["data"]["val_split"],
        num_workers=cfg["data"]["num_workers"],
        seed=seed,
    )

    model = NAFNetLitModule(
        width=cfg["model"]["width"],
        lr=cfg["training"]["lr"],
        max_epochs=cfg["training"]["epochs"],
    )

    ckpt_dir = output_dir / model_name

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}-{val_psnr:.2f}",
        monitor="val_psnr",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_psnr",
        patience=cfg["training"]["early_stopping_patience"],
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        default_root_dir=ckpt_dir,
    )

    logging.info(f"Starting NAFNet training (seed={seed}) for model '{model_name}'...")
    trainer.fit(model, datamodule=datamodule)
    logging.info(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
