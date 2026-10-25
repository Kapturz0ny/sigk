import argparse
import logging
from pathlib import Path
from typing import Any

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.super_resolution.train.dataset import SRDataModule
from src.super_resolution.train.lit_module import NAFNetSRLitModule

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Loads training configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    """Loads a YAML configuration, applies overrides, and runs NAFNet SR training."""
    parser = argparse.ArgumentParser(
        description="Train a NAFNet super-resolution model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/super_resolution/train/config_scale4.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--train_dir", type=str, default=None, help="Override data.train_dir."
    )
    parser.add_argument(
        "--test_dir", type=str, default=None, help="Override data.test_dir."
    )
    parser.add_argument(
        "--model_name", type=str, default=None, help="Override model.name."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Override model.output_dir."
    )
    parser.add_argument("--scale", type=int, default=None, help="Override model.scale.")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override training.epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override data.batch_size."
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Override data.num_workers."
    )
    parser.add_argument(
        "--precision", type=str, default=None, help="Override training.precision."
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=None,
        help="Optional fraction/absolute number of training batches per epoch.",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=None,
        help="Optional fraction/absolute number of validation batches per epoch.",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run a single train/validation batch to verify the pipeline.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the test split with the best checkpoint after training.",
    )

    args = parser.parse_args()
    cfg = load_config(Path(args.config))

    seed = cfg["seed"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    if args.train_dir:
        data_cfg["train_dir"] = args.train_dir
    if args.test_dir:
        data_cfg["test_dir"] = args.test_dir
    if args.model_name:
        model_cfg["name"] = args.model_name
    if args.output_dir:
        model_cfg["output_dir"] = args.output_dir
    if args.scale is not None:
        model_cfg["scale"] = args.scale
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.precision:
        training_cfg["precision"] = args.precision

    scale = int(model_cfg["scale"])
    model_name = model_cfg["name"]
    output_dir = Path(model_cfg["output_dir"])
    ckpt_dir = output_dir / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = ckpt_dir / "config.yaml"
    with open(resolved_config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(cfg, file, sort_keys=False)

    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    datamodule = SRDataModule(
        train_dir=Path(data_cfg["train_dir"]),
        test_dir=Path(data_cfg["test_dir"]),
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        hr_size=data_cfg["hr_size"],
        scale=scale,
        val_split=data_cfg["val_split"],
        seed=seed,
    )

    model_kwargs = {
        "in_channels": model_cfg["in_channels"],
        "out_channels": model_cfg["out_channels"],
        "embed_dim": model_cfg["embed_dim"],
        "num_blocks": model_cfg["num_blocks"],
        "scale": scale,
    }
    model = NAFNetSRLitModule(
        model_kwargs=model_kwargs,
        learning_rate=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
        max_epochs=training_cfg["epochs"],
        lpips_net_type=model_cfg.get("lpips_net_type", "alex"),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}-{val_psnr:.2f}",
        monitor="val_psnr",
        mode="max",
        save_top_k=1,
        save_weights_only=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_psnr",
        patience=training_cfg["early_stopping_patience"],
        mode="max",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer_kwargs: dict[str, Any] = {
        "max_epochs": training_cfg["epochs"],
        "accelerator": "auto",
        "devices": 1,
        "precision": training_cfg["precision"],
        "gradient_clip_val": training_cfg["gradient_clip_val"],
        "log_every_n_steps": 10,
        "callbacks": [checkpoint_callback, early_stop_callback, lr_monitor],
        "default_root_dir": ckpt_dir,
    }
    if args.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = True
    else:
        if args.limit_train_batches is not None:
            trainer_kwargs["limit_train_batches"] = args.limit_train_batches
        if args.limit_val_batches is not None:
            trainer_kwargs["limit_val_batches"] = args.limit_val_batches

    trainer = L.Trainer(**trainer_kwargs)

    logger.info(
        f"Starting NAFNet SR training (scale=x{scale}, seed={seed}) for '{model_name}'..."
    )
    trainer.fit(model, datamodule=datamodule)
    logger.info(
        f"Training completed. Best checkpoint: {checkpoint_callback.best_model_path}"
    )

    if args.test and checkpoint_callback.best_model_path:
        trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=checkpoint_callback.best_model_path,
        )


if __name__ == "__main__":
    main()
