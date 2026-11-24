import argparse
from pathlib import Path

import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from .datamodule import PhongSceneDataModule
from .module import PhongGANModule


def apply_overrides(cfg: dict, overrides: list) -> dict:
    for item in overrides:
        key, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"Błędne nadpisanie: {item!r} (oczekiwano klucz=wartość)")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
        print(f"nadpisano: {key} = {node[parts[-1]]!r}")
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="../configs/phong_gan.yaml")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="SEKCJA.KLUCZ=WARTOŚĆ",
        help="nadpisanie pojedynczych wartości z YAML-a",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.set)

    seed_everything(cfg["seed"], workers=True)
    t = cfg["train"]
    out_dir = Path(t["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    datamodule = PhongSceneDataModule(
        data_dir=cfg["data"]["dir"],
        batch_size=t["batch_size"],
        val_batch_size=t["val_batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    model = PhongGANModule(cfg)

    loggers = [
        CSVLogger(str(out_dir), name="logs"),
        TensorBoardLogger(str(out_dir), name="tb"),
    ]
    callbacks = [
        ModelCheckpoint(
            dirpath=str(out_dir / "checkpoints"),
            filename="epoch_{epoch:04d}",
            every_n_epochs=t["save_every_n_epochs"],
            save_top_k=-1,
            save_last=True,
        ),
    ]

    trainer = Trainer(
        default_root_dir=str(out_dir),
        accelerator=t["accelerator"],
        devices=t["devices"],
        precision=t["precision"],
        max_epochs=t["epochs"],
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=t["log_every_n_steps"],
        check_val_every_n_epoch=t["check_val_every_n_epoch"],
        deterministic="warn",
        gradient_clip_val=t.get("gradient_clip_val"),
    )
    trainer.fit(model, datamodule=datamodule)
    print(
        f"Zakończono. Logi: {out_dir / 'logs'} | Checkpointy: {out_dir / 'checkpoints'}"
    )


if __name__ == "__main__":
    main()
