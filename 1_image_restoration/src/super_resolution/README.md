# Super-resolution (NAFNet, x4 / x8)

Pipeline for 64→256 (x4) and 32→256 (x8) super-resolution on DIV2K, with an
OpenCV bicubic baseline for comparison.

## 1. Data

```bash
python -m src.super_resolution.download_data --output_dir data/raw
python -m src.super_resolution.prepare_data --raw_dir data/raw --output_dir data/processed
```

`prepare_data.py` center-crops DIV2K HR images to squares and resizes them to
`256x256` (`data/processed/train_256`, `data/processed/valid_256`). The dataset
creates LR inputs on the fly (64x64 for x4, 32x32 for x8).

If DIV2K is already extracted elsewhere, pass its directory as `--raw_dir`; it
must contain `DIV2K_train_HR` and `DIV2K_valid_HR`.

## 2. Training

```bash
python -m src.super_resolution.train.train --config src/super_resolution/train/config_scale4.yaml
python -m src.super_resolution.train.train --config src/super_resolution/train/config_scale8.yaml
```

Checkpoints are saved to `models/upscale/nafnet_sr_x4` and
`models/upscale/nafnet_sr_x8` together with the resolved `config.yaml`.

Useful overrides: `--epochs`, `--batch_size`, `--num_workers`, `--precision`,
`--scale`, `--model_name`, `--output_dir`, `--limit_train_batches`,
`--fast_dev_run`.

## 3. Evaluation against OpenCV bicubic

```bash
python -m src.super_resolution.evaluate \
  --config src/super_resolution/train/config_scale4.yaml \
  --weights models/upscale/nafnet_sr_x4/best-epoch=XX-val_psnr=XX.XX.ckpt

python -m src.super_resolution.evaluate \
  --config src/super_resolution/train/config_scale8.yaml \
  --weights models/upscale/nafnet_sr_x8/best-epoch=XX-val_psnr=XX.XX.ckpt
```

Results are written to `results/upscale/x4` and `results/upscale/x8`:

- `per_image.csv` – PSNR/SSIM/SNR/LPIPS per image and method,
- `summary.csv` – mean/std per method,
- `comparison.csv` – NAFNet vs OpenCV bicubic with `improvement` and
  `better_method` (`improvement > 0` always means NAFNet is better, including
  LPIPS where lower is better),
- `metrics.json` – metadata,
- `visuals/` – four-panel LR / bicubic / NAFNet / HR figures.

## 4. Combined results for both scales

```bash
python -m src.super_resolution.collect_results --results_dir results/upscale
```

This creates `summary_all_scales.csv`, `comparison_all_scales.csv` and
`comparison_all_scales.png` in `results/upscale`.

## Notes

- The model and data pipeline mirror the original notebook, but use the
  repository's `lightning` version, `MetricsEvaluator`, and preprocessing code.
- LPIPS defaults to the `alex` backbone (as in the denoising pipeline); pass
  `--lpips_net_type vgg` to evaluate with VGG instead.
- `data/` is git-ignored; checkpoints and results are written outside `data/`.
- On this machine the virtual environment uses `torch 2.14.1+cu130`. Running
  `uv sync` may replace it with the CPU wheel; reinstall the CUDA build with:

  ```bash
  uv pip install --python .venv/Scripts/python.exe --reinstall \
    "torch==2.14.1+cu130" "torchvision==0.29.1+cu130" \
    --extra-index-url https://download.pytorch.org/whl/cu130
  ```
