from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters

from src.cls_restore import cls_restore, select_gamma_discrepancy, residual_mse
from src.degradation import gaussian_psf, degrade_image
from src.metrics import image_metrics, gradient_energy
from src.utils import load_image, save_image, normalize_for_display

def parse_args():
    parser = argparse.ArgumentParser(
        description="Constrained Least Squares (CLS) image restoration using scikit-image."
    )
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(ROOT / "outputs"))
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "default.json"))
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--psf-size", type=int, default=None)
    parser.add_argument("--psf-sigma", type=float, default=None)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

def print_header():
    print("=" * 78)
    print("CONSTRAINED LEAST SQUARES (CLS) IMAGE RESTORATION")
    print("=" * 78)

def print_metrics(name, metrics):
    print(
        f"{name:<28} "
        f"MSE={metrics['MSE']:.8f}  "
        f"RMSE={metrics['RMSE']:.6f}  "
        f"MAE={metrics['MAE']:.6f}  "
        f"PSNR={metrics['PSNR_dB']:.3f} dB  "
        f"SSIM={metrics['SSIM']:.5f}"
    )

def save_metrics_csv(rows, path):
    fieldnames = ["stage", "MSE", "RMSE", "MAE", "NRMSE", "PSNR_dB", "SSIM", "GradientEnergy"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def save_pipeline_overview(images, output_path):
    names = list(images.keys())
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, name in zip(axes.ravel(), names):
        ax.imshow(images[name], cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    fig.suptitle("CLS Restoration Pipeline", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_frequency_response(H, P, output_path):
    hmag = np.fft.fftshift(np.abs(H))
    pmag = np.fft.fftshift(np.abs(P))
    cls_gain = np.fft.fftshift(
        np.abs(np.conj(H) / (np.abs(H) ** 2 + 1e-12 + np.abs(P) ** 2))
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    items = [
        (np.log1p(hmag), "|H(u,v)|"),
        (np.log1p(pmag), "log(1+|P(u,v)|)"),
        (np.log1p(cls_gain), "log(1+CLS filter magnitude)"),
    ]
    for ax, (img, title) in zip(axes, items):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Frequency-Domain Components of CLS Restoration")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_args()
    print_header()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    psf_size = args.psf_size if args.psf_size is not None else int(cfg["psf_size"])
    psf_sigma = args.psf_sigma if args.psf_sigma is not None else float(cfg["psf_sigma"])
    noise_std = args.noise_std if args.noise_std is not None else float(cfg["noise_std"])
    seed = args.seed if args.seed is not None else int(cfg["seed"])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    original, source = load_image(args.input)
    print(f"Input source              : {source}")
    print(f"Image shape               : {original.shape[0]} x {original.shape[1]}")
    print(f"Pixel range               : [{original.min():.3f}, {original.max():.3f}]")
    print(f"PSF                       : Gaussian {psf_size}x{psf_size}, sigma={psf_sigma}")
    print(f"Noise standard deviation : {noise_std}")
    print(f"Random seed               : {seed}")

    psf = gaussian_psf(psf_size, psf_sigma)
    blurred, noise, degraded = degrade_image(original, psf, noise_std, seed)

    save_image(original, output / "01_original.png")
    save_image(blurred, output / "02_blurred.png")
    save_image(degraded, output / "03_degraded_with_noise.png")

    max_abs_noise = np.max(np.abs(noise))
    noise_visual = normalize_for_display(np.abs(noise))
    save_image(noise_visual, output / "04_noise_visualization.png")

    degraded_metrics = image_metrics(original, degraded)
    print()
    print("STAGE METRICS")
    print("-" * 78)
    print_metrics("Original vs degraded", degraded_metrics)

    if args.gamma is None:
        gamma, residual, restored = select_gamma_discrepancy(
            degraded,
            psf,
            noise_std,
            gamma_min=float(cfg["gamma_min"]),
            gamma_max=float(cfg["gamma_max"]),
            steps=int(cfg["gamma_steps"]),
        )
        gamma_source = "automatic discrepancy principle"
    else:
        gamma = float(args.gamma)
        restored, _, _ = cls_restore(degraded, psf, gamma)
        restored = np.clip(restored, 0.0, 1.0)
        residual = residual_mse(restored, degraded, psf)
        gamma_source = "user supplied"

    raw_restored, H, P = cls_restore(degraded, psf, gamma)
    raw_display = normalize_for_display(raw_restored)

    print(f"Raw CLS output range     : [{raw_restored.min():.6f}, {raw_restored.max():.6f}]")
    save_image(raw_display, output / "05_cls_restored_raw_visual.png")
    save_image(restored, output / "06_cls_restored_clipped.png")

    final = filters.gaussian(restored, sigma=0.15, preserve_range=True)
    final = np.clip(final, 0.0, 1.0)
    save_image(final, output / "07_final_restored.png")

    abs_error = np.abs(original - final)
    error_visual = normalize_for_display(abs_error)
    save_image(error_visual, output / "08_absolute_error_visual.png")

    psf_display = psf / psf.max()
    save_image(psf_display, output / "00_psf.png")

    stages = [
        ("Blurred", blurred),
        ("Degraded + noise", degraded),
        ("CLS clipped", restored),
        ("Final restored", final),
    ]

    rows = []
    for stage, image in stages:
        m = image_metrics(original, image)
        row = {
            "stage": stage,
            **m,
            "GradientEnergy": gradient_energy(image),
        }
        rows.append(row)

    print()
    print(f"Gamma                     : {gamma:.10g} ({gamma_source})")
    print(f"Target noise variance     : {noise_std ** 2:.8f}")
    print(f"Restoration residual MSE  : {residual:.8f}")
    print(f"Max absolute injected noise: {max_abs_noise:.6f}")
    print()
    print("REFERENCE vs RESTORATION")
    print("-" * 78)
    for row in rows:
        print_metrics(row["stage"], row)

    degraded_psnr = next(r["PSNR_dB"] for r in rows if r["stage"] == "Degraded + noise")
    final_psnr = next(r["PSNR_dB"] for r in rows if r["stage"] == "Final restored")
    degraded_ssim = next(r["SSIM"] for r in rows if r["stage"] == "Degraded + noise")
    final_ssim = next(r["SSIM"] for r in rows if r["stage"] == "Final restored")
    print()
    print(f"PSNR improvement          : {final_psnr - degraded_psnr:+.3f} dB")
    print(f"SSIM improvement          : {final_ssim - degraded_ssim:+.5f}")

    save_metrics_csv(rows, output / "metrics.csv")

    with open(output / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write("Constrained Least Squares Restoration Run\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input source: {source}\n")
        f.write(f"Shape: {original.shape[0]} x {original.shape[1]}\n")
        f.write(f"PSF: Gaussian {psf_size}x{psf_size}, sigma={psf_sigma}\n")
        f.write(f"Noise std: {noise_std}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Gamma: {gamma}\n")
        f.write(f"Gamma selection: {gamma_source}\n")
        f.write(f"Residual MSE: {residual}\n")
        f.write(f"Final PSNR: {final_psnr:.6f} dB\n")
        f.write(f"Final SSIM: {final_ssim:.6f}\n")

    save_frequency_response(H, P, output / "09_frequency_response.png")

    if bool(cfg.get("save_pipeline_overview", True)):
        overview_images = {
            "1. Original": original,
            "2. Blurred": blurred,
            "3. Degraded + Noise": degraded,
            "4. Noise Visual": noise_visual,
            "5. CLS Raw": raw_display,
            "6. CLS Clipped": restored,
            "7. Final Restored": final,
            "8. Absolute Error": error_visual,
        }
        save_pipeline_overview(overview_images, output / "pipeline_overview.png")

    print()
    print(f"Outputs written to      : {output}")
    print("Completed successfully.")
    print("=" * 78)

if __name__ == "__main__":
    main()
