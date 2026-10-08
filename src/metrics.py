import numpy as np
from skimage.metrics import (
    mean_squared_error,
    normalized_root_mse,
    peak_signal_noise_ratio,
    structural_similarity,
)

def image_metrics(reference, image):
    reference = np.asarray(reference, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)

    mse = float(mean_squared_error(reference, image))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(reference - image)))
    nrmse = float(normalized_root_mse(reference, image))
    psnr = float(peak_signal_noise_ratio(reference, image, data_range=1.0))
    ssim = float(structural_similarity(reference, image, data_range=1.0))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "NRMSE": nrmse,
        "PSNR_dB": psnr,
        "SSIM": ssim,
    }

def gradient_energy(image):
    gx = np.diff(image, axis=1)
    gy = np.diff(image, axis=0)
    return float(np.mean(gx * gx) + np.mean(gy * gy))
