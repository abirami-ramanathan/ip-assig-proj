import numpy as np
from .degradation import psf_to_otf

def laplacian_psf():
    return np.array(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=np.float64
    )

def cls_restore(degraded, psf, gamma):
    H = psf_to_otf(psf, degraded.shape)
    P = psf_to_otf(laplacian_psf(), degraded.shape)
    G = np.fft.fft2(degraded)

    denominator = (np.abs(H) ** 2) + gamma * (np.abs(P) ** 2)
    denominator = np.maximum(denominator, 1e-12)

    restored_spectrum = (np.conj(H) / denominator) * G
    restored = np.real(np.fft.ifft2(restored_spectrum))
    return restored, H, P

def residual_mse(restored, degraded, psf):
    H = psf_to_otf(psf, degraded.shape)
    predicted = np.real(np.fft.ifft2(np.fft.fft2(restored) * H))
    return float(np.mean((predicted - degraded) ** 2))

def select_gamma_discrepancy(
    degraded, psf, noise_std, gamma_min=1e-6, gamma_max=1e-2, steps=70
):
    gammas = np.logspace(
        np.log10(gamma_min), np.log10(gamma_max), int(steps)
    )
    target = noise_std ** 2

    best = None
    for gamma in gammas:
        restored, _, _ = cls_restore(degraded, psf, float(gamma))
        restored_clipped = np.clip(restored, 0.0, 1.0)
        mse = residual_mse(restored_clipped, degraded, psf)
        score = abs(mse - target)
        item = (score, float(gamma), mse, restored_clipped)
        if best is None or item[0] < best[0]:
            best = item

    return best[1], best[2], best[3]
