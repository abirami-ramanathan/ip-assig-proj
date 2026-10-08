import numpy as np

def gaussian_psf(size=9, sigma=0.8):
    if size < 3 or size % 2 == 0:
        raise ValueError("PSF size must be an odd integer >= 3.")
    if sigma <= 0:
        raise ValueError("PSF sigma must be > 0.")

    radius = size // 2
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    psf = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    psf /= psf.sum()
    return psf

def psf_to_otf(psf, shape):
    psf = np.asarray(psf, dtype=np.float64)
    otf = np.zeros(shape, dtype=np.float64)
    kh, kw = psf.shape
    if kh > shape[0] or kw > shape[1]:
        raise ValueError("PSF must be smaller than the image.")
    otf[:kh, :kw] = psf
    otf = np.roll(otf, -(kh // 2), axis=0)
    otf = np.roll(otf, -(kw // 2), axis=1)
    return np.fft.fft2(otf)

def degrade_image(image, psf, noise_std, seed):
    rng = np.random.default_rng(seed)
    H = psf_to_otf(psf, image.shape)
    blurred = np.real(np.fft.ifft2(np.fft.fft2(image) * H))
    noise = rng.normal(0.0, noise_std, image.shape)
    noisy = np.clip(blurred + noise, 0.0, 1.0)
    return blurred, noise, noisy
