# CLS implementation notes

## 1. Objective

Restore a blurred and noisy image using the constrained least-squares approach from the textbook.

## 2. Degradation model

The project uses

`g = h*f + n`

where `f` is the clean image, `h` is a Gaussian point-spread function, and `n` is additive Gaussian noise.

The default experiment uses a 9x9 Gaussian PSF with sigma 0.8 and noise standard deviation 0.001. The degradation is intentionally mild so that the restoration can be visually close to the clean reference.

## 3. CLS constraint

The constraint is based on the 2-D Laplacian:

```text
 0  1  0
 1 -4  1
 0  1  0
```

The Laplacian penalizes large second-order variations and helps suppress unstable high-frequency noise while retaining important structures.

## 4. Frequency-domain solution

The code implements

`F_hat = H*G / (|H|^2 + gamma|P|^2)`

where `H` is the Fourier transform of the PSF and `P` is the Fourier transform of the Laplacian operator.

## 5. Gamma selection

When the user does not supply `--gamma`, the program searches a logarithmic range of gamma values and chooses the value whose forward-restoration residual is closest to the known synthetic noise variance. This follows a discrepancy-principle idea and does not use the clean original image to choose gamma.

## 6. Evaluation metrics

The terminal and `metrics.csv` report:

- MSE
- RMSE
- MAE
- NRMSE
- PSNR
- SSIM
- Gradient energy

PSNR and SSIM are computed against the clean reference in the synthetic experiment. For a real unknown degraded image, those reference-based metrics require a corresponding clean image.
