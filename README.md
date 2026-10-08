# Constrained Least Squares (CLS) Image Restoration

This project implements the constrained least-squares restoration method described in the supplied textbook page.

The mathematical model is

`g(x,y) = h(x,y) * f(x,y) + n(x,y)`

and the CLS objective is

`Q = min ||g - h*f||^2 + gamma ||L*f||^2`

where `L` is the Laplacian operator. In the frequency domain, the implemented solution is

`F_hat(u,v) = H*(u,v) G(u,v) / (|H(u,v)|^2 + gamma |P(u,v)|^2)`.

The code uses `scikit-image` for image loading, grayscale conversion, metrics, the built-in `camera()` sample image, and final image handling. NumPy FFTs implement the frequency-domain equations and SciPy is used only through packages already standard in a scientific Python environment.

## Project structure

```text
ip-assig-proj/
├── config/
│   └── default.json
├── input/
│   └── README.md
├── outputs/
├── reports/
├── src/
│   ├── __init__.py
│   ├── cls_restore.py
│   ├── degradation.py
│   ├── main.py
│   ├── metrics.py
│   └── utils.py
├── .gitignore
├── README.md
├── requirements.txt
├── run.bat
└── run.ps1
```

## Windows setup

Open PowerShell:

```powershell
cd "C:\Users\Abirami Ramanathan\Desktop\PROJECTS\IP-Proj\ip-assig-proj"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the complete demo:

```powershell
python src\main.py
```

## Use your own image

Place a JPG or PNG inside `input` and run:

```powershell
python src\main.py --input "input\my_image.jpg"
```

You can also override the blur/noise settings:

```powershell
python src\main.py --input "input\my_image.jpg" --psf-size 9 --psf-sigma 0.8 --noise-std 0.001
```

To force a particular CLS regularization value:

```powershell
python src\main.py --gamma 0.0003
```

Without `--gamma`, the program selects gamma using a discrepancy-principle search driven by the known synthetic noise level, not by the original image.

## What appears in `outputs`

`01_original.png` is the reference image.

`02_blurred.png` is the image after the Gaussian point-spread function.

`03_degraded_with_noise.png` is the observed degraded image.

`04_noise_visualization.png` visualizes the injected noise magnitude.

`05_cls_restored_raw_visual.png` shows the raw inverse FFT result scaled for viewing.

`06_cls_restored_clipped.png` is the CLS result constrained to a valid image range.

`07_final_restored.png` is the saved final restoration.

`08_absolute_error_visual.png` shows where the final result differs from the reference.

`00_psf.png` shows the Gaussian blur kernel.

`pipeline_overview.png` puts all major stages together.

`metrics.csv` contains MSE, RMSE, MAE, NRMSE, PSNR, SSIM, and gradient energy for every stage.

`run_summary.txt` contains the run configuration and final metrics.

## Important project note

For a restoration experiment, PSNR and SSIM need a known clean reference. The default demo therefore starts from a clean `skimage.data.camera()` image, creates a controlled blur/noise degradation, restores it with CLS, and compares the restoration against the known clean image. With a real unknown degraded photograph, the program can still restore it, but reference-based PSNR/SSIM cannot be truthfully computed unless a clean counterpart exists.

The default degradation is intentionally mild so that the restoration can visually approach the original while still demonstrating the CLS method.

## Recommended external image

Use a sharp grayscale image with strong edges and moderate texture. A portrait, building facade, road scene, or object with clear boundaries works well. Avoid already blurry, heavily compressed, or very noisy images for the first experiment.
