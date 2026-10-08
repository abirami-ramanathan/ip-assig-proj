# Input images

The project defaults to `skimage.data.camera()`, so no external image is required.

To use an internet image, save a JPG or PNG here and run:

```powershell
python src\main.py --input "input\my_image.jpg"
```

For the textbook constrained least-squares formulation, the image is converted to grayscale before restoration.
A good external image is a sharp grayscale portrait, building, road, or textured object with clear edges.
