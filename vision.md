# Images and Colours

## Intro to MATLAB

many diff and useful tools for img processing and comp vision, used for diff purposes
ex: fast, real time app -> C++ solution
deep learning, img processing -> python (many DL frameworks use python)

why MATLAB?
usefl for lin alg, espec for huge matrices and complex lin alg solutions
very convinient and easy for quickstarts
doesnt require other installations

(aside: openCV exists. open comp vision library, covers most stuff u can do w MATLAB)

important: select a coding environment and make it ur home
be comfy w trying out small things using the lang

MATLAB -> glorified calculator

How we look at images in MATLAB (and digitally in general)

colour images
what colour is
how we can represent colour ditigally

alpha channels and compositing
alpha channels: how we represent transparancey in images


Here’s a summary of the theoretical knowledge extracted from the lecture transcript, excluding any MATLAB coding parts:

### Introduction to Computer Vision:
- **Images and Colors**: An overview of how we perceive and represent images and colors digitally.
  
### Representation of Color:
- **Color Representation**: 
  - Color is represented digitally using red, green, and blue channels (RGB).
  - Each color channel stores information about how much red, green, or blue is present in a pixel.
  
- **Alpha Channels**: 
  - Alpha channels represent transparency in images. An alpha value of 1 means the pixel is fully opaque, and a value of 0 means it’s fully transparent.
  
### Image Encoding and Types:
- **Image Formats**:
  - **8-bit Representation**: Most images are encoded with 8-bit precision, allowing 256 levels of brightness.
  - **JPEG vs PNG**:
    - JPEG: Compressed images, resulting in some data loss.
    - PNG: Lossless compression, supports transparency via alpha channels.
  
### Image Resizing and Information Loss:
- **Resizing Images**:
  - **Downscaling**: Reducing the image size results in the loss of high-frequency details (like edges), which are lost permanently.
  - **Upscaling**: Enlarging a downscaled image results in blurriness since the original details are lost.
  - **Super Resolution**: A computer vision problem aimed at improving the quality of upscaled images by generating new details based on the low-frequency parts of the image.
  
### Quantization:
- **Quantization**:
  - The process of reducing the number of levels in an image’s color representation. For example, reducing an 8-bit image with 256 levels to fewer levels like 10 results in a stepwise or banded appearance in gradients.
  
### Human Vision and Color Perception:
- **Electromagnetic Spectrum**: Human vision only perceives a small portion of the electromagnetic spectrum.
- **Rod and Cone Cells**: 
  - **Rods**: Provide black-and-white vision and work better in low-light conditions.
  - **Cones**: Enable color vision and are concentrated in the center of the retina. There are three types of cone cells, sensitive to red, green, and blue.
  - **Color Blindness**: A result of malfunctioning cones, typically affecting the red and green ones.
  
- **Trichromatic Vision**: Human vision relies on three types of cone cells, giving a three-dimensional color experience. Some rare individuals (tetrachromats) have four types of cone cells.

### Color Spaces:
- **Color Spaces**:
  - **RGB**: Most intuitive for humans, since it reflects how we perceive color. However, it is inefficient for image encoding since the channels contain similar information.
  - **YCBCR**: Separates brightness (illumination) from color information. Used for efficient transmission (e.g., TV broadcasts).
  - **HSV (Hue, Saturation, Value)**: A color space that aligns better with human perception, often used in image editing applications.
  - **LAB Color Space**: Another space for representing colors, which can be used for color-based segmentation.

### Image Histograms:
- **Histograms**: A way to visualize the distribution of pixel intensities in an image. The x-axis represents pixel intensity, while the y-axis shows the frequency of pixels with that intensity.
  
### Alpha Compositing:
- **Alpha Compositing**: The technique of combining images with transparency. It uses the alpha channel to determine how the foreground image overlays onto the background. This technique is widely used in image editing, video production, and green screen compositing.

### Additional Reading Recommendations:
- **Color Photography**: The early techniques of capturing color images by taking multiple exposures with different color filters.
- **Alpha Compositing**: Further reading on how transparency and compositing are managed in images.
  
### Suggested Academic Reading:
- **Palette-Based Photo Recoloring**: A method for recoloring images presented at SIGGRAPH 2015.

The theoretical knowledge covered focuses on color perception, image representation, color spaces, and the fundamental principles of image processing, omitting any detailed coding or MATLAB-specific instructions.


# Image Filtering

# Edge Detection

# Deep Learning

# Signals & Images

# Sampling & Aliasing

# Harris Corner Detection

# Feature Invariane, Detection, and Matching

# Transformations & Image Alignment

# RANSAC

# Image Segmentation

# Optical Flow