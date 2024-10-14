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
saturation vs value: sat is intensity, ie low sat means it looks washed out and high sat means it looks super vivid
val is brightness, ie low val is black and high val is as light as it goes (closer to white)
  hgih sat high val: bright and vivid
  high sat low val: dark colours but vivid (deep or rich colour)
  low sat high val: basically white, light but washed out aka pastel
  low sat low val: washed out grayish colour, dull, blackish


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

### Image Filtering: Theoretical Notes

#### 1. **Introduction to Image Filtering**
   - **Image Filtering**: A process applied to images to enhance them or extract useful information. This lecture covers:
     - Box filters
     - Gaussian filters
     - Median filtering (non-linear)
     - Image morphology

#### 2. **Box Filter**
   - **Box Kernel**: A simple type of filter where all elements are equal. It is called a box filter because, when plotted, it resembles a box.
   - **Normalization**: It is important to normalize kernels so that the sum of all elements equals 1, ensuring that the filtered image maintains the original image's intensity range.
   - **Cross-Correlation**: A method of filtering where a kernel is applied across the signal or image to check how well the two match.
   - **Convolution**: A more widely used operation in image processing than cross-correlation. In convolution, the kernel is flipped before applying, and it has useful mathematical properties:
     - **Commutative**: Order of operations doesn’t matter.
     - **Associative**: Allows combining filters before applying them, saving computation.
     - **Distributive over addition**: Useful for certain image processing tasks.
   - **Spatial Domain vs. Frequency Domain**: Convolution in the spatial domain is equivalent to multiplication in the frequency domain.

#### 3. **Gaussian Filter**
   - **Gaussian Kernel**: A kernel derived from the Gaussian (normal) distribution, which is a commonly seen distribution in nature (also known as the Bell curve).
   - **Characteristics of Gaussian**: 
     - Defined by the standard deviation (σ). 
     - Convolving two Gaussians results in another Gaussian.
     - Gaussian filtering is useful for **denoising** images by smoothing out high-frequency noise.
   - **Separable Filters**: The Gaussian filter can be separated into two 1D filters, reducing computational complexity:
     - Instead of using a large 2D kernel, the image is filtered along one direction (e.g., x-axis), and then along the other direction (e.g., y-axis).
     - This reduces the number of operations significantly, especially for larger kernels.
   
#### 4. **Image Denoising**
   - **Noise in Images**: Caused by various factors such as sensor limitations or lighting conditions. 
   - **Box vs. Gaussian Filters**: Both can be used to blur and reduce noise, but the Gaussian filter provides a more natural-looking blur compared to the box filter, which can produce unnatural artifacts.
   - **Denoising Application**: Helps reduce noise while preserving essential features like edges.

#### 5. **Image Sharpening**
   - **Sharpening Process**: Emphasizes edges by adding back the difference between the original image and the blurred image (the blurred image contains the smoother content, and the difference highlights the edges).
   - **Edge Detection**: The difference between the original and the blurred image can be used to extract edges, which can then be added back to strengthen the image's sharpness.

#### 6. **Median Filtering**
   - **Non-Linear Filtering**: Unlike linear filters (box, Gaussian), the median filter is a non-linear process that replaces each pixel's value with the median value of its neighboring pixels.
   - **Noise Removal**: Particularly effective for removing **salt and pepper noise** (randomly occurring black and white pixels) because the median filter chooses the central value and discards extreme pixel values (0 or 1).
   - **Application of Median Filter**: Commonly used in scenarios where pixel values are either completely black or white due to sensor defects.

#### 7. **Image Morphology**
   - **Binary Maps**: Image morphology typically deals with black-and-white (binary) images, where pixels are either 0 (black) or 1 (white).
   - **Useful Operations**:
     - **Erosion**: Shrinks the white regions in an image. It removes pixels on the boundaries of white regions, making objects in the image smaller.
     - **Dilation**: Expands the white regions in an image, adding pixels to the boundaries, making objects larger.
     - **Opening (imopen)**: Erosion followed by dilation. Used to remove small objects from the image while preserving the shape and size of larger objects.
     - **Closing (imclose)**: Dilation followed by erosion. Used to fill small holes in the image while keeping the overall shape of objects intact.

#### 8. **Bilateral Filtering (Further Reading)**
   - **Edge-Aware Filters**: Filters like the bilateral filter preserve edges while reducing noise. Unlike linear filters, these filters do not treat all regions of an image equally, helping to preserve important details like edges.
   - **Application in Image Processing**: Bilateral filters are crucial in tasks like **image denoising** where preserving edges is essential for retaining important image details.

---

These notes provide a comprehensive theoretical overview of the key concepts of image filtering without MATLAB-specific content. Key topics include different types of filters (box, Gaussian, median), the importance of kernel normalization, separable filters, and image denoising and sharpening. Advanced concepts like bilateral filtering and image morphology (erosion, dilation) are also included.



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