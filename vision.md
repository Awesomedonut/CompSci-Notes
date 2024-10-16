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
     - **Erosion**: Shrinks the white regions in an image. It removes pixels on the boundaries of white regions, making objects in the image smaller. (edge pixels)
     - **Dilation**: Expands the white regions in an image, adding pixels to the boundaries, making objects larger.
     - **Opening (imopen)**: Erosion followed by dilation. Used to remove small objects from the image while preserving the shape and size of larger objects.
     - **Closing (imclose)**: Dilation followed by erosion. Used to fill small holes in the image while keeping the overall shape of objects intact.

#### 8. **Bilateral Filtering (Further Reading)**
   - **Edge-Aware Filters**: Filters like the bilateral filter preserve edges while reducing noise. Unlike linear filters, these filters do not treat all regions of an image equally, helping to preserve important details like edges.
   - **Application in Image Processing**: Bilateral filters are crucial in tasks like **image denoising** where preserving edges is essential for retaining important image details.

   ** convolution

---

These notes provide a comprehensive theoretical overview of the key concepts of image filtering without MATLAB-specific content. Key topics include different types of filters (box, Gaussian, median), the importance of kernel normalization, separable filters, and image denoising and sharpening. Advanced concepts like bilateral filtering and image morphology (erosion, dilation) are also included.



# Edge Detection

### Lecture Notes: Edge Detection

1. **Overview of Topics:**
   - Introduction to computing image gradients (derivatives) using convolution.
   - Discussion of the canny edge detection method.
   - Introduction to the Laplacian of Gaussian filter.

2. **Edges in Images:**
   - Edges occur due to depth differences, surface color changes, shadows (illumination), and surface discontinuities.
   - Detecting edges provides significant information about image content.

3. **Image Gradient Calculation:**
   - Use of grayscale images for gradient and edge detection.
   - Sobel filter is introduced for calculating the gradient in the x and y directions.
     - X-gradient emphasizes horizontal edges.
     - Y-gradient emphasizes vertical edges.
   - To compute edge magnitude, treat gradients as a vector and use the square root of the sum of squares of X and Y gradients.
   - The gradient magnitude image can be enhanced by scaling (e.g., multiplying by 2).

4. **Noise in Gradients:**
   - Noise can introduce unwanted gradients. 
   - Applying a Gaussian filter smooths the image before calculating gradients, reducing noise effects.

5. **Derivative of Gaussian:**
   - Instead of applying Gaussian and Sobel filters separately, convolve the Gaussian kernel with the derivative kernel to create the "Derivative of Gaussian" kernel.
   - This method simplifies the process and reduces noise while detecting edges.

6. **canny Edge Detection:**
   - Starts with Gaussian filtering and gradient computation.
   - **Non-Maximum Suppression**: 
     - Ensures only local maxima (strongest edges) are kept, removing weaker neighbors.
   - **Double Thresholding**: 
     - Two thresholds are applied: high threshold for strong edges and low threshold for probable edges.
   - **Edge Linking**: 
     - Probable edges are confirmed if connected to strong edges; otherwise, they are discarded.

7. **MATLAB and canny Edge Detection:**
   - MATLAB’s `edge` function is used to apply canny edge detection with automatic or custom threshold values.
   - Adjustable parameters include thresholds and the amount of smoothing.

8. **Laplacian of Gaussian Filter:**
   - The Laplacian operator detects edges by finding zero crossings in the second derivative of the image.
   - Direct use of the Laplacian operator creates noisy results.
   - Solution: Combine Gaussian filtering and Laplacian filtering into the "Laplacian of Gaussian" (LoG) to reduce noise and produce cleaner edge detection.

9. **Visualizing Filters and Edges:**
   - Visualization of Gaussian and derivative filters using large kernels helps to understand their shapes and effects.
   - Laplacian of Gaussian produces cleaner edge maps by detecting zero crossings more effectively.

10. **Required Reading:**
    - **Section 7.2** for detailed understanding of edge detection.
    - **Further Reading**: "Crisp Boundary Detection Using Point-Wise Mutual Information" by Filip Izola et al. (2014). The paper discusses a statistical approach to selecting meaningful edges.

11. **Conclusion:**
    - Edge detection methods are crucial for image analysis.
    - Next lecture: Introduction to convolutional neural networks.



# Deep Learning

### Notes on Deep Learning and Convolutional Neural Networks

#### Introduction to Deep Learning
- **Deep Learning**: A subset of machine learning focused on using neural networks to solve complex problems.
- **Neural Networks**: Architectures designed to learn and represent complex functions by combining many simpler functions.
- **Supervised Learning**: A learning method where the model is trained on labeled data (input-output pairs).

#### Neural Network Basics
- **Function Representation**: In computer vision, we aim to define a function that takes an image as input and produces a desired output (e.g., classification, depth estimation).
- **Layered Structure**: Instead of a single complex function, networks use multiple layers of simpler functions (e.g., convolutional, linear, non-linear).
- **Learning Process**: Each function processes input from the previous layer, and through many layers, the network approximates the target complex function.

#### Convolutional Neural Networks (CNNs)
- **Convolutions**: A core operation in CNNs where small filters (kernels) are applied to an image to detect patterns like edges or textures.
- **Activation Maps**: The result of convolution, where patterns that match the filter trigger a high response.
- **ReLU (Rectified Linear Unit)**: A common activation function that outputs zero for negative inputs and the input itself for positive values, adding non-linearity to the network.
- **Max Pooling**: Reduces the size of the representation by selecting the maximum value in a region, effectively downsampling the image while retaining important features.
- **Encoder-Decoder Networks**: 
   - **Encoder**: Compresses the input (e.g., image) into a smaller, abstract representation.
   - **Decoder**: Expands the compressed representation back to a larger output, such as for segmentation or inpainting.

#### Network Training
- **Backpropagation**: The process of propagating the error (calculated by a loss function) back through the network to update weights and minimize the error.
- **Loss Function**: Measures the difference between the predicted output and the ground truth, guiding the network to improve. Examples include mean squared error (MSE) and domain-specific loss functions.
- **Gradient Descent**: A method to update the network’s parameters by computing the gradient of the loss function and adjusting parameters to minimize the error.

#### Case Study: Alpha Matting
- **Alpha Matting**: A technique in image processing used to separate the foreground from the background by estimating per-pixel transparency (alpha values).
- **Challenge**: Alpha matting is an under-constrained problem, meaning there are more unknowns than equations, making it difficult to find the exact solution.
- **Trimap**: An input that defines regions as definite foreground, definite background, and unknown (gray area) where the network estimates alpha values.
- **Supervised Learning in Alpha Matting**: Networks are trained with ground truth alpha mattes, though generating these ground truths is time-consuming and costly.

#### Advanced Techniques
- **Generative Adversarial Networks (GANs)**: A type of network with two components:
   - **Generator**: Creates synthetic data.
   - **Discriminator**: Tries to distinguish between real and generated data. These networks compete to improve both.
- **Image Inpainting**: A deep learning task to fill in missing regions of an image based on the surrounding context.
- **Skip Connections**: A technique where intermediate layers are connected directly to later layers, helping retain high-resolution details in the output. Often used in U-Net architectures.

#### Data Augmentation
- **Importance of Data**: Neural networks require large datasets to train effectively. Data augmentation techniques, such as mirroring, cropping, and color shifting, help expand limited datasets.
- **Alpha Matting Dataset Augmentation**: By overlaying images with different backgrounds or altering their colors, many input-output pairs can be generated from a small set of ground truths, helping the network generalize better.

#### Limitations of Deep Learning
- **Generalization Issues**: Neural networks may struggle with corner cases (e.g., novel situations) if such examples are not well represented in the training data.
- **Lack of Full Understanding**: The internal workings of a trained neural network can be difficult to interpret, making it hard to predict its behavior in all situations.

#### Conclusion
- **Deep Learning as a Tool**: While powerful, deep learning is only one tool among many in computer vision and should be combined with classical methods for best results.
- **Further Reading**: Sections 5.3 and 5.4 on deep learning and CNNs, and Stanford’s post on backpropagation.

These notes cover the key points and elaborations from the lecture on deep learning and convolutional neural networks, focusing on neural architectures, convolutional operations, network training, and the challenges of deep learning.

CNNs use **convolutions**, a mathematical operation that lets the network focus on small chunks of an image at a time, just like focusing your eyes on one detail before zooming out. These chunks are called **filters** or **kernels**. The network uses these filters to scan across the image, detecting features like edges, textures, and patterns.

Key parts of a CNN:
1. **Convolutional layers**: Where filters slide over the image and extract features.
2. **Pooling layers**: Reduce the size of the image, keeping the important information but dropping the rest (like shrinking it while keeping the essence).
3. **Fully connected layers**: After the image has been reduced to important features, these layers make the final decision—like deciding what’s in the image (cat? car? You got it).

CNNs are powerful because they learn these filters automatically during training. They don't just look at the raw pixels—they **understand** the key patterns in the data!

We are the exception, and with CNNs, your computer vision game will be, too!


neural networks: These are the backbone of modern machine learning—modeled after the brain’s neurons but operating at a speed that turns data into power.

At the core, a **neural network** is made up of **layers of neurons**, or *nodes*. Each node is like a mini-brain that processes input data and passes the result to the next layer.

Here’s the breakdown:
1. **Input layer**: Where the raw data (images, numbers, etc.) enters. Each neuron in this layer represents one feature of the input.
2. **Hidden layers**: These layers do the heavy lifting. Each neuron takes input from previous neurons, applies a *weight* to it, adds a *bias*, and then passes it through an *activation function* (like a switch that decides if the signal is strong enough to pass on).
3. **Output layer**: The final decision! Whether it’s classifying an image or predicting a number, this is where the network delivers the result.

Neural networks learn by adjusting the weights and biases through a process called **backpropagation**, powered by **gradient descent**. The goal? Minimize the error in predictions by tweaking these parameters.

Neural nets are flexible and can handle all kinds of data—images, text, you name it. They're the foundation of most AI. 


# Signals & Images

Ah, my brother, the Fourier Transform is a tool that lets you break down complex signals into simple waves, just like tearing apart a chord to hear the individual notes!

Here’s the power play: any signal, no matter how complex, can be represented as a sum of sine and cosine waves with different frequencies, amplitudes, and phases. The Fourier Transform takes a signal from the **time domain** (how the signal changes over time) and translates it into the **frequency domain** (which frequencies are present and how strong they are).

The math? It's an integral that multiplies the signal by a bunch of sine and cosine functions, then sums it all up. The result tells you how much of each frequency is present in the original signal.

It’s like taking a song and identifying every note being played. This unlocks massive potential in signal processing, data analysis, and even quantum mechanics. The frequency domain is where hidden patterns reveal themselves!

# Sampling & Aliasing

# Harris Corner Detection

# Feature Invariane, Detection, and Matching

# Transformations & Image Alignment

# RANSAC

# Image Segmentation

# Optical Flow