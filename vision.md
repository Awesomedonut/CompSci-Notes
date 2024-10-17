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


A **kernel** (also called a **filter**) is a small matrix used for **convolution operations** in image processing. It slides over an image (which itself is represented as a matrix of pixel values) and transforms the image in specific ways, such as blurring, sharpening, edge detection, etc.

Now, let’s talk about the kernel your professor used: `kernel = [1 1 1]`.

This is a **1D kernel**. When applied, it performs a **moving average** or **smoothing** operation. Here’s what happens:
- The kernel slides over the input matrix (or array) and multiplies the corresponding elements in the matrix with the elements of the kernel.
- For each position, it sums up the result, which gives you a new value.

For example, if you apply `[1 1 1]` to an array or row of pixel values, it’ll average the neighboring values. This smooths out the data, reducing sharp transitions (e.g., blurring an image along one dimension).

In 2D (with images), you'd have a square kernel like:

```
[1 1 1]
[1 1 1]
[1 1 1]
```

This would blur an image across both dimensions. The kernel determines how strong the transformation is based on the values inside it and how large it is.

So, your professor’s `[1 1 1]` kernel is a simple, 1D smoothing filter—a tool to tame the wild fluctuations in data!



while there's a similar **essence** to matrix multiplication and convolution, they are **not exactly the same thing**. Convolutions are a bit more specialized in how they apply kernels to data. Let me break it down:

### **Matrix Multiplication**:
- You multiply corresponding elements of two matrices and sum them up to get a final product matrix.
- Matrix dimensions must align for multiplication to happen (e.g., an `m x n` matrix times an `n x p` matrix).

### **Convolution**:
- It’s more like a **sliding window operation**. You take a small matrix (the kernel/filter) and slide it across a larger matrix (your image or data).
- At each position, you **multiply** the elements of the kernel by the corresponding elements in the image, then **sum** those products to get a single value for that position.
- This process repeats for each position, producing a new output matrix.

### Key Differences:
1. **Sliding**: Convolution involves a kernel sliding across the input, while matrix multiplication is applied to the entire matrices directly.
2. **Size**: Convolution usually involves a smaller kernel operating on a larger matrix, whereas matrix multiplication involves specific-sized matrices.
3. **Padding/Striding**: Convolution can involve techniques like padding (adding zeros around the edges) or striding (moving the kernel by more than 1 position at a time), which don't exist in regular matrix multiplication.

Think of convolution as a **localized matrix multiplication** that happens over small patches of your data, applying transformations like edge detection, blurring, or sharpening. In contrast, matrix multiplication is more about combining matrices globally in one step.

Both involve multiplying and summing, but convolution is designed for **structured spatial relationships**—especially useful in computer vision!


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

  
Both the **box kernel** and the **Gaussian kernel** are used for smoothing (blurring), but they operate in different ways. Let’s break it down:

### **Box Kernel**:
- As we discussed, the **box kernel** gives **equal weight** to all the neighboring pixels. It’s a simple average of the surrounding pixel values.
- Example (3x3 box kernel):
  ```
  [1/9 1/9 1/9]
  [1/9 1/9 1/9]
  [1/9 1/9 1/9]
  ```
  This smooths out the image, but can sometimes produce a **harsh blur**, because it doesn’t consider distance when averaging pixels.

### **Gaussian Kernel**:
- The **Gaussian kernel** is much more sophisticated. Instead of giving equal weight to all pixels in the neighborhood, it gives more weight to pixels that are **closer to the center** and less weight to pixels that are farther away. The kernel follows a **Gaussian (normal) distribution**.
  
- Example (3x3 Gaussian kernel, simplified):
  ```
  [1/16  2/16  1/16]
  [2/16  4/16  2/16]
  [1/16  2/16  1/16]
  ```

  Notice how the center value is higher (4/16), and the values decrease as you move away from the center. This leads to **smoother** and more **natural-looking** blurs because it respects the spatial relationships between pixels.

### Key Differences:
1. **Weight Distribution**:
   - **Box Kernel**: Equal weights for all pixels in the neighborhood, treating all nearby pixels the same.
   - **Gaussian Kernel**: Higher weights for pixels near the center, gradually decreasing toward the edges. This mimics the way we perceive smooth changes in light and shadow.

2. **Blurring Effect**:
   - **Box Kernel**: Can lead to a "blocky" or "harsh" blur because it treats all pixels equally. It doesn't account for the distance of each pixel from the center.
   - **Gaussian Kernel**: Produces a more **natural**, softer blur. It reduces the influence of faraway pixels, making the blur appear more **gradual** and less blocky.

3. **Efficiency**:
   - **Box Kernel**: Simpler and faster, but can produce poor results if you're aiming for a visually pleasing blur.
   - **Gaussian Kernel**: Slightly more computationally expensive (especially with larger kernels), but the result is smoother and better for many applications like anti-aliasing, denoising, and more realistic image transformations.

### When to Use Each:
- **Box Kernel**: When you need quick and dirty blurring, like reducing noise in very simple cases.
- **Gaussian Kernel**: When you want a smooth, visually pleasing blur with more natural transitions—perfect for most image processing tasks in computer vision, especially when dealing with **image smoothing** or **preprocessing** before edge detection.

Victory lies in understanding the subtleties! The **box kernel** is simple and blunt, but the **Gaussian kernel** is your precise weapon for refined blurring.
   
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

The Laplacian of a Gaussian filter is the result of applying the Laplacian operator (which calculates the second derivative of an image) to a Gaussian filter (which smooths or blurs an image). It's commonly used in edge detection.

This combination, called **Laplacian of Gaussian (LoG)**, sharpens transitions and highlights edges by detecting regions of rapid intensity change. The Gaussian smoothing reduces noise, and the Laplacian highlights areas with high curvature or intensity change.

In image processing, this approach finds edges by locating zero crossings in the second derivative, which is key for detecting boundaries in images with precision.

Victory lies in the details, and LoG refines those edges!


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


image is a signal





the Fourier Transform is a tool that lets you break down complex signals into simple waves, just like tearing apart a chord to hear the individual notes!

Here’s the power play: any signal, no matter how complex, can be represented as a sum of sine and cosine waves with different frequencies, amplitudes, and phases. The Fourier Transform takes a signal from the **time domain** (how the signal changes over time) and translates it into the **frequency domain** (which frequencies are present and how strong they are).

The math? It's an integral that multiplies the signal by a bunch of sine and cosine functions, then sums it all up. The result tells you how much of each frequency is present in the original signal.

It’s like taking a song and identifying every note being played. This unlocks massive potential in signal processing, data analysis, and even quantum mechanics. The frequency domain is where hidden patterns reveal themselves!

### Notes: Signals and Images Lecture

1. **Introduction to Frequency Domain**
   - Discussed Fourier series to represent periodic signals.
   - Sine waves with different frequencies are added to approximate complex signals.
   - Harmonics of a signal are integer multiples of the original frequency.
   - Adding more sine waves improves the approximation of the signal.
   - A signal can be represented as the sum of sine wave components.

2. **Frequency Domain Representation**
   - The frequency domain represents the strength of each sine wave contributing to a signal.
   - Sine waves are visualized as contributing to sharp changes or smooth areas in the signal.
   - Frequency domain is an alternative to the time or spatial domain.

3. **1D Signals vs. 2D Images**
   - 1D signals: Example of sawtooth waveform, which can be generated by adding sine waves.
   - 2D signals (images): Fourier series can be extended to two dimensions, with frequency components in both X and Y directions.
   - Example: Frequency domain representation of images shows different frequency components.

4. **Fourier Transform in MATLAB**
   - Fourier Transform converts images into the frequency domain.
   - **High-frequency image**: Many edges, containing more high-frequency components.
   - **Low-frequency image**: Fewer edges, mostly smooth content.
   - Used FFT (Fast Fourier Transform) to convert discrete images into the frequency domain.
   - **FFT shift**: Visualizes repeating frequencies in MATLAB by shifting the frequency window.

5. **Magnitude and Phase of Frequency Components**
   - Frequency domain consists of complex numbers with magnitude (amplitude) and phase.
   - Magnitude indicates how much of a particular frequency is present.
   - Humans find amplitude more intuitive than phase; edges correspond to high frequencies, smooth areas to low frequencies.

6. **Mathematics Behind Fourier Transform**
   - Fourier Transform pairs, such as:
     - Box function transforms into a sinc function.
     - Gaussian function transforms into another Gaussian.
   - The **Convolution Theorem**: Convolution in spatial domain equals multiplication in the frequency domain, and vice versa.
   - This is why convolution is preferred over cross-correlation in image filtering.

7. **Filters and Their Frequency Responses**
   - **Gaussian filter**: Removes high frequencies, keeping low-frequency content.
   - **Box filter**: Allows some high-frequency details, which can cause artifacts in the image.
   - Visualized filtering results using FFT and inverse FFT.

8. **Low-Pass and High-Pass Filters**
   - Low-pass filter (Gaussian): Removes high-frequency details, resulting in a blurred image.
   - High-pass filter: Created by subtracting a Gaussian filter from a filter that does not change the image, leaving only high-frequency components (edges).
   - Demonstrated filtering using per-pixel multiplication in the frequency domain.

9. **Reading Assignment**
   - Section 3.4 on Fourier Transform.
   - Suggested videos on Fourier Transform for further understanding.

10. **Next Lecture**
   - The next topic will cover aliasing and image resizing.

These notes cover the key points, elaborations, and explanations from the lecture, omitting unrelated announcements or logistical information.


# Sampling & Aliasing

### Notes on Sampling and Aliasing

1. **Introduction to Sampling and Aliasing:**
   - Sampling involves converting a continuous signal into discrete samples.
   - Aliasing occurs when high-frequency components are undersampled, causing overlapping and distorted representations.

2. **Image Downsampling:**
   - When reducing an image's size by taking fewer pixels, low-frequency images appear normal, but high-frequency images exhibit patterns (e.g., moiré patterns).
   - These patterns are a result of aliasing due to insufficient pixels to represent high-frequency details.

3. **Frequency Domain and Sampling:**
   - In digital signals (including images), samples are discrete, unlike the continuous analog domain.
   - In the frequency domain, negative and positive frequencies appear symmetrically.
   - When sampling a signal, an impulse train is used, which is replicated in the frequency domain, repeating the original signal at intervals.

4. **Impact of Sampling on High Frequencies:**
   - Low-frequency signals can be reconstructed from samples more easily than high-frequency signals, which lose details.
   - High-frequency components "collide" when undersampled, causing corruption and aliasing in the signal.

5. **Addressing Aliasing:**
   - Increasing the number of samples reduces aliasing, but this isn't always feasible due to data and storage constraints.
   - The Nyquist Sampling Theorem states that the minimum sampling rate must be twice the highest frequency to avoid aliasing.

6. **Anti-Aliasing:**
   - Anti-aliasing involves applying a low-pass filter before sampling, which removes high frequencies that cannot be properly represented.
   - This prevents the frequency components from colliding and causing aliasing artifacts.

7. **MATLAB Example of Anti-Aliasing:**
   - A Gaussian filter was used to anti-alias a high-frequency image before downsampling, reducing aliasing artifacts in the resulting image.

8. **Anti-Aliasing in Neural Networks (Richard Zhang’s Paper):**
   - Neural networks, especially in max-pooling layers, often downsample without anti-aliasing, introducing aliasing and instability.
   - Shift invariance is lost in networks when downsampling, causing inconsistent classifications when an image is slightly shifted.
   - Adding anti-aliasing before downsampling (e.g., blurring) stabilizes the network's outputs, making it more shift-invariant and improving accuracy.

9. **Shift Invariance and Max-Pooling:**
   - Max-pooling breaks shift invariance by subsampling without anti-aliasing, leading to different results when shifting inputs by just one pixel.
   - The solution is to apply a blur before subsampling to preserve shift invariance and improve the stability of deep learning models.

10. **Performance Improvements:**
    - Anti-aliasing improves both consistency (shift-invariance) and accuracy in neural networks, as demonstrated in ImageNet classification experiments.

11. **Practical Application in Networks:**
    - Anti-aliasing can be applied to existing networks by adding a simple blur operation before subsampling, improving performance across different layers of the network.

12. **Richard Zhang’s Additional Work:**
    - Zhang works on deep learning for image synthesis, including image colorization, where black and white images are recolored based on predicted color components.

# Harris Corner Detection

**Harris Corner Detection Lecture Notes:**

- **Introduction to Feature Detection:**
  - The goal is to match features between two images.
  - Images could be consecutive in a video (tracking corners) or two different images with similar content but different angles.
  - Features should be robust against changes in illumination and structure.

- **Pipeline for Feature Detection:**
  1. **Detecting Features:** Finding corners or points of interest in the image.
  2. **Describing Features:** Describing the detected features for later matching.
  3. **Matching Features:** Finding correspondences between images and calculating transformations, like creating panoramas.

- **Harris Corner Detection Key Points:**
  - Features need to be localized accurately for tracking, 3D reconstruction, or motion estimation.
  - **Corners vs. Edges:**
    - A point (corner) can be shifted in any direction, creating noticeable differences in the image.
    - For edges, shifting in certain directions won't change the image much, so they are harder to pinpoint precisely.

- **Mathematical Approach:**
  - Energy \(E(u, v)\) measures the change in intensity when a window is shifted by \(u\) in the x-direction and \(v\) in the y-direction.
  - Calculating the sum of squared differences between the original and shifted windows helps identify strong features.
  - Performing this calculation for every pixel is computationally expensive, so mathematical approximations are used.

- **Taylor Series Expansion:**
  - Approximates the energy function using first-order derivatives (gradients in x and y directions).
  - Results in a simplified equation involving the image gradients, which are used to detect corners efficiently.

- **Second Moment Matrix:**
  - A matrix representing the intensity changes in both x and y directions.
  - The eigenvalues of this matrix determine the "cornerness" of a point.
  - If both eigenvalues are high, the point is likely a corner. If one is high and the other is low, it's an edge. If both are low, it’s a flat region.

- **Harris Corner Score:**
  - Calculated using:
    \[
    \text{Cornerness score} = \text{det}(M) - k \times (\text{trace}(M))^2
    \]
  - \(k\) is a small constant (typically between 0.04 and 0.06).
  - This score helps to identify corners in the image.

- **Practical Considerations:**
  - Thresholding is applied to remove weak corners.
  - Non-maxima suppression is used to retain only the strongest local corners.

- **Sobel and Gaussian Filters:**
  - Used to compute image gradients and smooth the image.
  - Gaussian filtering ensures a more robust detection of features.

- **Non-Maxima Suppression and Dilation:**
  - Non-maxima suppression identifies local maxima by comparing each pixel to its neighbors.
  - Image dilation is used to find the maximum values in a local neighborhood, helping to detect corner points.

- **Final Steps in MATLAB:**
  - Read the image, compute gradients using Sobel filters, and apply the second moment matrix.
  - Compute the cornerness function, apply non-maxima suppression, and visualize the detected corners.

- **Conclusion and Further Reading:**
  - Harris Corner Detection helps find stable and robust features, even under varying illumination.
  - For further exploration, review section 7.1.1 of the reading material and a specific paper on feature robustness under extreme conditions (e.g., matching day and night images).

- **Next Topic:**
  - The next lecture will cover SIFT (Scale-Invariant Feature Transform), which builds on the concepts of feature detection and matching.

# Feature Invariane, Detection, and Matching

# Transformations & Image Alignment

# RANSAC

# Image Segmentation

# Optical Flow

**Optical Flow Lecture Notes:**

- **Introduction to Optical Flow:**
  - Optical flow measures the motion between two or more video frames by analyzing pixel movement.
  - Motion is a powerful perceptual cue that helps identify shapes and objects, especially in noisy or texture-less environments.

- **Applications of Motion Estimation:**
  - 3D reconstruction, object segmentation, tracking dynamic elements, event/activity recognition, video editing.
  - Motion field: 3D scene motion projection onto the image.
  - Optical flow captures apparent motion, but may differ from the actual motion (motion field).

- **Key Assumptions for Optical Flow Estimation:**
  1. **Brightness Constancy:** Pixel brightness remains consistent across frames (can fail in cases like shadows).
  2. **Small Motion Assumption:** Points don't move too far between frames.
  3. **Spatial Coherence:** Close-by pixels move similarly, analogous to image segmentation.

- **Brightness Constancy Equation:**
  - Expanded using Taylor series to simplify into a linear equation.
  - Problem: One equation with two unknowns (u, v), leading to an under-constrained problem known as the **aperture problem**.

- **Aperture Problem:**
  - Perceived motion may differ from actual motion when viewed through a small "aperture" or window (e.g., barber pole illusion).
  - Solution involves adding more equations using spatial coherence (e.g., considering a 5x5 window of pixels), forming a **linear least squares problem**.

- **Corner Detection and the Second Moment Matrix:**
  - Corners provide a reliable motion estimation, while lines suffer from the aperture problem.
  - Similar to Harris corner detection, corners help resolve the under-constrained motion problem.

- **Lucas-Kanade Optical Flow:**
  - Popular approach for optical flow estimation.
  - Works well for small motions but struggles with large motions or non-uniform pixel motion.
  - Challenges arise when brightness constancy fails, or when objects move differently from their surroundings.

- **Depth and Motion:**
  - Objects closer to the camera move more, while distant objects exhibit smaller motions.
  - Lucas-Kanade struggles with large motions from close objects (e.g., a moving tree), leading to noisy results.

- **Multi-Resolution Estimation:**
  - Solves the problem of large motions by using images at multiple resolutions.
  - Apply Lucas-Kanade to smaller, blurred versions of the image to capture large motions, then refine at higher resolutions iteratively.

- **Feature Matching for Optical Flow:**
  - Instead of relying on brightness constancy, match distinctive features between frames.
  - Challenges: Finding good features, handling occlusions, and updating feature tracks over time.
  - Shi-Tomasi Feature Tracker: Identifies good features to track using eigenvalues of the second moment matrix.

- **Bidirectional Optical Flow:**
  - Estimate flow from frame t-1 to t and t to t-1, then check for agreement to detect possible errors.

- **Practical Example:**
  - Tracking objects (e.g., a traffic sign) across frames should yield consistent and accurate results when optical flow is applied correctly.

- **Further Reading:**
  - Section 7.1.5 on feature tracking.
  - OpenCV tutorial on optical flow for practical implementation across different languages.

These notes summarize the core concepts of optical flow, key assumptions, methods, and challenges in motion estimation.