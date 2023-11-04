# OpenCL Image Processing 

This program is designed to perform various image processing operations using OpenCL for accelerated GPU computations and Aravis for webcam/camera interfacing.

## Contents
- Introduction
- Source Files
- Prerequisites
- How the code works
- Usage
- License

### Introduction

This project performs operations like reading from an image, converting it to grayscale, and processing it via the OpenCL kernels. It also uses a camera for capturing an image, processes it using OpenCL and the Aravis library, and serves as an ideal starting point for developing more complex vision-based applications.

### Source Files

1. `main.c` contains the main function and the OpenCL image processing operations.

### Prerequisites

To build this project, you will need to have the following libraries installed:

1. [OpenCL](https://www.khronos.org/opencl/)
2. [Aravis](https://github.com/AravisProject/aravis) (For camera interfacing)
3. [libpng](http://www.libpng.org/pub/png/libpng.html) (For saving images in PNG format)

### How the code works

This code performs a grayscale conversion on an image. The image is initially read into a buffer, then copied into a new buffer where the processing will be done.

For input, the program expects a raw 16-bit grayscale image.

The `clock_t start, end` global variables are used to measure the time taken for the image processing operations.

In the `main` function, memory is allocated for the output image, and the original image is processed by an OpenCL kernel. This kernel increments every pixel value of the image by one. The processed image is then saved in PNG format.

There are two OpenCL kernels declared in the `programSource` string:

1. `convertToUInt16`: This kernel takes an input image buffer and an output data buffer and converts each pixel value to a 16-bit unsigned integer (uint).
    
2. `calculate_stats`: This kernel calculates the sum, mean, and deviation of window around each pixel in the image. The window size is defined by a macro `WINDOW_SIZE`.

The `range_min` and `range_max` variables are used as the upper and lower bounds of the window size for calculating the mean and deviation.

Each OpenCL kernel is run on the GPU with a 2D global work size defined by the IMAGE_WIDTH and IMAGE_HEIGHT constants.

The processed 16-bit image data is then converted back into an 8-bit image, and a lookup table is applied to it to generate an RGB image.

The Grayscale image is then written to a png file.

This process is run for every frame that is captured from the camera.

### Usage

-  Set up your environment to be able to compile OpenCL code
-  Install Aravis to interface with the camera
-  Compile the code using your OpenCL compiler
-  Run the program on images or a camera
-  The processed images will be saved as PNG files

### License

This project is released under the MIT License.
