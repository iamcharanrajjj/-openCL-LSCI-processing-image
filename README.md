# Laser Speckle Contrast Imaging (LSCI) Program

This repository contains a program that leverages OpenCL to perform Laser Speckle Contrast Imaging (LSCI). LSCI is a powerful, non-invasive bio-imaging technique for assessing blood flow dynamics in tissue.

The program initialises a Aravis-supported camera, captures an image, and then performs the LSCI on the GPU using OpenCL. The processed images are then written to disk.

## Source Files

The source code is mainly composed of a single file `lsci.c`.

The file contains several functions to perform various operations on captured images such as normalizing the image, computing the average variance, converting grayscale to RGB, and more. The core logic resides in the OpenCL kernels filled in the constant string `source_str`.

## Dependencies
* OpenCL
* [Aravis](https://github.com/AravisProject/aravis)
* C standard libraries

## How To Compile The Code

To compile this program, you will need to have OpenCL and the Aravis library installed in your system. Once you have these installed, you can use a gcc compiler to compile the `lsci.c` file:

```
gcc -o lsci lsci.c `pkg-config --cflags --libs aravis-* opencl`
```

## How To Run
After successful compilation, you can run the program using:

```
./lsci
```

## How The Program Works

1. The program first initializes the camera and captures images from it.
2. The captured images are then sent to the GPU for LSCI processing. The following steps are performed on each image:
    - The image is padded with zeros to match the window size.
    - Computation and normalisation of speckle contrast image.
    - Application of colormap to the grayscale image.
    - Computation of average speckle contrast image for ensemble averaging.

The program also has the functionality to preview the processed images.

## API Reference

- `int main(void)`: The main function from which the program is started.
- `camerainit()`: Initiates the camera and sets the proper settings.
- `camera()`: Captures images from the camera.
- `gpu_init()`: Establishes a connection with the OpenCL device.
- `gpu_lsci(uint16_t *buffer)`: Sends the data to the GPU, executes the kernel, and retrieves the result.

## Caution

This code is not ready for use in a production environment. It is a prototype created for demonstration and illustrative purposes.

The error handling in the code is not complete and in certain failure scenarios, the program may not clean up properly or crash.

## License

The code in this repository is available under the [MIT License](https://opensource.org/licenses/MIT). Please see the [LICENSE](./LICENSE) file for details.
