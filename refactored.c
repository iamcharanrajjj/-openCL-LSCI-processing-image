#include <CL/cl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"

extern int preview_flag;
extern int reanalyse_start;
extern int window_size;
extern int std_max;
extern int std_min;
extern float *avg_Z;

int colormap_calib = 0;
float cmap_min_val = 30;
float cmap_max_val = 120;

/******************LUT Colorspace*************************/
uint8_t lookup_table[256][3] = {
    {0, 1, 43},
    {0, 2, 47},
    {0, 3, 50},
    // ...
};

cl_context context;
cl_command_queue command_queue;

void checkError(cl_int error, const char *message) {
    if (error != CL_SUCCESS) {
        fprintf(stderr, "%s (Error: %d)\n", message, error);
        exit(1);
    }
}

cl_program createProgram(const char *kernelSource, cl_context context, cl_device_id device) {
    cl_int error;
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &error);
    checkError(error, "clCreateProgramWithSource");

    error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (error != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build log: %s\n", buffer);
        exit(1);
    }

    return program;
}

int gpu_init() {
    cl_int error;

    // Get the platform
    cl_platform_id platform;
    error = clGetPlatformIDs(1, &platform, NULL);
    checkError(error, "clGetPlatformIDs");

    // Get the device (you may need to choose a specific device if you have multiple devices)
    cl_device_id device;
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    checkError(error, "clGetDeviceIDs");

    // Create an OpenCL context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    checkError(error, "clCreateContext");

    // Create a command queue
    command_queue = clCreateCommandQueue(context, device, 0, &error);
    checkError(error, "clCreateCommandQueue");
const char *kernelFile = "kernel.cl"; // Replace with the correct file path
FILE *file = fopen(kernelFile, "r");
if (!file) {
    perror("Failed to open kernel file");
    exit(1);
}

// Get the size of the file
fseek(file, 0, SEEK_END);
size_t fileSize = ftell(file);
rewind(file);

// Allocate memory to store the kernel source
char *source = (char *)malloc(fileSize + 1);
if (!source) {
    perror("Memory allocation error");
    fclose(file);
    exit(1);
}

// Read the source code from the file
size_t bytesRead = fread(source, 1, fileSize, file);
if (bytesRead != fileSize) {
    perror("Failed to read kernel source");
    fclose(file);
    free(source);
    exit(1);
}

source[fileSize] = '\0'; // Null-terminate the source code

// Close the file
fclose(file);
cl_program program = createProgram(source, context, device);
clBuildProgram(program, 1, &device, NULL, NULL, NULL);
cl_kernel kernel = clCreateKernel(program, "your_kernel_function_name", &error);
cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * data_size, NULL, &error);
clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
clSetKernelArg(kernel, 1, sizeof(int), &data_size);
size_t global_work_size[1] = {data_size};
error = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);
    return 0;
}


void saveImageArrayAsBinary(const char *filename, unsigned char *imageArray, size_t arraySize) {
    FILE *file = fopen(filename, "wb");

    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    size_t elements_written = fwrite(imageArray, sizeof(unsigned char), arraySize, file);

    if (elements_written != arraySize) {
        perror("Error writing image array");
    }

    fclose(file);
}

void gpu_lsci(uint16_t buffer[BUFFER_SIZE]) {
    cl_int error;
    size_t global_size[2] = {wi, h}; // Set your global work size here

    // Create OpenCL buffers for input and output data
    cl_mem dev_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint16_t) * wi * h, buffer, &error);
    cl_mem dev_zimg = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * (wi + window_size - 1) * (h + window_size - 1), NULL, &error);

    // Copy input data from the host to the device
    error = clEnqueueWriteBuffer(command_queue, dev_buffer, CL_TRUE, 0, sizeof(uint16_t) * wi * h, buffer, 0, NULL, NULL);
    checkError(error, "clEnqueueWriteBuffer");

    // Compile the OpenCL kernel (replace "kernel_code" with your actual kernel code)
    cl_program program = createProgram(kernel_code, context, device);

    // Create a kernel from the compiled program
    cl_kernel fillZeroPaddedArrayKernel = clCreateKernel(program, "fillZeroPaddedArray", &error);
    checkError(error, "clCreateKernel");

    // Set kernel arguments
    clSetKernelArg(fillZeroPaddedArrayKernel, 0, sizeof(cl_mem), &dev_buffer);
    clSetKernelArg(fillZeroPaddedArrayKernel, 1, sizeof(cl_mem), &dev_zimg);
    clSetKernelArg(fillZeroPaddedArrayKernel, 2, sizeof(cl_int), &window_size);

    // Enqueue the kernel for execution
    error = clEnqueueNDRangeKernel(command_queue, fillZeroPaddedArrayKernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
    checkError(error, "clEnqueueNDRangeKernel");

    // Wait for the kernel to finish
    clFinish(command_queue);

    // Copy the result back from the device to the host
    error = clEnqueueReadBuffer(command_queue, dev_zimg, CL_TRUE, 0, sizeof(float) * (wi + window_size - 1) * (h + window_size - 1), zimg, 0, NULL, NULL);
    checkError(error, "clEnqueueReadBuffer");

    // Release OpenCL resources (buffers, kernel, program, etc.)
    clReleaseMemObject(dev_buffer);
    clReleaseMemObject(dev_zimg);
    clReleaseKernel(fillZeroPaddedArrayKernel);
    clReleaseProgram(program);
}


void colormap_avg_img(int num_imgs) {
    cl_int error;

    // Create OpenCL buffers for input and output data
    cl_mem d_avg_Z = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * wi * h, NULL, &error);
    cl_mem d_buffer_rgb = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(rgb) * wi * h, NULL, &error);

    // Copy input data (avg_Z) from the host to the device
    error = clEnqueueWriteBuffer(command_queue, d_avg_Z, CL_TRUE, 0, sizeof(float) * wi * h, avg_Z, 0, NULL, NULL);
    checkError(error, "clEnqueueWriteBuffer");

    // Compile the OpenCL kernel (replace "kernel_code" with your actual kernel code)
    cl_program program = createProgram(kernel_code, context, device);

    // Create a kernel from the compiled program
    cl_kernel computeAvgZKernel = clCreateKernel(program, "compute_avg_Z", &error);
    checkError(error, "clCreateKernel");

    // Set kernel arguments
    clSetKernelArg(computeAvgZKernel, 0, sizeof(cl_mem), &d_avg_Z);
    clSetKernelArg(computeAvgZKernel, 1, sizeof(cl_mem), &d_buffer_rgb);
    clSetKernelArg(computeAvgZKernel, 2, sizeof(cl_int), &num_imgs);
    clSetKernelArg(computeAvgZKernel, 3, sizeof(cl_int), &std_max);
    clSetKernelArg(computeAvgZKernel, 4, sizeof(cl_float), &cmap_min_val);
    clSetKernelArg(computeAvgZKernel, 5, sizeof(cl_float), &cmap_max_val);

    // Set global and local work sizes (adjust these based on your requirements)
    size_t global_size[2] = {wi, h}; // Set your global work size here
    size_t local_size[2] = {32, 32}; // Set your local work size here

    // Enqueue the kernel for execution
    error = clEnqueueNDRangeKernel(command_queue, computeAvgZKernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    checkError(error, "clEnqueueNDRangeKernel");

    // Wait for the kernel to finish
    clFinish(command_queue);

    // Copy the result back from the device to the host
    error = clEnqueueReadBuffer(command_queue, d_avg_Z, CL_TRUE, 0, sizeof(float) * wi * h, avg_Z, 0, NULL, NULL);
    checkError(error, "clEnqueueReadBuffer");

    error = clEnqueueReadBuffer(command_queue, d_buffer_rgb, CL_TRUE, 0, sizeof(rgb) * wi * h, buffer_rgb, 0, NULL, NULL);
    checkError(error, "clEnqueueReadBuffer");

    // Release OpenCL resources (buffers, kernel, program, etc.)
    clReleaseMemObject(d_avg_Z);
    clReleaseMemObject(d_buffer_rgb);
    clReleaseKernel(computeAvgZKernel);
    clReleaseProgram(program);
}

