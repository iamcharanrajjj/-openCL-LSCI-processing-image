#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1200
#define WINDOW_SIZE 5
#define HALF_WINDOW floor(WINDOW_SIZE / 2)

float image[IMAGE_HEIGHT * IMAGE_WIDTH];
const char *programSource =
    "#define IMAGE_WIDTH 1920;"
    "#define IMAGE_HEIGHT 1200;"
    "__kernel void calculate_stats(__global uint16 *inputImage,__global int *sum,__global float *mean, __global float *stdDev) {"
    ""
    "    int gid_x = get_global_id(0);"
    "    int gid_y = get_global_id(1);"
    "    int windowSum = 0;"
    "    float windowMean = 0.0;"
    "    float windowStdDev = 0.0;"
    "for (int i = -2; i <= 2; i++) {"
"    for (int j = -2; j <= 2; j++) {"
"        int x = gid_x + i;"
"        int y = gid_y + j;"
"        if (x >= 0 && x < IMAGE_WIDTH && y >= 0 && y < IMAGE_HEIGHT) {"
"            uint16 value = inputImage[y * IMAGE_WIDTH + x];"
"            windowSum += value;"
"        }"
"    }"
"}"

"windowMean = (float)windowSum / 25.0f;"

"for (int i = -2; i <= 2; i++) {"
"    for (int j = -2; j <= 2; j++) {"
"        int x = gid_x + i;"
"        int y = gid_y + j;"
"        if (x >= 0 && x < IMAGE_WIDTH && y >= 0 && y < IMAGE_HEIGHT) {"
"            uint16 value = inputImage[y * IMAGE_WIDTH + x];"
"            float diff = (float)value - windowMean;"
"            windowStdDev += diff * diff; // Accumulate squared differences"
"        }"
"    }"
"}"

"windowStdDev = sqrt(windowStdDev / 25.0f);"

"int outputIndex = gid_y * IMAGE_WIDTH + gid_x;"
"sum[outputIndex] = windowSum;"
"mean[outputIndex] = windowMean;"
"stdDev[outputIndex] = windowStdDev;"
    "}";

void readImageBinary(const char *filename, int width, int height) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening image file");
        exit(1);
    }
    fread(image, sizeof(uint16_t), width * height, file);
    fclose(file);
}

int main() {
    const int width = IMAGE_WIDTH;
    const int height = IMAGE_HEIGHT;
    const int numpixels = width * height;

    // Initialize OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Read the input image
    readImageBinary("input_image.bin", width, height);

    int *sum = (int *)malloc(sizeof(int) * numpixels);
    float *mean = (float *)malloc(sizeof(float) * numpixels);
    float *stdDev = (float *)malloc(sizeof(float) * numpixels);

    // Define input and output buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint16_t) * numpixels, image, NULL);
    cl_mem sumBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * numpixels, sum, NULL);
    cl_mem meanBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * numpixels, mean, NULL);
    cl_mem stdDevBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * numpixels, stdDev, NULL);
    cl_mem widthBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &width, NULL);
    cl_mem heightBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &height, NULL);

    // Load OpenCL code
    size_t programSize = strlen(programSource);
    cl_program program = clCreateProgramWithSource(context, 1, &programSource, &programSize, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_build_status buildStatus;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(buildStatus), &buildStatus, NULL);
    if (buildStatus != CL_SUCCESS) {
        char *buildLog;
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        buildLog = (char *)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog, NULL);
        printf("OpenCL program build error:\n%s\n", buildLog);
        free(buildLog);
        exit(1);
    }

    
    // Create a kernel from the program
    cl_kernel kernel = clCreateKernel(program, "calculate_stats", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &sumBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &meanBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &stdDevBuffer);
    clSetKernelArg(kernel, 4, sizeof(float), &width);
    clSetKernelArg(kernel, 5, sizeof(float), &height);
  
    // Define global and local work sizes
    size_t globalWorkSize[2] = {width, height};

    // Enqueue the kernel for execution
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    clFinish(queue);

    // Read the results back to the host
    clEnqueueReadBuffer(queue, sumBuffer, CL_TRUE, 0, sizeof(int) * numpixels, sum, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, meanBuffer, CL_TRUE, 0, sizeof(float) * numpixels, mean, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, stdDevBuffer, CL_TRUE, 0, sizeof(float) * numpixels, stdDev, 0, NULL, NULL);

    // Print the results
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixelIndex = i * width + j;
            printf("%.1f ", mean[pixelIndex]);
        }
        printf("\n");
    }

    // Cleanup and release OpenCL resources
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(sumBuffer);
    clReleaseMemObject(meanBuffer);
    clReleaseMemObject(stdDevBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(sum);
    free(mean);
    free(stdDev);

    return 0;
}
