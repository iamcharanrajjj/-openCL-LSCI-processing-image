#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <png.h>

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1200
#define WINDOW_SIZE 5
#define HALF_WINDOW (WINDOW_SIZE / 2)
uint16_t image[IMAGE_HEIGHT * IMAGE_WIDTH];
uint8_t rgb_image[IMAGE_HEIGHT][IMAGE_WIDTH][3];
typedef struct
{
  uint8_t r;
  uint8_t g;
  uint8_t b;
} rgb;

const cl_image_format img_format = {CL_RGBA, CL_UNORM_INT8};
// void save_png(const char *filename, uint16_t *buffer, int width, int height) {
//     FILE *fp = fopen(filename, "wb");
//     if (!fp) {
//         fprintf(stderr, "Failed to open file for writing: %s\n", filename);
//         return;
//     }

//     png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
//     if (!png) {
//         fprintf(stderr, "Failed to create PNG write structure\n");
//         fclose(fp);
//         return;
//     }

//     png_infop info = png_create_info_struct(png);
//     if (!info) {
//         fprintf(stderr, "Failed to create PNG info structure\n");
//         png_destroy_write_struct(&png, NULL);
//         fclose(fp);
//         return;
//     }

//     if (setjmp(png_jmpbuf(png))) {
//         fprintf(stderr, "Error during PNG file write\n");
//         png_destroy_write_struct(&png, &info);
//         fclose(fp);
//         return;
//     }

//     png_init_io(png, fp);
//     // Set image information
//     png_set_IHDR(png, info, width, height, 16, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
//                  PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

//     png_write_info(png, info);

//     // Write image data
//     for (int y = 0; y < height; y++) {
//         png_write_row(png, (png_bytep)&buffer[y * width]);
//     }

//     // End write
//     png_write_end(png, NULL);
//     png_destroy_write_struct(&png, &info);
//     fclose(fp);
// }


void save_rgb_png(const char *filename, uint8_t *buffer, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fprintf(stderr, "Failed to create PNG write structure\n");
        fclose(fp);
        return;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "Failed to create PNG info structure\n");
        png_destroy_write_struct(&png, NULL);
        fclose(fp);
        return;
    }

    if (setjmp(png_jmpbuf(png))) {
        fprintf(stderr, "Error during PNG file write\n");
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);
    // Set image information
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png, info);

    // Write image data
    for (int y = 0; y < height; y++) {
        png_write_row(png, (png_bytep)&buffer[y * width * 3]); // Assuming 3 channels (RGB)
    }

    // End write
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}

// Define functions to read and write binary image files
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

  float range_min =  0.88; //cmap_min_val 0.875
  float range_max =  0.97; // cmap_max_val 0.96
  float range = range_max - range_min;
  float rangeinv = 1 / range;
    // Initialize OpenCL
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    size_t imageSize = IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uint16_t);
    uint16_t *imageData = (uint16_t *)malloc(imageSize);
    // Read the input image
    readImageBinary("input_image.bin", IMAGE_WIDTH, IMAGE_HEIGHT);

    // Read the results back to the host
    int *sum = (int *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(int));
    float *mean = (float *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));
    float *perfusion = (float *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));

    const char *programSource =
        "__kernel void convertToUInt16(__global uchar *imageData, __global uint *outputData) {"
        "    int gid = get_global_id(0);"
        "    outputData[gid] = (uint)imageData[gid];"
        "}"
        "__kernel void calculate_stats(__global uint *inputImage, "
        "__global int *sum, "
        "__global float *mean, "
        "__global float *perfusion) {"
        "    int gid_x = get_global_id(0);"
        "    int gid_y = get_global_id(1);"
        "    int windowSum = 0;"
        "    float windowMean = 0.0;"
        "    float windowStdDev = 0.0;"
        "    for (int i = -2; i <= 2; i++) {"
        "        for (int j = -2; j <= 2; j++) {"
        "            int x = gid_x + i;"
        "            int y = gid_y + j;"
        "            if (x >= 0 && x < 1920 && y >= 0 && y < 1200) {"
        "                int value = inputImage[y * 1920 + x];"
        "                windowSum += value;"
        "                windowMean += (float)value;"
        "            }"
        "        }"
        "    }"
        "    windowMean /= 25.0f;"
        "    for (int i = -2; i <= 2; i++) {"
        "        for (int j = -2; j <= 2; j++) {"
        "            int x = gid_x + i;"
        "            int y = gid_y + j;"
        "            if (x >= 0 && x < 1920 && y >= 0 && y < 1200) {"
        "                int value = inputImage[y * 1920 + x];"
        "                windowStdDev += pow((float)value - windowMean, 2);"
        "            }"
        "        }"
        "    }"
        "    windowStdDev = sqrt(windowStdDev / 25.0f);"
        "    int outputIndex = gid_y * 1920 + gid_x;"
        "    sum[outputIndex] = windowSum;"
        "    mean[outputIndex] = windowMean;"
        "    perfusion[outputIndex] = windowMean / windowStdDev;"
        "}";

    // Save the converted image
    

    // Define input and output buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);
    cl_mem sumBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);
    cl_mem meanBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);
    cl_mem perfusion_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);

    cl_mem outputData = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);

    // Write data to the input buffer
    clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_HEIGHT, image, 0, NULL, NULL);

    // Load OpenCL code
    size_t programSize = strlen(programSource);
    cl_program program = clCreateProgramWithSource(context, 1, &programSource, &programSize, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create kernels
    cl_kernel convertKernel = clCreateKernel(program, "convertToUInt16", NULL);
    cl_kernel statsKernel = clCreateKernel(program, "calculate_stats", NULL);

    // Set kernel arguments for 'calculate_stats' kernel
    clSetKernelArg(statsKernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(statsKernel, 1, sizeof(cl_mem), &sumBuffer);
    clSetKernelArg(statsKernel, 2, sizeof(cl_mem), &meanBuffer);
    clSetKernelArg(statsKernel, 3, sizeof(cl_mem), &perfusion_buffer);

    // Set kernel arguments for 'convertToUInt16' kernel
    clSetKernelArg(convertKernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(convertKernel, 1, sizeof(cl_mem), &outputData);

    // Define global and local work sizes
    size_t globalWorkSize[2] = {IMAGE_WIDTH, IMAGE_HEIGHT};
    size_t localWorkSize[2] = {1, 1};
    
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
    for (int j = 0; j < IMAGE_WIDTH; j++) {
        int pixelIndex = i * IMAGE_WIDTH + j;
        image[pixelIndex] = 65535 - image[pixelIndex];
    }
}

for (int i = 0; i < IMAGE_HEIGHT; i++) {
    for (int j = 0; j < IMAGE_WIDTH; j++) {
        int pixelIndex = i * IMAGE_WIDTH + j;
        image[pixelIndex] = (uint16_t)((double)image[pixelIndex] / 65535.0 * 255.0);
    }
}

for(int i = 0; i < IMAGE_HEIGHT; i++) {
    for(int j = 0; j < IMAGE_WIDTH; j++) {
        image[i * IMAGE_WIDTH +j] = 255 - image[i * IMAGE_WIDTH +j];
    }
}

rgb *buffer_rgb = (rgb *)malloc(1920 * 1200 * sizeof(rgb));

uint8_t min_val = 255, max_val = 0;
for (int i = 0; i < IMAGE_HEIGHT; i++) {
    for (int j = 0; j < IMAGE_WIDTH; j++) {
     if(image[i * IMAGE_WIDTH +j] < min_val)min_val = image[i * IMAGE_WIDTH +j];
     if(image[i * IMAGE_WIDTH + j] > max_val) max_val = image[i * IMAGE_WIDTH + j];
    }
} 


for (int i = 0; i < IMAGE_HEIGHT; i++) {
    for (int j = 0; j < IMAGE_WIDTH; j++) {
                image[i * IMAGE_WIDTH + j] = (image[i * IMAGE_WIDTH + j] - min_val) * 255 / (max_val - min_val);
    }
}

/********************LUT Colorspce*************************/
uint8_t lookup_table[256][3] =
    {
        {0, 0, 131},
        {0, 0, 135},
        {0, 0, 139},
        {0, 0, 143},
        {0, 0, 147},
        {0, 0, 151},
        {0, 0, 155},
        {0, 0, 159},
        {0, 0, 163},
        {0, 0, 167},
        {0, 0, 171},
        {0, 0, 175},
        {0, 0, 179},
        {0, 0, 183},
        {0, 0, 187},
        {0, 0, 191},
        {0, 0, 195},
        {0, 0, 199},
        {0, 0, 203},
        {0, 0, 207},
        {0, 0, 211},
        {0, 0, 215},
        {0, 0, 219},
        {0, 0, 223},
        {0, 0, 227},
        {0, 0, 231},
        {0, 0, 235},
        {0, 0, 239},
        {0, 0, 243},
        {0, 0, 247},
        {0, 0, 251},
        {0, 0, 255},
        {0, 4, 255},
        {0, 8, 255},
        {0, 12, 255},
        {0, 16, 255},
        {0, 20, 255},
        {0, 24, 255},
        {0, 28, 255},
        {0, 32, 255},
        {0, 36, 255},
        {0, 40, 255},
        {0, 44, 255},
        {0, 48, 255},
        {0, 52, 255},
        {0, 56, 255},
        {0, 60, 255},
        {0, 64, 255},
        {0, 68, 255},
        {0, 72, 255},
        {0, 76, 255},
        {0, 80, 255},
        {0, 84, 255},
        {0, 88, 255},
        {0, 92, 255},
        {0, 96, 255},
        {0, 100, 255},
        {0, 104, 255},
        {0, 108, 255},
        {0, 112, 255},
        {0, 116, 255},
        {0, 120, 255},
        {0, 124, 255},
        {0, 128, 255},
        {0, 131, 255},
        {0, 135, 255},
        {0, 139, 255},
        {0, 143, 255},
        {0, 147, 255},
        {0, 151, 255},
        {0, 155, 255},
        {0, 159, 255},
        {0, 163, 255},
        {0, 167, 255},
        {0, 171, 255},
        {0, 175, 255},
        {0, 179, 255},
        {0, 183, 255},
        {0, 187, 255},
        {0, 191, 255},
        {0, 195, 255},
        {0, 199, 255},
        {0, 203, 255},
        {0, 207, 255},
        {0, 211, 255},
        {0, 215, 255},
        {0, 219, 255},
        {0, 223, 255},
        {0, 227, 255},
        {0, 231, 255},
        {0, 235, 255},
        {0, 239, 255},
        {0, 243, 255},
        {0, 247, 255},
        {0, 251, 255},
        {0, 255, 255},
        {4, 255, 251},
        {8, 255, 247},
        {12, 255, 243},
        {16, 255, 239},
        {20, 255, 235},
        {24, 255, 231},
        {28, 255, 227},
        {32, 255, 223},
        {36, 255, 219},
        {40, 255, 215},
        {44, 255, 211},
        {48, 255, 207},
        {52, 255, 203},
        {56, 255, 199},
        {60, 255, 195},
        {64, 255, 191},
        {68, 255, 187},
        {72, 255, 183},
        {76, 255, 179},
        {80, 255, 175},
        {84, 255, 171},
        {88, 255, 167},
        {92, 255, 163},
        {96, 255, 159},
        {100, 255, 155},
        {104, 255, 151},
        {108, 255, 147},
        {112, 255, 143},
        {116, 255, 139},
        {120, 255, 135},
        {124, 255, 131},
        {128, 255, 128},
        {131, 255, 124},
        {135, 255, 120},
        {139, 255, 116},
        {143, 255, 112},
        {147, 255, 108},
        {151, 255, 104},
        {155, 255, 100},
        {159, 255, 96},
        {163, 255, 92},
        {167, 255, 88},
        {171, 255, 84},
        {175, 255, 80},
        {179, 255, 76},
        {183, 255, 72},
        {187, 255, 68},
        {191, 255, 64},
        {195, 255, 60},
        {199, 255, 56},
        {203, 255, 52},
        {207, 255, 48},
        {211, 255, 44},
        {215, 255, 40},
        {219, 255, 36},
        {223, 255, 32},
        {227, 255, 28},
        {231, 255, 24},
        {235, 255, 20},
        {239, 255, 16},
        {243, 255, 12},
        {247, 255, 8},
        {251, 255, 4},
        {255, 255, 0},
        {255, 251, 0},
        {255, 247, 0},
        {255, 243, 0},
        {255, 239, 0},
        {255, 235, 0},
        {255, 231, 0},
        {255, 227, 0},
        {255, 223, 0},
        {255, 219, 0},
        {255, 215, 0},
        {255, 211, 0},
        {255, 207, 0},
        {255, 203, 0},
        {255, 199, 0},
        {255, 195, 0},
        {255, 191, 0},
        {255, 187, 0},
        {255, 183, 0},
        {255, 179, 0},
        {255, 175, 0},
        {255, 171, 0},
        {255, 167, 0},
        {255, 163, 0},
        {255, 159, 0},
        {255, 155, 0},
        {255, 151, 0},
        {255, 147, 0},
        {255, 143, 0},
        {255, 139, 0},
        {255, 135, 0},
        {255, 131, 0},
        {255, 128, 0},
        {255, 124, 0},
        {255, 120, 0},
        {255, 116, 0},
        {255, 112, 0},
        {255, 108, 0},
        {255, 104, 0},
        {255, 100, 0},
        {255, 96, 0},
        {255, 92, 0},
        {255, 88, 0},
        {255, 84, 0},
        {255, 80, 0},
        {255, 76, 0},
        {255, 72, 0},
        {255, 68, 0},
        {255, 64, 0},
        {255, 60, 0},
        {255, 56, 0},
        {255, 52, 0},
        {255, 48, 0},
        {255, 44, 0},
        {255, 40, 0},
        {255, 36, 0},
        {255, 32, 0},
        {255, 28, 0},
        {255, 24, 0},
        {255, 20, 0},
        {255, 16, 0},
        {255, 12, 0},
        {255, 8, 0},
        {255, 4, 0},
        {255, 0, 0},
        {251, 0, 0},
        {247, 0, 0},
        {243, 0, 0},
        {239, 0, 0},
        {235, 0, 0},
        {231, 0, 0},
        {227, 0, 0},
        {223, 0, 0},
        {219, 0, 0},
        {215, 0, 0},
        {211, 0, 0},
        {207, 0, 0},
        {203, 0, 0},
        {199, 0, 0},
        {195, 0, 0},
        {191, 0, 0},
        {187, 0, 0},
        {183, 0, 0},
        {179, 0, 0},
        {175, 0, 0},
        {171, 0, 0},
        {167, 0, 0},
        {163, 0, 0},
        {159, 0, 0},
        {155, 0, 0},
        {151, 0, 0},
        {147, 0, 0},
        {143, 0, 0},
        {139, 0, 0},
        {135, 0, 0},
        {131, 0, 0},
        {128, 0, 0}};
for(int i = 0; i < IMAGE_HEIGHT; i++) {
    for(int j = 0; j < IMAGE_WIDTH; j++) {
        uint8_t r  = lookup_table[image[i * IMAGE_WIDTH + j]][0];
        uint8_t g  = lookup_table[image[i * IMAGE_WIDTH + j]][1];
        uint8_t b  = lookup_table[image[i * IMAGE_WIDTH + j]][2];

         buffer_rgb[i * IMAGE_WIDTH + j].r = r; // Red
         buffer_rgb[i * IMAGE_WIDTH + j].g = g; // Green
         buffer_rgb [i * IMAGE_WIDTH + j].b = b; // Blue
    }
}
    // Enqueue the kernel for execution
    clEnqueueNDRangeKernel(queue, convertKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, statsKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

    // Read the result buffers
    clEnqueueReadBuffer(queue, sumBuffer, CL_TRUE, 0, sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT, sum, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, meanBuffer, CL_TRUE, 0, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT, mean, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, perfusion_buffer, CL_TRUE, 0, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT, perfusion, 0, NULL, NULL);

    // save_png("normal_image.png", image, IMAGE_WIDTH, IMAGE_HEIGHT);
    // save_png("converted_image.png", perfusion, IMAGE_WIDTH, IMAGE_HEIGHT);

// Call the function to save the RGB image to a PNG file
save_rgb_png("output_image.png", (uint8_t *)buffer_rgb, IMAGE_WIDTH, IMAGE_HEIGHT);
    
    // Print the results
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            int pixelIndex = i * IMAGE_WIDTH + j;
            //printf("Pixel (%d, %d): Sum=%d, Mean=%f, StdDev=%f\n", i, j, sum[pixelIndex], mean[pixelIndex], stdDev[pixelIndex]);
            
        }
    }
    // Cleanup and release OpenCL resources
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(sumBuffer);
    clReleaseMemObject(meanBuffer);
    clReleaseMemObject(perfusion_buffer);
    clReleaseMemObject(outputData);
    clReleaseKernel(convertKernel);
    clReleaseKernel(statsKernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(sum);
    free(mean);
    free(perfusion);
    free(imageData);

    return 0;
}
