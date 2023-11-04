#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <png.h>
#include<time.h>
#include <hpdf.h>
#include <arv.h>
#define PADDING 2
#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1200
int wi = IMAGE_WIDTH;
int h = IMAGE_HEIGHT;
uint16_t image[IMAGE_HEIGHT * IMAGE_WIDTH];
#define WINDOW_SIZE 5
#define HALF_WINDOW (WINDOW_SIZE / 2)
typedef struct
{
  uint8_t r;
  uint8_t g;
  uint8_t b;
} rgb;
rgb *buffer_rgb;
// Example of initializing pointers
clock_t start,end;
uint8_t lookup_table[256][3];
/********************LUT Colorspce*************************/
ArvCamera *gcamera;
uint16_t* processImage(uint16_t *image, int width, int height) {
    uint16_t *processedImage = (uint16_t *)malloc(width * height * sizeof(uint16_t));
    if (processedImage == NULL) {
        printf("Memory allocation failed for the processed image buffer.\n");
        return NULL; // Handle memory allocation failure
    }   

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixelIndex = i * width + j;
            processedImage[pixelIndex] = image[pixelIndex] + 1; // Process the image data (example: increment by 1)
        }
    }
    return processedImage; // Return the processed image
}

void camerainit()
{
  uint16_t *image = (uint16_t*) malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uint16_t));
  GError *error = NULL;
  /* Connect to the first available camera */
  gcamera = arv_camera_new(NULL, &error);

  if (ARV_IS_CAMERA(gcamera))
  {
    printf("Found camera '%s'\n", arv_camera_get_model_name(gcamera, NULL));
    // printf ("hearbeat = %d\n", arv_camera_get_integer (gcamera, "GevHeartbeatTimeout", NULL));
    // digitalWrite(laser,HIGH);
    // arv_camera_set_integer (gcamera, "GevHeartbeatTimeout", 10000, NULL);
    arv_camera_set_frame_rate(gcamera, 50.0, NULL);
    arv_camera_set_gain(gcamera, 1.0, &error); // gain of the camera
    arv_camera_set_exposure_time(gcamera, 5000.0, &error);
  }
  
}void camera() {
    ArvStream *stream;
    GError *error = NULL;
    uint16_t *image = (uint16_t*) malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uint16_t));
    // Request a stream object from the camera
    stream = arv_camera_create_stream(gcamera, NULL, NULL, &error);

    if (ARV_IS_STREAM(stream)) {
        printf("Got a stream object from the camera\n");

        // Push at least one buffer into the stream input buffer queue
        arv_stream_push_buffer(stream, arv_buffer_new(1920 * 1200, NULL));

        // Start the acquisition on the camera
        arv_camera_start_acquisition(gcamera, NULL);

        // Get a buffer object from the output buffer queue
        ArvBuffer *arv_buffer = arv_stream_timeout_pop_buffer(stream, 2000000);

        if (arv_buffer != NULL && arv_buffer_get_status(arv_buffer) == ARV_BUFFER_STATUS_SUCCESS) {
            size_t size;
            // Retrieve the raw data from the buffer object
          const int* data = arv_buffer_get_data(arv_buffer, &size);

            // Assuming that the input data is 8-bit grayscale, copy it into the image buffer
            memcpy(image, data, size);

            // Release the buffer object
            g_object_unref(arv_buffer);
        } else {
            printf("Can't retrieve the image from the camera\n");
        }

        // Stop the acquisition on the camera
        arv_camera_stop_acquisition(gcamera, NULL);

        // Release the stream object
        g_object_unref(stream);
    }
}

const cl_image_format img_format = {CL_RGBA, CL_UNORM_INT8};
void save_png(const char *filename, uint16_t *buffer, int width, int height) {
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
    png_set_IHDR(png, info, width, height, 16, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png, info);

    // Write image data
    for (int y = 0; y < height; y++) {
        png_write_row(png, (png_bytep)&buffer[y * width]);
    }

    // End write
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}


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
// void readImageBinary(const char *filename, int width, int height) {
//     FILE *file = fopen(filename, "rb");
//     if (file == NULL) {
//         perror("Error opening image file");
//         exit(1);
//     }
//     fread(image, sizeof(uint16_t), width * height, file);
//     fclose(file);
// }

int main() {
    uint16_t *image = (uint16_t*) malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uint16_t));
        uint16_t* processedImage = processImage(image, IMAGE_WIDTH, IMAGE_HEIGHT);
    buffer_rgb = (rgb *)malloc(1920 * 1200 * sizeof(rgb));
    if (buffer_rgb == NULL) {
        fprintf(stderr, "Failed to allocate memory for buffer_rgb.\n");
        return 1;
    }
int *sum = (int *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(int));
float *mean = (float *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));
float *perfusion = (float *)malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));    
    start = clock();
    // Initialize the zero-padded image with zeros
size_t padded_size = (IMAGE_HEIGHT + 2 * PADDING) * (IMAGE_WIDTH + 2 * PADDING);
uint16_t *padded_image = (uint16_t*) malloc(padded_size * sizeof(uint16_t));

// Initialize the zero-padded image with zeros
memset(padded_image, 0, padded_size * sizeof(uint16_t));

// Copy the actual image to the center of the zero-padded image
for (int i = 0; i < IMAGE_HEIGHT; i++) {
    uint16_t* image_row = image + i * IMAGE_WIDTH;
    uint16_t* padded_row = padded_image + (i + PADDING) * (IMAGE_WIDTH + 2 * PADDING) + PADDING;
    memcpy(padded_row, image_row, IMAGE_WIDTH * sizeof(uint16_t));
}

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
    // Read the input image
    // readImageBinary("input_image.bin", IMAGE_WIDTH, IMAGE_HEIGHT);


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

    
    // Load OpenCL code
    size_t programSize = strlen(programSource);
    cl_program program = clCreateProgramWithSource(context, 1, &programSource, &programSize, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create kernels
    cl_kernel convertKernel = clCreateKernel(program, "convertToUInt16", NULL);
    cl_kernel statsKernel = clCreateKernel(program, "calculate_stats", NULL);
  
    // Save the converted image
    
    // Define input and output buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);
    cl_mem sumBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);
    cl_mem meanBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);
    cl_mem perfusion_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);
    cl_mem outputData = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_HEIGHT, NULL, NULL);

    // Write data to the input buffer
    clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_HEIGHT, image, 0, NULL, NULL);

      
    
    // Define global and local work sizes
  size_t global_work_size[2] = {h , wi };
  size_t local_work_size[2] = {32, 32};
    
     // Enqueue the kernel for execution
    clEnqueueNDRangeKernel(queue, convertKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    // Define global and local work sizes
  
    // Set kernel arguments for 'calculate_stats' kernel
    clSetKernelArg(statsKernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(statsKernel, 1, sizeof(cl_mem), &sumBuffer);
    clSetKernelArg(statsKernel, 2, sizeof(cl_mem), &meanBuffer);
    clSetKernelArg(statsKernel, 3, sizeof(cl_mem), &perfusion_buffer);


    // Set kernel arguments for 'convertToUInt16' kernel
    clSetKernelArg(convertKernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(convertKernel, 1, sizeof(cl_mem), &outputData);

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
 int lut_size = 256; // The size of the LUT
    int lookup_table[lut_size][3];

    // Generate the LUT
    for (int i = 0; i < lut_size; i++) {
        lookup_table[i][0] = 1; // Red component
        lookup_table[i][1] = 1; // Green component
        lookup_table[i][2] = i; // Blue component (based on grayscale value)
    }

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
    camerainit();
    processImage(image,1920,1200);
    // uint16_t *image = (uint16_t*) malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uint16_t));
// Ensure the allocation was successful
if (image == NULL) {
    fprintf(stderr, "Failed to allocate memory for the image.\n");
    return 1;
}
    camera();
    clEnqueueNDRangeKernel(queue, statsKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, sumBuffer, CL_TRUE, 0, sizeof(int) * IMAGE_WIDTH * IMAGE_HEIGHT, sum, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, meanBuffer, CL_TRUE, 0, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT, mean, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, perfusion_buffer, CL_TRUE, 0, sizeof(float) * IMAGE_WIDTH * IMAGE_HEIGHT, perfusion, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, outputData, CL_TRUE, 0,sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_HEIGHT,image, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, padded_size * sizeof(uint16_t), padded_image, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_HEIGHT, image, 0, NULL, NULL);
    // Check if processing was successful (not NULL)
    if (processedImage != NULL) {
        // Save the processed image using save_png or save_rgb_png
        // save_png("processed_image.png", processedImage, IMAGE_WIDTH, IMAGE_HEIGHT);
        
        // Free the processed image buffer
        free(processedImage);
    }
          save_rgb_png("output_image.png",(uint8_t *)image, IMAGE_WIDTH, IMAGE_HEIGHT);
    end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    // printf("CPU time used: %f seconds\n", cpu_time_used);
    //   for (int i = 0; i < IMAGE_HEIGHT; i++) {
    //     for (int j = 0; j < IMAGE_WIDTH; j++) {
    //         int pixelIndex = i * IMAGE_WIDTH + j;
    //         printf("Pixel (%d, %d): Sum=%d, Mean=%f, perfusion=%lf\n", i, j, sum[pixelIndex], mean[pixelIndex], perfusion[pixelIndex]);
            
    //     }
    // }
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
    free(buffer_rgb);
    free(sum);
    free(mean);
    free(perfusion);
    free(padded_image); 
    free(image);
    return 0;
}
