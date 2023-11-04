// Another variant
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "lsci.h"
// Define constants
#define BUFFER_SIZE 1920 * 1200

// Define global variables
int window_size = 5;
int std_max = 255;
int std_min = 0;
float beta = 1.0;
float SG = 2.0;
int calib_flag = 1;
int preview_flag = 0;
int reanalyse_start = 0;
int wi = 1920;
int h = 1200;
int w, half_w;

// Define data structures
typedef struct
{
    uint8_t r;
    uint8_t g;
    uint8_t b;
} rgb;

ArvCamera *gcamera;
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
  }
  
  void camera() {
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

// Define kernel source code
const char *source_str = "#include \"lsci.h\"\n"
                         "__kernel void fillZeroPaddedArray(__global uint16_t *buffer, __global float *zimg, int w, int h, int wi)\n"
                         "{\n"
                         "    int half_w = w / 2;\n"
                         "    int i = get_global_id(0);\n"
                         "    int j = get_global_id(1);\n"
                         "\n"
                         "    if (i < h + w - 1 && j < wi + w - 1)\n"
                         "    {\n"
                         "        if ((i >= half_w && j >= half_w) && (i < h + half_w && j < wi + half_w))\n"
                         "        {\n"
                         "                        zimg[i * (wi + w - 1) + j] = (float)buffer[(i - half_w) * wi + (j - half_w)] != 0 ? (float)buffer[(i - half_w) * wi + (j - half_w)] : 1;"
                         "        }\n"
                         "        else\n"
                         "        {\n"
                         "                        zimg[i * (wi + w - 1) + j] = 1.0;\n"
                         "        }\n"
                         "    }\n"
                         "}\n"
                         "\n"
                         "__kernel void lsci_kernel(__global float *zimg, __global float *Z, int w, int h)\n"
                         "{\n"
                         "    int half_w = w / 2;\n"
                         "    int i = get_global_id(0);\n"
                         "    int j = get_global_id(1);\n"
                         "\n"
                         "    if ((i >= half_w && i < h + half_w) && (j >= half_w && j < wi + half_w))\n"
                         "    {\n"
                         "        float s = 0.0;\n"
                         "        float m = 0.0;\n"
                         "        float sd = 0.0;\n"
                         "        float SD = 0.0;\n"
                         "\n"
                         "        for (int iter_i = -half_w; iter_i <= half_w; iter_i++)\n"
                         "        {\n"
                         "            for (int iter_j = -half_w; iter_j <= half_w; iter_j++)\n"
                         "            {\n"
                         "                s += zimg[(i + iter_i) * (wi + w - 1) + (j + iter_j)];\n"
                         "            }\n"
                         "        }\n"
                         "\n"
                         "        m = s / (w * w);\n"
                         "\n"
                         "        for (int iter_i = -half_w; iter_i <= half_w; iter_i++)\n"
                         "        {\n"
                         "            for (int iter_j = -half_w; iter_j <= half_w; iter_j++)\n"
                         "            {\n"
                         "                SD += (zimg[(i + iter_i) * (wi + w - 1) + (j + iter_j)] - m) * (zimg[(i + iter_i) * (wi + w - 1) + (j + iter_j)] - m);\n"
                         "            }\n"
                         "        }\n"
                         "\n"
                         "        sd = SD / (w * w);\n"
                         "        sd = sqrt(sd);\n"
                         "\n"
                         "        Z[(i - half_w) * wi + (j - half_w)] = sd / m;\n"
                         "    }\n"
                         "}\n"
                         "\n"
                         "__kernel void normalize_invert(__global float *Z, __global float *Z_val, __global float *avg_Z, float beta, float SG, int calib_flag, int std_max, int std_min)\n"
                         "{\n"
                         "    int i = get_global_id(0);\n"
                         "    int j = get_global_id(1);\n"
                         "\n"
                         "    if (i < h && j < wi)\n"
                         "    {\n"
                         "        Z_val[i * wi + j] = Z[i * wi + j];\n"
                         "        Z[i * wi + j] = (1.0 / (Z[i * wi + j] * beta * beta) - 1.0) * SG;\n"
                         "\n"
                         "        if (Z[i * wi + j] < 1.0)\n"
                         "        {\n"
                         "            Z[i * wi + j] = 1.0;\n"
                         "        }\n"
                         "        else if (Z[i * wi + j] > std_max)\n"
                         "        {\n"
                         "            Z[i * wi + j] = std_max;\n"
                         "        }\n"
                         "\n"
                         "        Z[i * wi + j] = Z[i * wi + j] / std_max;\n"
                         "        avg_Z[i * wi + j] += Z[i * wi + j];\n"
                         "    }\n"
                         "}\n"
                         "\n"
                         "__kernel void sum_Z(__global float *Z, __global float *Z_avg, __global float *Z_sum0, __global float *Z_sum1, int h, int wi)\n"
                         "{\n"
                         "    int i = get_global_id(0);\n"
                         "    int j = get_global_id(1);\n"
                         "\n"
                         "    if (i < h && j < wi)\n"
                         "    {\n"
                         "        Z_avg[i * wi + j] = (Z[i * wi + j] + Z_sum0[i * wi + j] + Z_sum1[i * wi + j]) / 3.0;\n"
                         "        Z_sum1[i * wi + j] = Z_sum0[i * wi + j];\n"
                         "        Z_sum0[i * wi + j] = Z[i * wi + j];\n"
                         "    }\n"
                         "}\n"
                         "\n"
                         "__kernel void grayscale_to_rgb_kernel(__global float *Z, __global uint8_t *lookup_table, __global rgb *buffer_rgb, int std_max, float cmap_min_val, float cmap_max_val)\n"
                         "{\n"
                         "    int idx = get_global_id(0);\n"
                         "    int idy = get_global_id(1);\n"
                         "\n"
                         "    if (idx < h && idy < wi)\n"
                         "    {\n"
                         "        float range_min = cmap_min_val / std_max;\n"
                         "        float range_max = cmap_max_val / std_max;\n"
                         "        float range = range_max - range_min;\n"
                         "        float rangeinv = 1.0 / range;\n"
                         "\n"
                         "        float grey = (Z[idx * wi + idy] > range_min) ? ((Z[idx * wi + idy] < range_max) ? ((Z[idx * wi + idy] - range_min) * rangeinv * 255.0) : 255.0) : 0.0;\n"
                         "\n"
                         "        buffer_rgb[idx * wi + idy].b = lookup_table[(int)grey * 3 + 2];\n"
                         "        buffer_rgb[idx * wi + idy].g = lookup_table[(int)grey * 3 + 1];\n"
                         "        buffer_rgb[idx * wi + idy].r = lookup_table[(int)grey * 3];\n"
                         "    }\n"
                         "}\n"
                         "\n"
                         "__kernel void compute_avg_Z(__global float *Z, __global uint8_t *lookup_table, __global rgb *buffer_rgb, int num_imgs, int std_max, float cmap_min_val, float cmap_max_val)\n"
                         "{\n"
                         "    int idx = get_global_id(0);\n"
                         "    int idy = get_global_id(1);\n"
                         "\n"
                         "    if (idx < h && idy < wi)\n"
                         "    {\n"
                         "        float temp = Z[idx * wi + idy] / num_imgs;\n"
                         "        float range_min = cmap_min_val / std_max;\n"
                         "        float range_max = cmap_max_val / std_max;\n"
                         "        float range = range_max - range_min;\n"
                         "        float rangeinv = 1.0 / range;\n"
                         "\n"
                         "        float grey = (temp > range_min) ? ((temp < range_max) ? ((temp - range_min) * rangeinv * 255.0) : 255.0) : 0.0;\n"
                         "\n"
                         "        buffer_rgb[idx * wi + idy].b = lookup_table[(int)grey * 3 + 2];\n"
                         "        buffer_rgb[idx * wi + idy].g = lookup_table[(int)grey * 3 + 1];\n"
                         "        buffer_rgb[idx * wi + idy].r = lookup_table[(int)grey * 3];\n"
                         "    }\n"
                         "}\n";

// Declare global variables
float *Z;
float *Z_val;
float *Z_avg;
float *Z_sum0;
float *Z_sum1;
rgb *buffer_rgb;

// Declare OpenCL variables
cl_context context;
cl_command_queue command_queue;
cl_program program;
cl_kernel fillZeroPaddedArray_kernel;
cl_kernel lsci_kernel;
cl_kernel normalize_invert_kernel;
cl_kernel sum_Z_kernel;
cl_kernel grayscale_to_rgb_kernel;
cl_kernel compute_avg_Z_kernel;

// Function to allocate host memory
void allocate_host_mem()
{
    w = window_size;
    half_w = w / 2;
    Z = (float *)malloc(wi * h * sizeof(float));
    Z_val = (float *)malloc(wi * h * sizeof(float));
    Z_avg = (float *)malloc(wi * h * sizeof(float));
    Z_sum0 = (float *)malloc(wi * h * sizeof(float));
    Z_sum1 = (float *)malloc(wi * h * sizeof(float));
    buffer_rgb = (rgb *)malloc(wi * h * sizeof(rgb));
}

// Function to allocate device memory
void allocate_device_mem()
{
    cl_int error_id;
    cl_mem dev_buffer;
    cl_mem dev_zimg;
    cl_mem d_lookup_table;
    cl_mem d_buffer_rgb;

    dev_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint16_t) * h * wi, NULL, &error_id);
    dev_zimg = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * (h + w - 1) * (wi + w - 1), NULL, &error_id);
    d_lookup_table = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * 3 * sizeof(uint8_t), NULL, &error_id);
    d_buffer_rgb = clCreateBuffer(context, CL_MEM_READ_WRITE, h * wi * sizeof(rgb), NULL, &error_id);

    clEnqueueWriteBuffer(command_queue, d_lookup_table, CL_TRUE, 0, 256 * 3 * sizeof(uint8_t), lookup_table, 0, NULL, NULL);

    clSetKernelArg(fillZeroPaddedArray_kernel, 0, sizeof(cl_mem), &dev_buffer);
    clSetKernelArg(fillZeroPaddedArray_kernel, 1, sizeof(cl_mem), &dev_zimg);
    clSetKernelArg(fillZeroPaddedArray_kernel, 2, sizeof(int), &w);
    clSetKernelArg(fillZeroPaddedArray_kernel, 3, sizeof(int), &h);
    clSetKernelArg(fillZeroPaddedArray_kernel, 4, sizeof(int), &wi);

    clSetKernelArg(lsci_kernel, 0, sizeof(cl_mem), &dev_zimg);
    clSetKernelArg(lsci_kernel, 1, sizeof(cl_mem), &d_Z);
    clSetKernelArg(lsci_kernel, 2, sizeof(int), &w);
    clSetKernelArg(lsci_kernel, 3, sizeof(int), &h);

    clSetKernelArg(normalize_invert_kernel, 0, sizeof(cl_mem), &d_Z);
    clSetKernelArg(normalize_invert_kernel, 1, sizeof(cl_mem), &d_Z_val);
    clSetKernelArg(normalize_invert_kernel, 2, sizeof(cl_mem), &d_avg_Z);
    clSetKernelArg(normalize_invert_kernel, 3, sizeof(float), &beta);
    clSetKernelArg(normalize_invert_kernel, 4, sizeof(float), &SG);
    clSetKernelArg(normalize_invert_kernel, 5, sizeof(int), &calib_flag);
    clSetKernelArg(normalize_invert_kernel, 6, sizeof(int), &std_max);
    clSetKernelArg(normalize_invert_kernel, 7, sizeof(int), &std_min);

    clSetKernelArg(sum_Z_kernel, 0, sizeof(cl_mem), &d_Z);
    clSetKernelArg(sum_Z_kernel, 1, sizeof(cl_mem), &d_Z_avg);
    clSetKernelArg(sum_Z_kernel, 2, sizeof(cl_mem), &d_Z_sum0);
    clSetKernelArg(sum_Z_kernel, 3, sizeof(cl_mem), &d_Z_sum1);
    clSetKernelArg(sum_Z_kernel, 4, sizeof(int), &h);
    clSetKernelArg(sum_Z_kernel, 5, sizeof(int), &wi);

    clSetKernelArg(grayscale_to_rgb_kernel, 0, sizeof(cl_mem), &d_Z);
    clSetKernelArg(grayscale_to_rgb_kernel, 1, sizeof(cl_mem), &d_lookup_table);
    clSetKernelArg(grayscale_to_rgb_kernel, 2, sizeof(cl_mem), &d_buffer_rgb);
    clSetKernelArg(grayscale_to_rgb_kernel, 3, sizeof(int), &std_max);
    clSetKernelArg(grayscale_to_rgb_kernel, 4, sizeof(float), &cmap_min_val);
    clSetKernelArg(grayscale_to_rgb_kernel, 5, sizeof(float), &cmap_max_val);

    clSetKernelArg(compute_avg_Z_kernel, 0, sizeof(cl_mem), &d_avg_Z);
    clSetKernelArg(compute_avg_Z_kernel, 1, sizeof(cl_mem), &d_lookup_table);
    clSetKernelArg(compute_avg_Z_kernel, 2, sizeof(cl_mem), &d_buffer_rgb);
    clSetKernelArg(compute_avg_Z_kernel, 3, sizeof(int), &num_imgs);
    clSetKernelArg(compute_avg_Z_kernel, 4, sizeof(int), &std_max);
    clSetKernelArg(compute_avg_Z_kernel, 5, sizeof(float), &cmap_min_val);
    clSetKernelArg(compute_avg_Z_kernel, 6, sizeof(float), &cmap_max_val);
}

// Function to initialize the GPU
int gpu_init()
{
    cl_int error_id;
    cl_uint num_platforms;
    cl_platform_id platform_id;
    cl_device_id device_id;

    // Get the number of available platforms
    error_id = clGetPlatformIDs(0, NULL, &num_platforms);
    if (error_id != CL_SUCCESS)
    {
        printf("clGetPlatformIDs failed with error code %d\n", error_id);
        return -1;
    }

    // Get the first available platform
    error_id = clGetPlatformIDs(1, &platform_id, NULL);
    if (error_id != CL_SUCCESS)
    {
        printf("clGetPlatformIDs failed with error code %d\n", error_id);
        return -1;
    }

    // Get the number of available devices
    error_id = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (error_id != CL_SUCCESS)
    {
        printf("clGetDeviceIDs failed with error code %d\n", error_id);
        return -1;
    }

    // Create a context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &error_id);
    if (error_id != CL_SUCCESS)
    {
        printf("clCreateContext failed with error code %d\n", error_id);
        return -1;
    }

    // Create a command queue
    command_queue = clCreateCommandQueue(context, device_id, 0, &error_id);
    if (error_id != CL_SUCCESS)
    {
        printf("clCreateCommandQueue failed with error code %d\n", error_id);
        return -1;
    }

    // Create a program from the source code
    program = clCreateProgramWithSource(context, 1, &source_str, NULL, &error_id);
    if (error_id != CL_SUCCESS)
    {
        printf("clCreateProgramWithSource failed with error code %d\n", error_id);
        return -1;
    }

    // Build the program
    error_id = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (error_id != CL_SUCCESS)
    {
        printf("clBuildProgram failed with error code %d\n", error_id);
        return -1;
    }

    // Create the kernels
    fillZeroPaddedArray_kernel = clCreateKernel(program, "fillZeroPaddedArray", &error_id);
    if (error_id != CL_SUCCESS)
    {
        printf("clCreateKernel fillZeroPaddedArray failed with error code %d\n", error_id);
        return -1;
    }

    lsci_kernel = clCreateKernel(program, "lsci_kernel", &error_id);
    if (error_id != CL_SUCCESS)
    {
        printf("clCreateKernel lsci_kernel failed with error code %d\n", error_id);
        return -1;
    }

    normalize_invert_kernel = clCreateKernel(program, "normalize_invert", &error_id);
    if (error_id != CL_SUCCESS)
    {
        printf("clCreateKernel normalize_invert failed with error code %d\n", error_id);
        return -1;
    }

    sum_Z_kernel = clCreateKernel(program, "sum_Z", &error_id);
    if (error_id != CL_SUCCESS)
    {
        printf("clCreateKernel sum_Z failed with error code %d\n", error_id);
        return -1;
    }

    grayscale_to_rgb_kernel = clCreateKernel(program, "grayscale_to_rgb_kernel", &error_id);
    if (error_id != CL_SUCCESS)
    {
        printf("clCreateKernel grayscale_to_rgb_kernel failed with error code %d\n", error_id);
        return -1;
    }

    compute_avg_Z_kernel = clCreateKernel(program, "compute_avg_Z", &error_id);
    if (error_id != CL_SUCCESS)
    {
        printf("clCreateKernel compute_avg_Z failed with error code %d\n", error_id);
        return -1;
    }

    return 0;
}

// Function to perform LSCI on the GPU
void gpu_lsci(uint16_t *buffer)
{
    cl_int error_id;
    cl_mem dev_buffer;
    cl_mem dev_zimg;
    cl_mem d_Z;
    cl_mem d_avg_Z;
    cl_mem d_Z_val;
    cl_mem d_Z_avg;
    cl_mem d_Z_sum0;
    cl_mem d_Z_sum1;
    cl_mem d_lookup_table;
    cl_mem d_buffer_rgb;

    // Create device buffers
    dev_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint16_t) * h * wi, NULL, &error_id);
    dev_zimg = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * (h + w - 1) * (wi + w - 1), NULL, &error_id);
    d_Z = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * h * wi, NULL, &error_id);
    d_avg_Z = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * h * wi, NULL, &error_id);
    d_Z_val = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * h * wi, NULL, &error_id);
    d_Z_avg = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * h * wi, NULL, &error_id);
    d_Z_sum0 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * h * wi, NULL, &error_id);
    d_Z_sum1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * h * wi, NULL, &error_id);
    d_lookup_table = clCreateBuffer(context, CL_MEM_READ_ONLY, 256 * 3 * sizeof(uint8_t), NULL, &error_id);
    d_buffer_rgb = clCreateBuffer(context, CL_MEM_READ_WRITE, h * wi * sizeof(rgb), NULL, &error_id);

    // Copy input array from host to device
    clEnqueueWriteBuffer(command_queue, dev_buffer, CL_TRUE, 0, sizeof(uint16_t) * h * wi, buffer, 0, NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(fillZeroPaddedArray_kernel, 0, sizeof(cl_mem), &dev_buffer);
    clSetKernelArg(fillZeroPaddedArray_kernel, 1, sizeof(cl_mem), &dev_zimg);
    clSetKernelArg(fillZeroPaddedArray_kernel, 2, sizeof(int), &w);
    clSetKernelArg(fillZeroPaddedArray_kernel, 3, sizeof(int), &h);
    clSetKernelArg(fillZeroPaddedArray_kernel, 4, sizeof(int), &wi);

    clSetKernelArg(lsci_kernel, 0, sizeof(cl_mem), &dev_zimg);
    clSetKernelArg(lsci_kernel, 1, sizeof(cl_mem), &d_Z);
    clSetKernelArg(lsci_kernel, 2, sizeof(int), &w);
    clSetKernelArg(lsci_kernel, 3, sizeof(int), &h);

    clSetKernelArg(normalize_invert_kernel, 0, sizeof(cl_mem), &d_Z);
    clSetKernelArg(normalize_invert_kernel, 1, sizeof(cl_mem), &d_Z_val);
    clSetKernelArg(normalize_invert_kernel, 2, sizeof(cl_mem), &d_avg_Z);
    clSetKernelArg(normalize_invert_kernel, 3, sizeof(float), &beta);
    clSetKernelArg(normalize_invert_kernel, 4, sizeof(float), &SG);
    clSetKernelArg(normalize_invert_kernel, 5, sizeof(int), &calib_flag);
    clSetKernelArg(normalize_invert_kernel, 6, sizeof(int), &std_max);
    clSetKernelArg(normalize_invert_kernel, 7, sizeof(int), &std_min);

    clSetKernelArg(sum_Z_kernel, 0, sizeof(cl_mem), &d_Z);
    clSetKernelArg(sum_Z_kernel, 1, sizeof(cl_mem), &d_Z_avg);
    clSetKernelArg(sum_Z_kernel, 2, sizeof(cl_mem), &d_Z_sum0);
    clSetKernelArg(sum_Z_kernel, 3, sizeof(cl_mem), &d_Z_sum1);
    clSetKernelArg(sum_Z_kernel, 4, sizeof(int), &h);
    clSetKernelArg(sum_Z_kernel, 5, sizeof(int), &wi);

    clSetKernelArg(grayscale_to_rgb_kernel, 0, sizeof(cl_mem), &d_Z);
    clSetKernelArg(grayscale_to_rgb_kernel, 1, sizeof(cl_mem), &d_lookup_table);
    clSetKernelArg(grayscale_to_rgb_kernel, 2, sizeof(cl_mem), &d_buffer_rgb);
    clSetKernelArg(grayscale_to_rgb_kernel, 3, sizeof(int), &std_max);
    clSetKernelArg(grayscale_to_rgb_kernel, 4, sizeof(float), &cmap_min_val);
    clSetKernelArg(grayscale_to_rgb_kernel, 5, sizeof(float), &cmap_max_val);

    clSetKernelArg(compute_avg_Z_kernel, 0, sizeof(cl_mem), &d_avg_Z);
    clSetKernelArg(compute_avg_Z_kernel, 1, sizeof(cl_mem), &d_lookup_table);
    clSetKernelArg(compute_avg_Z_kernel, 2, sizeof(cl_mem), &d_buffer_rgb);
    clSetKernelArg(compute_avg_Z_kernel, 3, sizeof(int), &std_max);
    clSetKernelArg(compute_avg_Z_kernel, 4, sizeof(float), &cmap_min_val);
    clSetKernelArg(compute_avg_Z_kernel, 5, sizeof(float), &cmap_max_val);
size_t global_work_size[2] = {h + w - 1, wi + w - 1};
size_t local_work_size[2] = {5, 5};
clEnqueueNDRangeKernel(command_queue, fillZeroPaddedArray_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

global_work_size[0] = h;
global_work_size[1] = wi;
local_work_size[0] = 32;
local_work_size[1] = 32;
clEnqueueNDRangeKernel(command_queue, lsci_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

clEnqueueNDRangeKernel(command_queue, normalize_invert_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

if (preview_flag != 0)
{
    clEnqueueNDRangeKernel(command_queue, sum_Z_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
}

clEnqueueNDRangeKernel(command_queue, grayscale_to_rgb_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

if (calib_flag != 0)
{
    clEnqueueNDRangeKernel(command_queue, compute_avg_Z_kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
}
    // Copy result from device to host
    clEnqueueReadBuffer(command_queue, d_buffer_rgb, CL_TRUE, 0, h * wi * sizeof(rgb), buffer_rgb, 0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, d_Z_val, CL_TRUE, 0, h * wi * sizeof(float), Z_val, 0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, d_Z_avg, CL_TRUE, 0, h * wi * sizeof(float), Z_avg, 0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, d_Z_sum0, CL_TRUE, 0, h * wi * sizeof(float), Z_sum0, 0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, d_Z_sum1, CL_TRUE, 0, h * wi * sizeof(float), Z_sum1, 0, NULL, NULL);

    // Clean up
    clReleaseMemObject(dev_buffer);
    clReleaseMemObject(dev_zimg);
    clReleaseMemObject(d_Z);
    clReleaseMemObject(d_avg_Z);
    clReleaseMemObject(d_Z_val);
    clReleaseMemObject(d_Z_avg);
    clReleaseMemObject(d_Z_sum0);
    clReleaseMemObject(d_Z_sum1);
    clReleaseMemObject(d_lookup_table);
    clReleaseMemObject(d_buffer_rgb);

    clReleaseProgram(program);
    clReleaseKernel(fillZeroPaddedArray_kernel);
    clReleaseKernel(lsci_kernel);
    clReleaseKernel(normalize_invert_kernel);
    clReleaseKernel(sum_Z_kernel);
    clReleaseKernel(grayscale_to_rgb_kernel);
    clReleaseKernel(compute_avg_Z_kernel);

    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}

int main(void)
{ 
// Allocate space for image
    uint16_t* image = (uint16_t*) malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(uint16_t));
                                                                                                                         
    camerainit(); // Initialize the camera
	 for(int i = 0; i < 10; i++) {
        camera(); // Capture image from camera
        uint16_t *processedImage = processImage(image, IMAGE_WIDTH, IMAGE_HEIGHT); // Process the image
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

        free(processedImage);
	 }
int gpu_init(void);
void gpu_lsci(uint16_t *buffer);
void colormap_avg_img(int);
}
