#include "common.c"
#include <stdio.h>
#include "my_ocl.h"
#include <CL/cl.h>

#define BLOCK_SIZE 32

void remove_noiseOCL(float *im, float *image_out, 
	float threshold, int window_size,
	int height, int width)
{
	// OpenCL host variables
	cl_uint num_devs_returned;
	cl_context_properties properties[3];
	cl_device_id device_id;					// compute device id
	cl_int err;								// error code returned from OpenCL calls
	cl_platform_id platform_id;				// compute platform id
	cl_uint num_platforms_returned;			// compute num platforms
	cl_context context;						// compute context
	cl_command_queue command_queue;			// compute command queue
	cl_program program;						// compute program
	cl_kernel kernel;						// compute kernel
	size_t global[2];                  		// global domain size
	size_t local[2];						// local domain size

	cl_mem image_input;
	cl_mem image_output;

	// Variables used to read kernel source file
	FILE *fp;
	long filelen;
	long readlen;
	char *kernel_src;  // char string to hold kernel source

	// Read the kernel
	fp = fopen ("noiseReductionKernel.cl", "r");
	fseek (fp, 0L, SEEK_END);
	filelen = ftell (fp);
	rewind (fp);

	kernel_src = malloc (sizeof (char) * (filelen + 1));
	readlen = fread (kernel_src, 1, filelen, fp);

	if (readlen != filelen)
	{
		printf ("Error reading filen\n");
		exit (1);
	}

	// Ensure the string is null terminated
	kernel_src[filelen] = '\0';

	/************** 1. **************/
    // Definir la/s plataforma/s = devices + context + queues

	// Set up platform and GPU device
	cl_uint numPlatforms;

	// Find number of platforms
	err = clGetPlatformIDs (0, NULL, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf ("Error: failed to find a platform!\n%s\n", err_code (err));
		return;
	}

	// Get all platforms
	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs (numPlatforms, Platform, NULL);

	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf ("Error: failed to get the platform!\n%s\n", err_code (err));
		return;
	}

	// Secure a GPU
	int i;
	for (int i = 0; i < numPlatforms; i++)
	{
		err = clGetDeviceIDs (Platform[i], DEVICE, 1, &device_id, NULL);
		if (err == CL_SUCCESS)
			break;
	}

	if (device_id == NULL)
	{
		printf ("Error: Failed to create a device group!\n%s\n", err_code (err));
		return;
	}

	err = output_device_info (device_id);

	// Create a compute context 
	context = clCreateContext (0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf ("Error: Failed to create a compute context!\n%s\n", err_code (err));
		return;
	}

	// Create a command queue
	cl_command_queue commands = clCreateCommandQueue (context, device_id, 0, &err);
	if (!commands)
	{
		printf ("Error: Failed to create a command commands!\n%s\n", err_code (err));
		return;
	}

	// create command queue 
	command_queue = clCreateCommandQueue (context, device_id, 0, &err);
	if (err != CL_SUCCESS)
	{	
		printf ("Unable to create command queue. Error Code=%d\n", err);
		exit (1);
	}

	/************** 2. **************/
    // Crear y construir el programa ---> kernels como librerías dinámicas

    // Create program object from source. 
	// kernel_src contains source read from file earlier
	program = clCreateProgramWithSource (context, 1 , (const char **)
                                          &kernel_src, NULL, &err);
	if (err != CL_SUCCESS)
	{	
		printf ("Unable to create program object. Error Code=%d\n", err);
		exit (1);
	}       
	
	err = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
        printf ("Build failed. Error Code=%d\n", err);
		// Determine the size of the log
        size_t len;
	    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

	    // Allocate memory for the log
	    char *log = (char *) malloc(len);

	    // Get the log
		clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG,
                                  len, log, NULL);
		printf ("--- Build Log -- \n %s\n", log);
		exit (1);
	}

	kernel = clCreateKernel (program, "noiseReduction", &err);
	if (err != CL_SUCCESS)
	{	
		printf ("Unable to create kernel object. Error Code=%d\n", err);
		exit (1);
	}

	/************** 3. **************/
    // Gestionar los objetos de memoria

    const int imgSize = sizeof (float) * height * width;

	// Create buffer objects to input and output args of kernel function
	image_input = clCreateBuffer (context, CL_MEM_READ_ONLY, imgSize, NULL, NULL);
	image_output = clCreateBuffer (context, CL_MEM_WRITE_ONLY, imgSize, NULL, NULL);

	err = clEnqueueWriteBuffer(commands, image_input, CL_TRUE, 0, imgSize, im, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf ("Error: failed to write the source image into image_input!\n%s\n", err_code (err));
		exit (1);
	}

	// Set the kernel arguments
	if (clSetKernelArg (kernel, 0, sizeof (cl_mem), &image_input) ||
		clSetKernelArg (kernel, 1, sizeof (cl_mem), &image_output) ||
		clSetKernelArg (kernel, 2, sizeof (cl_float), &threshold) ||
		clSetKernelArg (kernel, 3, sizeof (cl_int), &window_size) ||
		clSetKernelArg (kernel, 4, sizeof (cl_int), &height) ||
		clSetKernelArg (kernel, 5, sizeof (cl_int), &width) != CL_SUCCESS)
	{
		printf ("Unable to set kernel arguments. Error code = %d\n", err);
		exit (1);
	}

	// Set the global work dimension size
	global[0] = width;
	global[1] = height;

	local[0] = BLOCK_SIZE;
	local[1] = BLOCK_SIZE;

	double t0d = getMicroSeconds ();
	err = clEnqueueNDRangeKernel (command_queue, kernel, 2, NULL, 
                               global, local, 0, NULL, NULL);
	double t1d = getMicroSeconds ();

	if (err != CL_SUCCESS)
	{
		printf ("Unable to enqueue kernel command. Error code = %d\n", err);
		exit (1);
	}

	// Wait for the command to finish
	clFinish (command_queue);

	// The output back to host memory
	err = clEnqueueReadBuffer(commands, image_output, CL_TRUE, 0, imgSize, image_out, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf ("Error enqueuing read buffer command. Error code = %d\n", err);
		exit (1);
	}

	// Clean up
	clReleaseProgram (program);
	clReleaseKernel (kernel);
	clReleaseCommandQueue (command_queue);
	clReleaseContext (context);
	free (kernel_src);
	clReleaseMemObject (image_input);
	clReleaseMemObject (image_output);
}
