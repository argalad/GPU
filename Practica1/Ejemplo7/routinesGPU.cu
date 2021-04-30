#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

#define BLOCK_SIZE 16

__global__ void noiseReduction (uint8_t *im, float *NR, int height, int width)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x+2;
	int i = blockIdx.y * blockDim.y + threadIdx.y+2;
	int col = threadIdx.x+2;
	int row = threadIdx.y+2;

	__shared__ uint8_t subim[BLOCK_SIZE + 4][BLOCK_SIZE + 4];
	subim[row][col] = im[i*width + j];

	if (threadIdx.x == 0)
	{
		subim[threadIdx.y+2][threadIdx.x+1] = im[i*width + (j-1)];
		subim[threadIdx.y+2][threadIdx.x  ] = im[i*width + (j-2)];
	}
	else if (threadIdx.x == BLOCK_SIZE-1)
	{
		subim[threadIdx.y+2][threadIdx.x+3] = im[i*width + (j+1)];
		subim[threadIdx.y+2][threadIdx.x+4] = im[i*width + (j+2)];
	}

	if (threadIdx.y == 0)
	{
		subim[threadIdx.y+1][threadIdx.x+2] = im[(i-1)*width + j];
		subim[threadIdx.y  ][threadIdx.x+2] = im[(i-2)*width + j]; 
	}
	else if (threadIdx.y == BLOCK_SIZE-1)
	{
		subim[threadIdx.y+3][threadIdx.x+2] = im[(i+1)*width + j];
		subim[threadIdx.y+4][threadIdx.x+2] = im[(i+2)*width + j];
	}

	if (((i >= 2) && (i < height-2)) && ((j >= 2) && (j < width-2)))
	{
		NR[i*width + j] =
			(2.0*subim[row-2][col-2] +  4.0*subim[row-2][col-1] +  5.0*subim[row-2][col] +  4.0*subim[row-2][col+1] + 2.0*subim[row-2][col+2]
		   + 4.0*subim[row-1][col-2] +  9.0*subim[row-1][col-1] + 12.0*subim[row-1][col] +  9.0*subim[row-1][col+1] + 4.0*subim[row-1][col+2]
		   + 5.0*subim[row  ][col-2] + 12.0*subim[row  ][col-1] + 15.0*subim[row  ][col] + 12.0*subim[row  ][col+1] + 5.0*subim[row  ][col+2]
		   + 4.0*subim[row+1][col-2] +  9.0*subim[row+1][col-1] + 12.0*subim[row+1][col] +  9.0*subim[row+1][col+1] + 4.0*subim[row+1][col+2]
		   + 2.0*subim[row+2][col-2] +  4.0*subim[row+2][col-1] +  5.0*subim[row+2][col] +  4.0*subim[row+2][col+1] + 2.0*subim[row+2][col+2])
		   /159.0;
	}
}

__global__ void gradient (float *NR, float *G, float *phi, float *Gx, float *Gy, int height, int width)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	float PI = 3.141593;

	if(((i >= 2) && (i < height-2)) && ((j >= 2) && (j < width-2))) {
		Gx[i*width + j] = 
			 (1.0*NR[(i-2)*width + (j-2)] +  2.0*NR[(i-2)*width + (j-1)] +  (-2.0)*NR[(i-2)*width + (j+1)] + (-1.0)*NR[(i-2)*width + (j+2)]
			+ 4.0*NR[(i-1)*width + (j-2)] +  8.0*NR[(i-1)*width + (j-1)] +  (-8.0)*NR[(i-1)*width + (j+1)] + (-4.0)*NR[(i-1)*width + (j+2)]
			+ 6.0*NR[(i  )*width + (j-2)] + 12.0*NR[(i  )*width + (j-1)] + (-12.0)*NR[(i  )*width + (j+1)] + (-6.0)*NR[(i  )*width + (j+2)]
			+ 4.0*NR[(i+1)*width + (j-2)] +  8.0*NR[(i+1)*width + (j-1)] +  (-8.0)*NR[(i+1)*width + (j+1)] + (-4.0)*NR[(i+1)*width + (j+2)]
			+ 1.0*NR[(i+2)*width + (j-2)] +  2.0*NR[(i+2)*width + (j-1)] +  (-2.0)*NR[(i+2)*width + (j+1)] + (-1.0)*NR[(i+2)*width + (j+2)]);


		Gy[i*width + j] = 
			 ((-1.0)*NR[(i-2)*width + (j-2)] + (-4.0)*NR[(i-2)*width + (j-1)] +  (-6.0)*NR[(i-2)*width + (j)] + (-4.0)*NR[(i-2)*width + (j+1)] + (-1.0)*NR[(i-2)*width + (j+2)]
			+ (-2.0)*NR[(i-1)*width + (j-2)] + (-8.0)*NR[(i-1)*width + (j-1)] + (-12.0)*NR[(i-1)*width + (j)] + (-8.0)*NR[(i-1)*width + (j+1)] + (-2.0)*NR[(i-1)*width + (j+2)]
			+    2.0*NR[(i+1)*width + (j-2)] +    8.0*NR[(i+1)*width + (j-1)] +    12.0*NR[(i+1)*width + (j)] +    8.0*NR[(i+1)*width + (j+1)] +    2.0*NR[(i+1)*width + (j+2)]
			+    1.0*NR[(i+2)*width + (j-2)] +    4.0*NR[(i+2)*width + (j-1)] +     6.0*NR[(i+2)*width + (j)] +    4.0*NR[(i+2)*width + (j+1)] +    1.0*NR[(i+2)*width + (j+2)]);

		G[i*width + j] = sqrtf ((Gx[i*width + j]*Gx[i*width + j]) + (Gy[i*width + j]*Gy[i*width + j]));	//G = √Gx²+Gy²
		phi[i*width + j] = atan2f (fabs (Gy[i*width + j]), fabs (Gx[i*width + j]));

	if(fabs (phi[i*width + j]) <= PI/8 )
		phi[i*width + j] = 0;
	else if (fabs (phi[i*width + j]) <= 3*(PI/8))
		phi[i*width + j] = 45;
	else if (fabs (phi[i*width + j]) <= 5*(PI/8))
		phi[i*width + j] = 90;
	else if (fabs (phi[i*width + j]) <= 7*(PI/8))
		phi[i*width + j] = 135;
	else 
		phi[i*width + j] = 0;
	}
}

__global__ void pedge_calculation (float *G, float *phi, uint8_t *pedge, int height, int width)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (((i >= 3) && (i < height-3)) && ((j >= 3) && (j < width-3))) {
		if (phi[i*width + j] == 0)
		{
			if (G[i*width + j] > G[i*width + j+1] && G[i*width + j] > G[i*width + j-1]) //edge is in N-S
				pedge[i*width + j] = 1;

		} else if (phi[i*width + j] == 45) {
			if (G[i*width + j] > G[(i+1)*width + j+1] && G[i*width + j] > G[(i-1)*width + j-1]) // edge is in NW-SE
				pedge[i*width + j] = 1;

		} else if (phi[i*width + j] == 90) {
			if (G[i*width + j] > G[(i+1)*width + j] && G[i*width + j] > G[(i-1)*width + j]) //edge is in E-W
				pedge[i*width + j] = 1;

		} else if (phi[i*width + j] == 135) {
			if (G[i*width + j] > G[(i+1)*width + j-1] && G[i*width + j] > G[(i-1)*width + j+1]) // edge is in NE-SW
				pedge[i*width + j] = 1;
		}
	}
}

__global__ void hysteresis_thresholding (uint8_t *image_out, float *G, uint8_t *pedge, float level, int height, int width)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int ii, jj;
	float lowthres = level/2;
	float hithres = 2*level;

	if (((i >= 3) && (i < height-3)) && ((j >= 3) && (j < width-3))) {
		if (G[i*width + j] > hithres && pedge[i*width + j])
			image_out[i*width + j] = 255;
		else if (pedge[i*width + j] && G[i*width + j] >= lowthres && G[i*width + j] < hithres)
			// check neighbours 3x3
			for (ii = -1; ii <= 1; ii++)
				for (jj = -1; jj <= 1; jj++)
					if (G[(i + ii)*width + j + jj] > hithres)
						image_out[i*width + j] = 255;
	}
}

__global__ void houghtransform_GPU (uint8_t *im, int width, int height, uint32_t *accumulators, 
	float *sin_table, float *cos_table, float hough_h)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int theta;

	float center_x = width/2.0; 
	float center_y = height/2.0;

	if (((i >= 0) && (i < height)) && ((j >= 0) && (j < width)))
	{
		if (im[(i*width) + j] > 250) // Pixel is edge  
		{  
			for (theta = 0; theta < 180; theta++)  
			{  
				float rho = (((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
				accumulators[(int)((round (rho + hough_h)*180.0)) + theta]++;
			} 
		} 
	}
}

__global__ void getlines_GPU (int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int theta = blockIdx.x * blockDim.x + threadIdx.x;
	int rho = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t max;

	if (((rho >= 0) && (rho < accu_height)) && ((theta >= 0) && (theta < accu_width)))
	{	
		if(accumulators[(rho*accu_width) + theta] >= threshold)  
		{ 
			//Is this point a local maxima (9x9)  
			max = accumulators[(rho*accu_width) + theta]; 
			for (int ii = -4; ii <= 4; ii++)  
			{  
				for (int jj = -4; jj <= 4; jj++)  
				{  
					if ((ii + rho >= 0 && ii + rho <accu_height) && (jj + theta >= 0 && jj + theta < accu_width))  
					{  
						if (accumulators[((rho + ii) * accu_width) + (theta + jj)] > max )
							max = accumulators[((rho + ii) * accu_width) + (theta + jj)];
					}  
				}  
			}  

			if(max == accumulators[(rho*accu_width) + theta]) //local maxima
			{
				int x1, y1, x2, y2;  
				x1 = y1 = x2 = y2 = 0; 

				if(theta >= 45 && theta <= 135)  
				{
					if (theta > 90) {
						//y = (r - x cos(t)) / sin(t)  
						x1 = width/2;  
						y1 = ((float)(rho - (accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
						x2 = width;  
						y2 = ((float)(rho - (accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);  
					} else {
						//y = (r - x cos(t)) / sin(t)  
						x1 = 0;  
						y1 = ((float)(rho - (accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
						x2 = width * 2/5;  
						y2 = ((float)(rho - (accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2); 
					}
				} else {
					//x = (r - y sin(t)) / cos(t);  
					y1 = 0;  
					x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
					y2 = height;  
					x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
				}
				x1_lines[*lines] = x1;
				y1_lines[*lines] = y1;
				x2_lines[*lines] = x2;
				y2_lines[*lines] = y2;
				(*lines)++;
			}
		}
	}
}

void line_asist_GPU (uint8_t *im, int height, int width,
	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float *sin_table, float *cos_table,
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *x2, int *y1, int *y2, int *nlines)
{
	int img_size = height * width;
	uint8_t *im_gpu, *imEdge_gpu;
	float *NR_gpu;
	float *G_gpu;
	float *phi_gpu;
	float *Gx_gpu;
	float *Gy_gpu;
	uint8_t *pedge_gpu;
	uint32_t *accum_gpu;
	float *sin_table_gpu, *cos_table_gpu;
	int threshold;

	cudaMalloc ((uint8_t**)&im_gpu, sizeof(uint8_t) * img_size);
	cudaMalloc ((uint8_t**)&imEdge_gpu, sizeof (uint8_t) * img_size);
	cudaMalloc ((float**)&NR_gpu, sizeof (float) * img_size);
	cudaMalloc ((float**)&G_gpu, sizeof (float) * img_size);
	cudaMalloc ((float**)&phi_gpu, sizeof (float) * img_size);
	cudaMalloc ((float**)&Gx_gpu, sizeof (float) * img_size);
	cudaMalloc ((float**)&Gy_gpu, sizeof (float) * img_size);
	cudaMalloc ((uint8_t**)&pedge_gpu, sizeof (uint8_t) * img_size);
	cudaMalloc ((uint32_t**)&accum_gpu, sizeof (uint32_t) * accu_width * accu_height);
	cudaMalloc ((float**)&sin_table_gpu, sizeof (float) * 180);
	cudaMalloc ((float**)&cos_table_gpu, sizeof (float) * 180);

	cudaMemcpy (im_gpu, im, sizeof (uint8_t) * img_size, cudaMemcpyHostToDevice);

	dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE);

	int dimblock1;
	if (height % BLOCK_SIZE == 0)
		dimblock1 = height/BLOCK_SIZE;
	else
		dimblock1 = height/BLOCK_SIZE+1;

	int dimblock2;
	if (width % BLOCK_SIZE == 0)
		dimblock2 = width/BLOCK_SIZE;
	else
		dimblock2 = width/BLOCK_SIZE+1;

	dim3 dimGrid(dimblock1, dimblock2);

	/* Canny */
	noiseReduction<<<dimGrid, dimBlock>>> (im_gpu, NR_gpu, height, width);
	cudaDeviceSynchronize ();

	gradient<<<dimGrid, dimBlock>>> (NR_gpu, G_gpu, phi_gpu, Gx_gpu, Gy_gpu, height, width);
	cudaDeviceSynchronize ();

	pedge_calculation<<<dimGrid, dimBlock>>> (G_gpu, phi_gpu, pedge_gpu, height, width);
	cudaDeviceSynchronize ();

	hysteresis_thresholding<<<dimGrid, dimBlock>>> (imEdge_gpu, G_gpu, pedge_gpu, 1000.0f, height, width);
	cudaDeviceSynchronize ();

	cudaMemcpy (sin_table_gpu, sin_table, sizeof (float) * 180, cudaMemcpyHostToDevice);
	cudaMemcpy (cos_table_gpu, cos_table, sizeof (float) * 180, cudaMemcpyHostToDevice);
	
	for(int k = 0; k < accu_width * accu_height; k++)
		accum[k] = 0;

	cudaMemcpy (accum_gpu, accum, sizeof (uint32_t) * accu_width * accu_height, cudaMemcpyHostToDevice);

	/* hough transform */
	float hough_h = ((sqrt(2.0) * (float)(height > width ? height : width)) / 2.0);	
	houghtransform_GPU<<<dimGrid, dimBlock>>> (imEdge_gpu, width, height, accum_gpu, sin_table_gpu, cos_table_gpu, hough_h);
	cudaDeviceSynchronize ();

	if (width > height) threshold = width/6;
	else threshold = height/6;

	getlines_GPU<<<dimGrid, dimBlock>>> (threshold, accum_gpu, accu_width, accu_height, width, height,
		sin_table_gpu, cos_table_gpu,
		x1, y1, x2, y2, nlines);
	cudaDeviceSynchronize ();

	cudaFree (im_gpu);
	cudaFree (imEdge_gpu);
	cudaFree (NR_gpu);
	cudaFree (G_gpu);
	cudaFree (Gx_gpu);
	cudaFree (Gy_gpu);
	cudaFree (phi_gpu);
	cudaFree (pedge_gpu);
	cudaFree (accum_gpu);
	cudaFree (sin_table_gpu);
	cudaFree (cos_table_gpu);
}
