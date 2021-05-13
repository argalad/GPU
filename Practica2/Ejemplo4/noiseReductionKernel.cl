#define MAX_WINDOW_SIZE 5*5

__kernel 
void noiseReduction (__global float *im, __global float * im_out, 
								float threshold, int window_size, 
								int height, int width)
{
	int idx = get_global_id(0);
	int idy = get_global_id(1);

	float window[MAX_WINDOW_SIZE];
	float median;
	int ws2 = (window_size-1)>>1;

	im_out[idy * width + idx] = 0.0;

	if (idx >= ws2 && idx < width - ws2 && idy >= ws2 && idy < height - ws2)
	{
		for (int ii = -ws2; ii <= ws2; ii++)
			for (int jj = -ws2; jj <= ws2; jj++)
				window[(ii + ws2)*window_size + jj + ws2] = im[(idy + ii)*width + idx + jj];

		int i, j;
		int size = window_size * window_size;
		float tmp;

		for (i = 1; i < size; i++)
			for (j = 0 ; j < size - i; j++)
				if (window[j] > window[j+1])
				{
					tmp = window[j];
					window[j] = window[j+1];
					window[j+1] = tmp;
				}

		median = window[(window_size*window_size-1)>>1];

		if (fabs ((median-im[idy*width + idx]) / median) <= threshold)
			im_out[idy*width + idx] = im[idy*width + idx];
		else
			im_out[idy*width + idx] = median;
	}
}