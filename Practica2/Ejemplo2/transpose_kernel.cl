// transpose_kernel.cl
// Kernel source file for calculating the transpose of a matrix

__kernel
void matrixTranspose(__global float * output,
                     __global float * input,
                     const    uint    width)

{
	int idx = get_global_id(0);
	int idy = get_global_id(1);

	if (idx < width && idy < width)
		output[idy*width + idx] = input[idx*width + idy]; 
}


/* __kernel
void matrixTransposeLocal(__global float * output,
                          __global float * input,
                          ...,
                          const    uint    width)

{
	int idx = get_global_id(0);
	int idy = get_global_id(1);
	
	int blockIdxx = get_group_id(0);
	int blockIdxy = get_group_id(1);
	
	int threadIdxx = get_local_id(0);
	int threadIdxy = get_local_id(1);

	int pos = idx*width + idy;
	
	if(pos >= 0 && pos < width*width)
	{
		//Trasponemos al copiar
		block[threadIdxx*BLOCK_SIZE + threadIdxy] = input [idx*width + idy];

		//Copiamos el bloque traspuesto
		output[idy*width + idx] = block[threadIdxx* BLOCK_SIZE+ threadIdxy];					
	} 	
} */
