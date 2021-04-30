#include <stdio.h>
#include "cublas_v2.h"
#include "matrix_mul.h"

// Host multiplication function
// Compute C = A * B
// hA is the height of A
// wA is the width of A
// wB is the width of B

extern "C"
void Mul(float* A, float* B, int hA, int wA, int wB, float* C)
{
	int size;
	cublasHandle_t handle;
    const float alpha = 1.0f;
	const float beta  = 0.0f;

	// Load A and B to the device
	float* Ad;
	size = hA * wA * sizeof(float);
	cudaMalloc((void**)&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	float* Bd;
	size = wA * wB * sizeof(float);
	cudaMalloc((void**)&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	// Allocate C on the device
	float* Cd;
	size = hA * wB * sizeof(float);
	cudaMalloc((void**)&Cd, size);

    cublasCreate(&handle);
	// Compute the execution configuration
	cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
		hA,				/* [m] */ 
		wB,				/* [n] */  
		wA,				/* [k] */ 
		&alpha,			/* alfa */ 
		A, wA,			/* A[m][k], num columnas (lda) */ 
		B, wB,			/* B[k][n], num columnas (ldb) */
		&beta,			/* beta */
		C, wB			/* C[m][n], num columnas (ldc) */
	);

	// Read C from the device
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);
}
