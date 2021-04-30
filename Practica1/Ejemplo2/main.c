#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <math.h>

// Thread block size
#define BLOCK_SIZE 16 

// Forward declaration of the device multiplication function
__global__ void multMatrixGPU(float* A, float* B, int wA, int wB, float* C)

void Mul(float *A, float *B, int hA, int wA, int wB, float *C)
{
	int i,j,k;

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++){
			C[i*wB+j] = 0.0;
			for (k=0; k<wA; k++)
				C[i*wB+j] += A[i*wA+k]*B[k*wB+j];
		}
}

// Device multiplication function called by Mul()
// Compute C = A * B
// wA is the width of A
// wB is the width of B
__global__ void multMatrixGPU(float* A, float* B, int wA, int wB, float* C)
{
	int k;
	int i= threadIdx.y + blockIdx.y*blockDim.y;
	int j= threadIdx.x + blockIdx.x*blockDim.x;

	if((i < hA) && (j < wB) ) 
	{
		C[i*wB + j] = 0.0;
		for (k = 0; k < wA; k++)
			C[i*wB + j] += A[i*wA + k] * B[k*wA + j];
	}
}

void init_matrix(float *M, int hM, int wM, float k)
{
	int i,j;

	for (i=0; i<hM; i++)
		for (j=0; j<wM; j++)
			if (i==j)
				M[i*wM+j] = k*1.0f;
			else
				M[i*wM+j] = -1.0f/(float)(wM);
}

void print_matrix(float *M, int hM, int wM)
{
	int i,j;

	for (i=0; i<hM; i++){
//		printf("Line %i: ", i);
		for (j=0; j<wM; j++)
			printf("%4.1f ", M[i*wM+j]);
		printf("\n");
	}
}

int diff(float *A, float *B, int hA, int wA, int wB, float *C)
{
	float *C_cpu;
	int size_C = wB * hA;
	C_cpu = (float*)malloc(size_C*sizeof(float));

	int i,j,k;

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++){
			C_cpu[i*wB+j] = 0.0;
			for (k=0; k<wA; k++){
				C_cpu[i*wB+j] += A[i*wA+k]*B[k*wB+j];
			}
		}
	//printf("\n\nMATRIX C_cpu\n");print_matrix(C_cpu, hA, wB);

	for (i=0; i<hA; i++)
		for (j=0; j<wB; j++)
			if (fabsf(C_cpu[i*wB+j]-C[i*wB+j])>1e-5)
			{
				printf("[%i,%i]: %f!=%f\n", i, j, C_cpu[i*wB+j], C[i*wB+j]);
				return(0);
			}


	return(1);

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// Matrix variables
	float *A, *B, *C;
	int hA, wA, hB, wB;
	int i;

	setbuf(stdout, NULL);

	if (argc!=4){
		printf("./exec hA hB/WA wB\n");
		exit(-1);
	}

	hA = atoi(argv[1]);
	hB = wA = atoi(argv[2]);
	wB = atoi(argv[3]);

	// Init A and B, malloc C
	int size_A = wA * hA;
	A = (float*)malloc(size_A*sizeof(float));
	init_matrix(A, hA, wA, 1.0);

	int size_B = wB * hB;
	B = (float*)malloc(size_B*sizeof(float));
	init_matrix(B, hB, wB, 2.0);

	int size_C = wB * hA;
	C = (float*)malloc(size_C*sizeof(float));
	for (i = 0; i < (hA*wB); i++) {
		C[i] = 0.0;
	}


	Mul(A, B, hA, wA, wB, C);
	//printf("\n\nMATRIX A\n");print_matrix(A, hA, wA);
	//printf("\n\nMATRIX B\n");print_matrix(B, hB, wB);
	//printf("\n\nMATRIX C\n");print_matrix(C, hA, wB);

	if (!diff(A, B, hA, wA, wB, C))
		printf("ERROR=GPU.vs.CPU matrix mult differs\n");
	

	// print Matrix
	//printf("\n\nMATRIX A\n");print_matrix(A, hA, wA);
	//printf("\n\nMATRIX B\n");print_matrix(B, hB, wB);
	//printf("\n\nMATRIX C\n");print_matrix(C, hA, wB);

	return (1);
}

