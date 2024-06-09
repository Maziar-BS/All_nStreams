#include<stdlib.h>
#include<cuda.h>
#include<assert.h>
#include<cuda_runtime.h>
#include<helper_cuda.h>
#include<helper_functions.h>
#include "device_launch_parameters.h"

int numElements = 50000;
size_t size;
float *h_A_Vecadd;
float *h_B_Vecadd;
float *h_C_Vecadd;
float *d_A_Vecadd = NULL;
float *d_B_Vecadd = NULL;
float *d_C_Vecadd = NULL;
int threadsPerBlock;
int blocksPerGrid;

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		C[i] = A[i] + B[i];
	}
}

void B1_Malloc(int argc, char** argv)
{
	cudaError_t err = cudaSuccess;
	
	printf("Enter Number of Elements for VecAdd Benchmark:\n");
	if (checkCmdLineFlag(argc, (const char **)argv, "NE"))
	{
		numElements = getCmdLineArgumentInt(argc, (const char **)argv, "NE");
	}
	size = numElements * sizeof(float);
	printf("[Vector addition of %d elements]\n", numElements);

	// Allocate the host input vector A
	cudaMallocHost((void **)&h_A_Vecadd, size);
	// Allocate the host input vector B
	cudaMallocHost((void **)&h_B_Vecadd, size);
	// Allocate the host output vector C
	cudaMallocHost((void **)&h_C_Vecadd, size);
	// Verify that allocations succeeded
	if (h_A_Vecadd == NULL || h_B_Vecadd == NULL || h_C_Vecadd == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A_Vecadd[i] = rand() / (float)RAND_MAX;
		h_B_Vecadd[i] = rand() / (float)RAND_MAX;
	}

	// Allocate the device input vector A
	
	err = cudaMalloc((void **)&d_A_Vecadd, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	
	err = cudaMalloc((void **)&d_B_Vecadd, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	
	err = cudaMalloc((void **)&d_C_Vecadd, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	threadsPerBlock = 256;
	blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	
}
void B1_H2D(cudaStream_t Stream)
{
	cudaError_t err = cudaSuccess;
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpyAsync(d_A_Vecadd, h_A_Vecadd, size, cudaMemcpyHostToDevice, Stream);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyAsync(d_B_Vecadd, h_B_Vecadd, size, cudaMemcpyHostToDevice, Stream);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void B1_Kernel(cudaStream_t Stream)
{
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock, 0, Stream >> >(d_A_Vecadd, d_B_Vecadd, d_C_Vecadd, numElements);
}
void B1_D2H(cudaStream_t Stream)
{
	cudaError_t err = cudaSuccess;
	err = cudaMemcpyAsync(h_C_Vecadd, d_C_Vecadd, size, cudaMemcpyDeviceToHost, Stream);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void B1_CL_Mem(cudaStream_t Stream)
{
	cudaStreamSynchronize(Stream);
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(h_A_Vecadd[i] + h_B_Vecadd[i] - h_C_Vecadd[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test PASSED\n");
	cudaFree(d_A_Vecadd);
	cudaFree(d_B_Vecadd);
	cudaFree(d_C_Vecadd);
	cudaFreeHost(h_A_Vecadd);
	cudaFreeHost(h_B_Vecadd);
	cudaFreeHost(h_C_Vecadd);
}