int block_size;
float *h_A;
float *h_B;
float *h_C;
unsigned int mem_size_A;
unsigned int mem_size_B;
unsigned int mem_size_C;
float *d_A, *d_B, *d_C;
dim3 B2_threads;
dim3 B2_blocks;
int B2_A_x;
int B2_B_x;
int B2_C_x;
int B2_C_y;

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
	a <= aEnd;
		a += aStep, b += bStep)
	{

		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

void constantInit(float *data, int size, float val)
{
	for (int i = 0; i < size; ++i)
	{
		data[i] = val;
	}
}

void matrixMultiply(int argc, char **argv, dim3 &dimsA, dim3 &dimsB)
{
	// Error code to check return values for CUDA calls
	cudaError_t err;
	cudaError_t error;

	unsigned int size_A = dimsA.x * dimsA.y;
	mem_size_A = sizeof(float) * size_A;
	err = cudaMallocHost((void **)&h_A, mem_size_A);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	unsigned int size_B = dimsB.x * dimsB.y;
	mem_size_B = sizeof(float) * size_B;
	
	err = cudaMallocHost((void **)&h_B, mem_size_B);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate device memory
	

	// Allocate host matrix C
	dim3 dimsC(dimsB.x, dimsA.y, 1);

	B2_A_x = dimsA.x;
	B2_B_x = dimsB.x;
	B2_C_x = dimsC.x;
	B2_C_y = dimsC.y;

	mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	
	err = cudaMallocHost((void **)&h_C, mem_size_C);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate host memory for matrices A and B

	error = cudaMalloc((void **)&d_A, mem_size_A);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_B, mem_size_B);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_C, mem_size_C);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Initialize host memory
	const float valB = 0.01f;
	constantInit(h_A, size_A, 1.0f);
	constantInit(h_B, size_B, valB);
	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	B2_threads = threads;
	B2_blocks = grid;
}

void B2_Malloc(int argc, char** argv)
{
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
		checkCmdLineFlag(argc, (const char **)argv, "?"))
	{
		printf("Usage -device=n (n >= 0 for deviceID)\n");
		printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
		printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
		printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

		exit(EXIT_SUCCESS);
	}

	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	int devID = 0;

	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
		cudaSetDevice(devID);
	}

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	// Use a larger block size for Fermi and above
	block_size = (deviceProp.major < 2) ? 16 : 32;
	dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
	dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

	// width of Matrix A
	if (checkCmdLineFlag(argc, (const char **)argv, "wA"))
	{
		dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
	}

	// height of Matrix A
	if (checkCmdLineFlag(argc, (const char **)argv, "hA"))
	{
		dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
	}

	// width of Matrix B
	if (checkCmdLineFlag(argc, (const char **)argv, "wB"))
	{
		dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
	}

	// height of Matrix B
	if (checkCmdLineFlag(argc, (const char **)argv, "hB"))
	{
		dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
	}

	if (dimsA.x != dimsB.y)
	{
		printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
			dimsA.x, dimsB.y);
		exit(EXIT_FAILURE);
	}

	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

	matrixMultiply(argc, argv, dimsA, dimsB);

}
void B2_H2D(cudaStream_t Stream)
{
	cudaError_t error;
	// copy host memory to device
	printf("Copy output data from the CUDA device to the host memory\n");
	error = cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, Stream);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, Stream);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
}

void B2_Kernel(cudaStream_t Stream)
{
	cudaError_t error;
	printf("Computing result using CUDA Kernel...\n");

	if (block_size == 16)
	{
		matrixMulCUDA<16> << < B2_blocks, B2_threads, 0, Stream >> >(d_C, d_A, d_B, B2_A_x, B2_B_x);
	}
	else
	{
		matrixMulCUDA<32> << < B2_blocks, B2_threads, 0, Stream >> >(d_C, d_A, d_B, B2_A_x, B2_B_x);
	}
	error = cudaGetLastError();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	// Copy the device result vector in device memory to the host result vector
	// in host memory.
}

void B2_D2H(cudaStream_t Stream)
{
	cudaError_t error;
	error = cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, Stream);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
}

void B2_CL_Mem(cudaStream_t Stream)
{
	cudaStreamSynchronize(Stream);
	// Record the start event

	printf("Checking computed result for correctness: ");
	bool correct = true;
	const float valB = 0.01f;
	// test relative error by the formula
	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
	double eps = 1.e-6; // machine zero

	for (int i = 0; i < (int)(B2_C_x * B2_C_y); i++)
	{
		double abs_err = fabs(h_C[i] - (B2_A_x * valB));
		double dot_length = B2_A_x;
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err / abs_val / dot_length;

		if (rel_err > eps)
		{
			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], B2_A_x*valB, eps);
			correct = false;
		}
	}

	printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

	// Clean up memory
	cudaFreeHost(h_A);
	cudaFreeHost(h_B);
	cudaFreeHost(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");
}
