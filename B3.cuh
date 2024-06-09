float *h_A_scalarProd, *h_B_scalarProd, *h_C_scalarProd_CPU, *h_C_scalarProd_GPU;
float *d_A_scalarProd, *d_B_scalarProd, *d_C_scalarProd;
double delta_scalarProd, ref_scalarProd, sum_delta_scalarProd, sum_ref_scalarProd, L1norm_scalarProd;

#define B3_IMUL(a, b) __mul24(a, b)

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
// Parameters restrictions:
// 1) ElementN is strongly preferred to be a multiple of warp size to
//    meet alignment constraints of memory coalescing.
// 2) ACCUM_N must be a power of two.
///////////////////////////////////////////////////////////////////////////////
#define ACCUM_N 1024
__global__ void scalarProdGPU(
	float *d_C,
	float *d_A,
	float *d_B,
	int vectorN,
	int elementN
	)
{
	//Accumulators cache
	__shared__ float accumResult[ACCUM_N];

	////////////////////////////////////////////////////////////////////////////
	// Cycle through every pair of vectors,
	// taking into account that vector counts can be different
	// from total number of thread blocks
	////////////////////////////////////////////////////////////////////////////
	for (int vec = blockIdx.x; vec < vectorN; vec += gridDim.x)
	{
		int vectorBase = B3_IMUL(elementN, vec);
		int vectorEnd = vectorBase + elementN;

		////////////////////////////////////////////////////////////////////////
		// Each accumulator cycles through vectors with
		// stride equal to number of total number of accumulators ACCUM_N
		// At this stage ACCUM_N is only preferred be a multiple of warp size
		// to meet memory coalescing alignment constraints.
		////////////////////////////////////////////////////////////////////////
		for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x)
		{
			float sum = 0;

			for (int pos = vectorBase + iAccum; pos < vectorEnd; pos += ACCUM_N)
				sum += d_A[pos] * d_B[pos];

			accumResult[iAccum] = sum;
		}

		////////////////////////////////////////////////////////////////////////
		// Perform tree-like reduction of accumulators' results.
		// ACCUM_N has to be power of two at this stage
		////////////////////////////////////////////////////////////////////////
		for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();

			for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x)
				accumResult[iAccum] += accumResult[stride + iAccum];
		}

		if (threadIdx.x == 0) d_C[vec] = accumResult[0];
	}
}
////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
	float t = (float)rand() / (float)RAND_MAX;
	return (1.0f - t) * low + t * high;
}
///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C"
void scalarProdCPU(
	float *h_C,
	float *h_A,
	float *h_B,
	int vectorN,
	int elementN
	)
{
	for (int vec = 0; vec < vectorN; vec++)
	{
		int vectorBase = elementN * vec;
		int vectorEnd = vectorBase + elementN;

		double sum = 0;

		for (int pos = vectorBase; pos < vectorEnd; pos++)
			sum += h_A[pos] * h_B[pos];

		h_C[vec] = (float)sum;
	}
}



///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
///////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////

//Total number of input vector pairs; arbitrary
const int VECTOR_N = 256;
//Number of elements per vector; arbitrary,
//but strongly preferred to be a multiple of warp size
//to meet memory coalescing constraints
int ELEMENT_N = 1048576;
//Total number of data elements
int    DATA_N = VECTOR_N * ELEMENT_N;

int   DATA_SZ = DATA_N * sizeof(float);
int RESULT_SZ = VECTOR_N  * sizeof(float);


void B3_Malloc(int argc, char** argv)
{
	//scalarProd
	if (checkCmdLineFlag(argc, (const char **)argv, "ELEMENT_N"))
	{
		ELEMENT_N = getCmdLineArgumentInt(argc, (const char **)argv, "ELEMENT_N");
	}

	DATA_N = VECTOR_N * ELEMENT_N;

	DATA_SZ = DATA_N * sizeof(float);
	RESULT_SZ = VECTOR_N  * sizeof(float);

	printf("%s Starting...\n\n", argv[0]);

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	findCudaDevice(argc, (const char **)argv);

	printf("Initializing data...\n");
	printf("...allocating CPU memory.\n");
	cudaMallocHost((void**)&h_A_scalarProd, DATA_SZ);
	cudaMallocHost((void**)&h_B_scalarProd, DATA_SZ);
	cudaMallocHost((void**)&h_C_scalarProd_GPU, RESULT_SZ);
	h_C_scalarProd_CPU = (float *)malloc(RESULT_SZ);

	printf("...allocating GPU memory.\n");
	checkCudaErrors(cudaMalloc((void **)&d_A_scalarProd, DATA_SZ));
	checkCudaErrors(cudaMalloc((void **)&d_B_scalarProd, DATA_SZ));
	checkCudaErrors(cudaMalloc((void **)&d_C_scalarProd, RESULT_SZ));

	printf("...generating input data in CPU mem.\n");
	srand(123);

	//Generating input data on CPU
	for (int i = 0; i < DATA_N; i++)
	{
		h_A_scalarProd[i] = RandFloat(0.0f, 1.0f);
		h_B_scalarProd[i] = RandFloat(0.0f, 1.0f);
	}

	printf("...copying input data to GPU mem.\n");
	//Copy options data to GPU memory for further processing
}
void B3_H2D(cudaStream_t Stream)
{
	printf("Copy input data from the host memory to the CUDA device\n");

	checkCudaErrors(cudaMemcpyAsync(d_A_scalarProd, h_A_scalarProd, DATA_SZ, cudaMemcpyHostToDevice, Stream));
	checkCudaErrors(cudaMemcpyAsync(d_B_scalarProd, h_B_scalarProd, DATA_SZ, cudaMemcpyHostToDevice, Stream));
}

void B3_Kernel(cudaStream_t Stream)
{
	printf("Executing GPU kernel...\n");


	scalarProdGPU << <128, 256, 0, Stream >> >(d_C_scalarProd, d_A_scalarProd, d_B_scalarProd, VECTOR_N, ELEMENT_N);
}
void B3_D2H(cudaStream_t Stream)
{
	checkCudaErrors(cudaMemcpyAsync(h_C_scalarProd_GPU, d_C_scalarProd, RESULT_SZ, cudaMemcpyDeviceToHost, Stream));
}

void B3_CL_Mem(cudaStream_t Stream)
{
	checkCudaErrors(cudaStreamSynchronize(Stream));
	printf("..running CPU scalar product calculation\n");
	scalarProdCPU(h_C_scalarProd_CPU, h_A_scalarProd, h_B_scalarProd, VECTOR_N, ELEMENT_N);

	printf("...comparing the results\n");
	//Calculate max absolute difference and L1 distance
	//between CPU and GPU results
	sum_delta_scalarProd = 0;
	sum_ref_scalarProd = 0;

	for (int i = 0; i < VECTOR_N; i++)
	{
		delta_scalarProd = fabs(h_C_scalarProd_GPU[i] - h_C_scalarProd_CPU[i]);
		ref_scalarProd = h_C_scalarProd_CPU[i];
		sum_delta_scalarProd += delta_scalarProd;
		sum_ref_scalarProd += ref_scalarProd;
	}

	L1norm_scalarProd = sum_delta_scalarProd / sum_ref_scalarProd;

	printf("Shutting down...\n");
	checkCudaErrors(cudaFree(d_C_scalarProd));
	checkCudaErrors(cudaFree(d_B_scalarProd));
	checkCudaErrors(cudaFree(d_A_scalarProd));
	cudaFreeHost(h_C_scalarProd_GPU);
	cudaFreeHost(h_B_scalarProd);
	cudaFreeHost(h_A_scalarProd);
	free(h_C_scalarProd_CPU);

	printf("L1 error: %E\n", L1norm_scalarProd);
	printf((L1norm_scalarProd < 1e-6) ? "Test passed\n" : "Test failed!\n");
}