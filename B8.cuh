const char *B8_sSDKsample = "Transpose";
#define B8_TILE_DIM    16
#define B8_BLOCK_ROWS  16
int B8_MUL_FACTOR = B8_TILE_DIM;

#define B8_FLOOR(a,b) (a-(a%b))

#define B8_NUM_REPS  10

dim3 B8_threads;
dim3 B8_grid;

int B8_size_x, B8_size_y;
float *B8_h_idata;
float *B8_h_odata;
float *B8_d_idata;
float *B8_d_odata;
const char *B8_kernelName;
void(*B8_kernel)(float *, float *, int, int);
size_t B8_mem_size;
float *B8_transposeGold;
float *B8_gold;

__global__ void copySharedMem(float *odata, float *idata, int width, int height)
{
	__shared__ float tile[B8_TILE_DIM][B8_TILE_DIM];

	int xIndex = blockIdx.x * B8_TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * B8_TILE_DIM + threadIdx.y;

	int index = xIndex + width * yIndex;

	for (int i = 0; i<B8_TILE_DIM; i += B8_BLOCK_ROWS)
	{
		if (xIndex < width && yIndex < height)
		{
			tile[threadIdx.y][threadIdx.x] = idata[index];
		}
	}

	__syncthreads();

	for (int i = 0; i<B8_TILE_DIM; i += B8_BLOCK_ROWS)
	{
		if (xIndex < height && yIndex < width)
		{
			odata[index] = tile[threadIdx.y][threadIdx.x];
		}
	}
}

void B8_computeTransposeGold(float *gold, float *idata)
{
	for (int y = 0; y < B8_size_y; ++y)
	{
		for (int x = 0; x < B8_size_x; ++x)
		{
			gold[(x * B8_size_y) + y] = idata[(y * B8_size_x) + x];
		}
	}
}

void B8_getParams(int argc, char **argv, cudaDeviceProp &deviceProp, int max_tile_dim)
{
	// set matrix size (if (x,y) dim of matrix is not square, then this will have to be modified
	if (checkCmdLineFlag(argc, (const char **)argv, "CSMdimX"))
	{
		B8_size_x = getCmdLineArgumentInt(argc, (const char **)argv, "CSMdimX");

		if (B8_size_x > max_tile_dim)
		{
			printf("> MatrixSize X = %d is greater than the recommended size = %d\n", B8_size_x, max_tile_dim);
		}
		else
		{
			printf("> MatrixSize X = %d\n", B8_size_x);
		}
	}
	else
	{
		B8_size_x = max_tile_dim;
		B8_size_x = B8_FLOOR(B8_size_x, 512);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "CSMdimY"))
	{
		B8_size_y = getCmdLineArgumentInt(argc, (const char **)argv, "CSMdimY");

		if (B8_size_y > max_tile_dim)
		{
			printf("> MatrixSize Y = %d is greater than the recommended size = %d\n", B8_size_y, max_tile_dim);
		}
		else
		{
			printf("> MatrixSize Y = %d\n", B8_size_y);
		}
	}
	else
	{
		B8_size_y = max_tile_dim;

		// If this is SM12 hardware, we want to round down to a multiple of 512
		if ((deviceProp.major == 1 && deviceProp.minor >= 2) || deviceProp.major > 1)
		{
			B8_size_y = B8_FLOOR(B8_size_y, 512);
		}
		else     // else for SM10,SM11 we round down to a multiple of 384
		{
			B8_size_y = B8_FLOOR(B8_size_y, 384);
		}
	}
}


void
B8_showHelp()
{
	printf("\n%s : Command line options\n", B8_sSDKsample);
	printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
	printf("> The default matrix size can be overridden with these parameters\n");
	printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
	printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}

void B8_Malloc(int argc, char** argv)
{
	int MATRIX_SIZE_X = 1024;
	int MATRIX_SIZE_Y = 1024;
	if (checkCmdLineFlag(argc, (const char **)argv, "CSMdimX"))
	{
		MATRIX_SIZE_X = getCmdLineArgumentInt(argc, (const char **)argv, "CSMdimX");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "CSMdimY"))
	{
		MATRIX_SIZE_Y = getCmdLineArgumentInt(argc, (const char **)argv, "CSMdimY");
	}
	printf("Enter Matrix_Size_X & Matrix_Size_Y (TransposeCopySharedMem Benchmark):\n");
	int MAX_TILES = (B8_FLOOR(MATRIX_SIZE_X, 512) * B8_FLOOR(MATRIX_SIZE_Y, 512)) / (B8_TILE_DIM *B8_TILE_DIM);
	// Start logs

	printf("%s Starting...\n\n", B8_sSDKsample);

	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	{
		B8_showHelp();
	}

	int devID = findCudaDevice(argc, (const char **)argv);
	cudaDeviceProp deviceProp;

	checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	// compute the scaling factor (for GPUs with fewer MPs)
	float scale_factor, total_tiles;
	scale_factor = max((192.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount)), 1.0f);

	printf("> Device %d: \"%s\"\n", devID, deviceProp.name);
	printf("> SM Capability %d.%d detected:\n", deviceProp.major, deviceProp.minor);

	// Calculate number of tiles we will run for the Matrix Transpose performance tests
	int max_matrix_dim, matrix_size_test;

	matrix_size_test = 512;  // we round down max_matrix_dim for this perf test
	total_tiles = (float)MAX_TILES / scale_factor;

	max_matrix_dim = B8_FLOOR((int)(floor(sqrt(total_tiles))* B8_TILE_DIM), matrix_size_test);

	// This is the minimum size allowed
	if (max_matrix_dim == 0)
	{
		max_matrix_dim = matrix_size_test;
	}

	printf("> [%s] has %d MP(s) x %d (Cores/MP) = %d (Cores)\n",
		deviceProp.name, deviceProp.multiProcessorCount,
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

	printf("> Compute performance scaling factor = %4.2f\n", scale_factor);

	// Extract parameters if there are any, command line -dimx and -dimy can override
	// any of these settings
	B8_getParams(argc, argv, deviceProp, max_matrix_dim);

	if (B8_size_x != B8_size_y)
	{
		printf("\n[%s] does not support non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n", B8_sSDKsample, B8_size_x, B8_size_y);
		exit(EXIT_FAILURE);
	}

	if (B8_size_x%B8_TILE_DIM != 0 || B8_size_y % B8_TILE_DIM != 0)
	{
		printf("[%s] Matrix size must be integral multiple of tile size\nExiting...\n\n", B8_sSDKsample);
		exit(EXIT_FAILURE);
	}

	// execution configuration parameters
	dim3 grid(B8_size_x / B8_TILE_DIM, B8_size_y / B8_TILE_DIM), threads(B8_TILE_DIM, B8_BLOCK_ROWS);
	B8_grid = grid;
	B8_threads = threads;

	if (grid.x < 1 || grid.y < 1)
	{
		printf("[%s] grid size computation incorrect in test \nExiting...\n\n", B8_sSDKsample);
		exit(EXIT_FAILURE);
	}


	// size of memory required to store the matrix
	B8_mem_size = static_cast<size_t>(sizeof(float) * B8_size_x*B8_size_y);

	if (2 * B8_mem_size > deviceProp.totalGlobalMem)
	{
		printf("Input matrix size is larger than the available device memory!\n");
		printf("Please choose a smaller size matrix\n");
		exit(EXIT_FAILURE);
	}

	// allocate host memory

	cudaMallocHost((void **)&B8_h_idata, B8_mem_size);
	cudaMallocHost((void **)&B8_h_odata, B8_mem_size);
	cudaMallocHost((void **)&B8_transposeGold, B8_mem_size);
	if (B8_h_idata == NULL || B8_h_odata == NULL || B8_transposeGold == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// allocate device memory
	checkCudaErrors(cudaMalloc((void **)&B8_d_idata, B8_mem_size));
	checkCudaErrors(cudaMalloc((void **)&B8_d_odata, B8_mem_size));

	// initialize host data
	for (int i = 0; i < (B8_size_x*B8_size_y); ++i)
	{
		B8_h_idata[i] = (float)i;
	}
	B8_computeTransposeGold(B8_transposeGold, B8_h_idata);
	B8_kernel = &copySharedMem;
	B8_kernelName = "shared memory copy";

	B8_gold = B8_h_idata;

	// Clear error status
	checkCudaErrors(cudaGetLastError());

}
void B8_H2D(cudaStream_t Stream)
{
	cudaError_t err = cudaSuccess;
	printf("Copy input data from the host memory to the CUDA device\n");
	// copy host data to device
	err = cudaMemcpyAsync(B8_d_idata, B8_h_idata, B8_mem_size, cudaMemcpyHostToDevice, Stream);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector h_idata from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void B8_Kernel(cudaStream_t Stream)
{
	B8_kernel << <B8_grid, B8_threads, 0, Stream >> >(B8_d_odata, B8_d_idata, B8_size_x, B8_size_y);
}
void B8_D2H(cudaStream_t Stream)
{
	checkCudaErrors(cudaMemcpyAsync(B8_h_odata, B8_d_odata, B8_mem_size, cudaMemcpyDeviceToHost, Stream));
	printf("Copy output data from the CUDA device to the host memory\n");

}

void B8_CL_Mem(cudaStream_t Stream)
{
	cudaStreamSynchronize(Stream);
	printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
		B8_size_x, B8_size_y, B8_size_x / B8_TILE_DIM, B8_size_y / B8_TILE_DIM, B8_TILE_DIM, B8_TILE_DIM, B8_TILE_DIM, B8_BLOCK_ROWS);
	bool success = true;
	bool res = compareData(B8_gold, B8_h_odata, B8_size_x*B8_size_y, 0.01f, 0.0f);

	if (res == false)
	{
		printf("*** %s kernel FAILED ***\n", B8_kernelName);
		success = false;
	}
	printf("Test Passed!\n");
	cudaFree(B8_d_idata);
	cudaFree(B8_d_odata);
	cudaFreeHost(B8_h_idata);
	cudaFreeHost(B8_h_odata);
	cudaFreeHost(B8_transposeGold);
	if (!success)
	{
		printf("Test failed!\n");
		exit(EXIT_FAILURE);
	}

}
