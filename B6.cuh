const char *B6_sSDKsample = "Transpose";
#define B6_TILE_DIM    16
#define B6_BLOCK_ROWS  16
int B6_MUL_FACTOR = B6_TILE_DIM;

#define B6_FLOOR(a,b) (a-(a%b))

#define B6_NUM_REPS  10

dim3 B6_threads;
dim3 B6_grid;

int B6_size_x, B6_size_y;
float *B6_h_idata;
float *B6_h_odata;
float *B6_d_idata;
float *B6_d_odata;
const char *B6_kernelName;
void(*B6_kernel)(float *, float *, int, int);
size_t B6_mem_size;
float *B6_transposeGold;
float *B6_gold;

__global__ void transposeCoalesced(float *odata, float *idata, int width, int height)
{
	__shared__ float tile[B6_TILE_DIM][B6_TILE_DIM];

	int xIndex = blockIdx.x * B6_TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * B6_TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;

	xIndex = blockIdx.y * B6_TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * B6_TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;

	for (int i = 0; i<B6_TILE_DIM; i += B6_BLOCK_ROWS)
	{
		tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
	}

	__syncthreads();

	for (int i = 0; i<B6_TILE_DIM; i += B6_BLOCK_ROWS)
	{
		odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
	}
}

void B6_computeTransposeGold(float *gold, float *idata)
{
	for (int y = 0; y < B6_size_y; ++y)
	{
		for (int x = 0; x < B6_size_x; ++x)
		{
			gold[(x * B6_size_y) + y] = idata[(y * B6_size_x) + x];
		}
	}
}

void B6_getParams(int argc, char **argv, cudaDeviceProp &deviceProp, int max_tile_dim)
{
	// set matrix size (if (x,y) dim of matrix is not square, then this will have to be modified
	if (checkCmdLineFlag(argc, (const char **)argv, "TCdimX"))
	{
		B6_size_x = getCmdLineArgumentInt(argc, (const char **)argv, "TCdimX");

		if (B6_size_x > max_tile_dim)
		{
			printf("> MatrixSize X = %d is greater than the recommended size = %d\n", B6_size_x, max_tile_dim);
		}
		else
		{
			printf("> MatrixSize X = %d\n", B6_size_x);
		}
	}
	else
	{
		B6_size_x = max_tile_dim;
		B6_size_x = B6_FLOOR(B6_size_x, 512);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "TCdimY"))
	{
		B6_size_y = getCmdLineArgumentInt(argc, (const char **)argv, "TCdimY");

		if (B6_size_y > max_tile_dim)
		{
			printf("> MatrixSize Y = %d is greater than the recommended size = %d\n", B6_size_y, max_tile_dim);
		}
		else
		{
			printf("> MatrixSize Y = %d\n", B6_size_y);
		}
	}
	else
	{
		B6_size_y = max_tile_dim;

		// If this is SM12 hardware, we want to round down to a multiple of 512
		if ((deviceProp.major == 1 && deviceProp.minor >= 2) || deviceProp.major > 1)
		{
			B6_size_y = B6_FLOOR(B6_size_y, 512);
		}
		else     // else for SM10,SM11 we round down to a multiple of 384
		{
			B6_size_y = B6_FLOOR(B6_size_y, 384);
		}
	}
}


void
B6_showHelp()
{
	printf("\n%s : Command line options\n", B6_sSDKsample);
	printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
	printf("> The default matrix size can be overridden with these parameters\n");
	printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
	printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}

void B6_Malloc(int argc, char** argv)
{
	int MATRIX_SIZE_X = 1024;
	int MATRIX_SIZE_Y = 1024;
	if (checkCmdLineFlag(argc, (const char **)argv, "TCdimX"))
	{
		MATRIX_SIZE_X = getCmdLineArgumentInt(argc, (const char **)argv, "TCdimX");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "TCdimY"))
	{
		MATRIX_SIZE_Y = getCmdLineArgumentInt(argc, (const char **)argv, "TCdimY");
	}
	printf("Enter Matrix_Size_X & Matrix_Size_Y (TransposeCoalesced Benchmark):\n");
	int MAX_TILES = (B6_FLOOR(MATRIX_SIZE_X, 512) * B6_FLOOR(MATRIX_SIZE_Y, 512)) / (B6_TILE_DIM *B6_TILE_DIM);
	// Start logs

	printf("%s Starting...\n\n", B6_sSDKsample);

	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	{
		B6_showHelp();
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

	max_matrix_dim = B6_FLOOR((int)(floor(sqrt(total_tiles))* B6_TILE_DIM), matrix_size_test);

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
	B6_getParams(argc, argv, deviceProp, max_matrix_dim);

	if (B6_size_x != B6_size_y)
	{
		printf("\n[%s] does not support non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n", B6_sSDKsample, B6_size_x, B6_size_y);
		exit(EXIT_FAILURE);
	}

	if (B6_size_x%B6_TILE_DIM != 0 || B6_size_y % B6_TILE_DIM != 0)
	{
		printf("[%s] Matrix size must be integral multiple of tile size\nExiting...\n\n", B6_sSDKsample);
		exit(EXIT_FAILURE);
	}

	// execution configuration parameters
	dim3 grid(B6_size_x / B6_TILE_DIM, B6_size_y / B6_TILE_DIM), threads(B6_TILE_DIM, B6_BLOCK_ROWS);
	B6_grid = grid;
	B6_threads = threads;

	if (grid.x < 1 || grid.y < 1)
	{
		printf("[%s] grid size computation incorrect in test \nExiting...\n\n", B6_sSDKsample);
		exit(EXIT_FAILURE);
	}


	// size of memory required to store the matrix
	B6_mem_size = static_cast<size_t>(sizeof(float) * B6_size_x*B6_size_y);

	if (2 * B6_mem_size > deviceProp.totalGlobalMem)
	{
		printf("Input matrix size is larger than the available device memory!\n");
		printf("Please choose a smaller size matrix\n");
		exit(EXIT_FAILURE);
	}

	// allocate host memory

	cudaMallocHost((void **)&B6_h_idata, B6_mem_size);
	cudaMallocHost((void **)&B6_h_odata, B6_mem_size);
	cudaMallocHost((void **)&B6_transposeGold, B6_mem_size);
	if (B6_h_idata == NULL || B6_h_odata == NULL || B6_transposeGold == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// allocate device memory
	checkCudaErrors(cudaMalloc((void **)&B6_d_idata, B6_mem_size));
	checkCudaErrors(cudaMalloc((void **)&B6_d_odata, B6_mem_size));

	// initialize host data
	for (int i = 0; i < (B6_size_x*B6_size_y); ++i)
	{
		B6_h_idata[i] = (float)i;
	}
	B6_computeTransposeGold(B6_transposeGold, B6_h_idata);
	B6_kernel = &transposeCoalesced;
	B6_kernelName = "coalesced         ";

	B6_gold = B6_transposeGold;

	// Clear error status
	checkCudaErrors(cudaGetLastError());

}
void B6_H2D(cudaStream_t Stream)
{
	cudaError_t err = cudaSuccess;
	printf("Copy input data from the host memory to the CUDA device\n");
	// copy host data to device
	err = cudaMemcpyAsync(B6_d_idata, B6_h_idata, B6_mem_size, cudaMemcpyHostToDevice, Stream);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector h_idata from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void B6_Kernel(cudaStream_t Stream)
{
	B6_kernel << <B6_grid, B6_threads, 0, Stream >> >(B6_d_odata, B6_d_idata, B6_size_x, B6_size_y);
}
void B6_D2H(cudaStream_t Stream)
{
	checkCudaErrors(cudaMemcpyAsync(B6_h_odata, B6_d_odata, B6_mem_size, cudaMemcpyDeviceToHost, Stream));
	printf("Copy output data from the CUDA device to the host memory\n");

}

void B6_CL_Mem(cudaStream_t Stream)
{
	cudaStreamSynchronize(Stream);
	printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
		B6_size_x, B6_size_y, B6_size_x / B6_TILE_DIM, B6_size_y / B6_TILE_DIM, B6_TILE_DIM, B6_TILE_DIM, B6_TILE_DIM, B6_BLOCK_ROWS);
	bool success = true;
	bool res = compareData(B6_gold, B6_h_odata, B6_size_x*B6_size_y, 0.01f, 0.0f);

	if (res == false)
	{
		printf("*** %s kernel FAILED ***\n", B6_kernelName);
		success = false;
	}
	printf("Test Passed!\n");
	cudaFree(B6_d_idata);
	cudaFree(B6_d_odata);
	cudaFreeHost(B6_h_idata);
	cudaFreeHost(B6_h_odata);
	cudaFreeHost(B6_transposeGold);
	if (!success)
	{
		printf("Test failed!\n");
		exit(EXIT_FAILURE);
	}

}
