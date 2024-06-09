const char *B12_sSDKsample = "Transpose";
#define B12_TILE_DIM    16
#define B12_BLOCK_ROWS  16
int B12_MUL_FACTOR = B12_TILE_DIM;

#define B12_FLOOR(a,b) (a-(a%b))

#define B12_NUM_REPS  10

dim3 B12_threads;
dim3 B12_grid;

int B12_size_x, B12_size_y;
float *B12_h_idata;
float *B12_h_odata;
float *B12_d_idata;
float *B12_d_odata;
const char *B12_kernelName;
void(*B12_kernel)(float *, float *, int, int);
size_t B12_mem_size;
float *B12_transposeGold;
float *B12_gold;

__global__ void transposeFineGrained(float *odata, float *idata, int width, int height)
{
	__shared__ float block[B12_TILE_DIM][B12_TILE_DIM + 1];

	int xIndex = blockIdx.x * B12_TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * B12_TILE_DIM + threadIdx.y;
	int index = xIndex + (yIndex)*width;

	for (int i = 0; i < B12_TILE_DIM; i += B12_BLOCK_ROWS)
	{
		block[threadIdx.y + i][threadIdx.x] = idata[index + i * width];
	}

	__syncthreads();

	for (int i = 0; i < B12_TILE_DIM; i += B12_BLOCK_ROWS)
	{
		odata[index + i * height] = block[threadIdx.x][threadIdx.y + i];
	}
}

void B12_computeTransposeGold(float *gold, float *idata)
{
	for (int y = 0; y < B12_size_y; ++y)
	{
		for (int x = 0; x < B12_size_x; ++x)
		{
			gold[(x * B12_size_y) + y] = idata[(y * B12_size_x) + x];
		}
	}
}

void B12_getParams(int argc, char **argv, cudaDeviceProp &deviceProp, int max_tile_dim)
{
	// set matrix size (if (x,y) dim of matrix is not square, then this will have to be modified
	if (checkCmdLineFlag(argc, (const char **)argv, "TFGdimX"))
	{
		B12_size_x = getCmdLineArgumentInt(argc, (const char **)argv, "TFGdimX");

		if (B12_size_x > max_tile_dim)
		{
			printf("> MatrixSize X = %d is greater than the recommended size = %d\n", B12_size_x, max_tile_dim);
		}
		else
		{
			printf("> MatrixSize X = %d\n", B12_size_x);
		}
	}
	else
	{
		B12_size_x = max_tile_dim;
		B12_size_x = B12_FLOOR(B12_size_x, 512);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "TFGdimY"))
	{
		B12_size_y = getCmdLineArgumentInt(argc, (const char **)argv, "TFGdimY");

		if (B12_size_y > max_tile_dim)
		{
			printf("> MatrixSize Y = %d is greater than the recommended size = %d\n", B12_size_y, max_tile_dim);
		}
		else
		{
			printf("> MatrixSize Y = %d\n", B12_size_y);
		}
	}
	else
	{
		B12_size_y = max_tile_dim;

		// If this is SM12 hardware, we want to round down to a multiple of 512
		if ((deviceProp.major == 1 && deviceProp.minor >= 2) || deviceProp.major > 1)
		{
			B12_size_y = B12_FLOOR(B12_size_y, 512);
		}
		else     // else for SM10,SM11 we round down to a multiple of 384
		{
			B12_size_y = B12_FLOOR(B12_size_y, 384);
		}
	}
}


void
B12_showHelp()
{
	printf("\n%s : Command line options\n", B12_sSDKsample);
	printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
	printf("> The default matrix size can be overridden with these parameters\n");
	printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
	printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}

void B12_Malloc(int argc, char** argv)
{
	int MATRIX_SIZE_X = 1024;
	int MATRIX_SIZE_Y = 1024;
	if (checkCmdLineFlag(argc, (const char **)argv, "TFGdimX"))
	{
		MATRIX_SIZE_X = getCmdLineArgumentInt(argc, (const char **)argv, "TFGdimX");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "TFGdimY"))
	{
		MATRIX_SIZE_Y = getCmdLineArgumentInt(argc, (const char **)argv, "TFGdimY");
	}
	printf("Enter Matrix_Size_X & Matrix_Size_Y (TransposeFineGrained Benchmark):\n");
	int MAX_TILES = (B12_FLOOR(MATRIX_SIZE_X, 512) * B12_FLOOR(MATRIX_SIZE_Y, 512)) / (B12_TILE_DIM *B12_TILE_DIM);
	// Start logs

	printf("%s Starting...\n\n", B12_sSDKsample);

	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	{
		B12_showHelp();
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

	max_matrix_dim = B12_FLOOR((int)(floor(sqrt(total_tiles))* B12_TILE_DIM), matrix_size_test);

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
	B12_getParams(argc, argv, deviceProp, max_matrix_dim);

	if (B12_size_x != B12_size_y)
	{
		printf("\n[%s] does not support non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n", B12_sSDKsample, B12_size_x, B12_size_y);
		exit(EXIT_FAILURE);
	}

	if (B12_size_x%B12_TILE_DIM != 0 || B12_size_y % B12_TILE_DIM != 0)
	{
		printf("[%s] Matrix size must be integral multiple of tile size\nExiting...\n\n", B12_sSDKsample);
		exit(EXIT_FAILURE);
	}

	// execution configuration parameters
	dim3 grid(B12_size_x / B12_TILE_DIM, B12_size_y / B12_TILE_DIM), threads(B12_TILE_DIM, B12_BLOCK_ROWS);
	B12_grid = grid;
	B12_threads = threads;

	if (grid.x < 1 || grid.y < 1)
	{
		printf("[%s] grid size computation incorrect in test \nExiting...\n\n", B12_sSDKsample);
		exit(EXIT_FAILURE);
	}


	// size of memory required to store the matrix
	B12_mem_size = static_cast<size_t>(sizeof(float) * B12_size_x*B12_size_y);

	if (2 * B12_mem_size > deviceProp.totalGlobalMem)
	{
		printf("Input matrix size is larger than the available device memory!\n");
		printf("Please choose a smaller size matrix\n");
		exit(EXIT_FAILURE);
	}

	// allocate host memory

	cudaMallocHost((void **)&B12_h_idata, B12_mem_size);
	cudaMallocHost((void **)&B12_h_odata, B12_mem_size);
	cudaMallocHost((void **)&B12_transposeGold, B12_mem_size);
	if (B12_h_idata == NULL || B12_h_odata == NULL || B12_transposeGold == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// allocate device memory
	checkCudaErrors(cudaMalloc((void **)&B12_d_idata, B12_mem_size));
	checkCudaErrors(cudaMalloc((void **)&B12_d_odata, B12_mem_size));

	// initialize host data
	for (int i = 0; i < (B12_size_x*B12_size_y); ++i)
	{
		B12_h_idata[i] = (float)i;
	}
	B12_computeTransposeGold(B12_transposeGold, B12_h_idata);
	B12_kernel = &transposeFineGrained;
	B12_kernelName = "fine-grained      ";

	B12_gold = B12_h_odata;

	// Clear error status
	checkCudaErrors(cudaGetLastError());

}
void B12_H2D(cudaStream_t Stream)
{
	cudaError_t err = cudaSuccess;
	printf("Copy input data from the host memory to the CUDA device\n");
	// copy host data to device
	err = cudaMemcpyAsync(B12_d_idata, B12_h_idata, B12_mem_size, cudaMemcpyHostToDevice, Stream);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector h_idata from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void B12_Kernel(cudaStream_t Stream)
{
	B12_kernel << <B12_grid, B12_threads, 0, Stream >> >(B12_d_odata, B12_d_idata, B12_size_x, B12_size_y);
}
void B12_D2H(cudaStream_t Stream)
{
	checkCudaErrors(cudaMemcpyAsync(B12_h_odata, B12_d_odata, B12_mem_size, cudaMemcpyDeviceToHost, Stream));
	printf("Copy output data from the CUDA device to the host memory\n");

}

void B12_CL_Mem(cudaStream_t Stream)
{
	cudaStreamSynchronize(Stream);
	printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
		B12_size_x, B12_size_y, B12_size_x / B12_TILE_DIM, B12_size_y / B12_TILE_DIM, B12_TILE_DIM, B12_TILE_DIM, B12_TILE_DIM, B12_BLOCK_ROWS);
	bool success = true;
	bool res = compareData(B12_gold, B12_h_odata, B12_size_x*B12_size_y, 0.01f, 0.0f);

	if (res == false)
	{
		printf("*** %s kernel FAILED ***\n", B12_kernelName);
		success = false;
	}
	printf("Test Passed!\n");
	cudaFree(B12_d_idata);
	cudaFree(B12_d_odata);
	cudaFreeHost(B12_h_idata);
	cudaFreeHost(B12_h_odata);
	cudaFreeHost(B12_transposeGold);
	if (!success)
	{
		printf("Test failed!\n");
		exit(EXIT_FAILURE);
	}

}
