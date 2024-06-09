const char *B7_sSDKsample = "Transpose";
#define B7_TILE_DIM    16
#define B7_BLOCK_ROWS  16
int B7_MUL_FACTOR = B7_TILE_DIM;

#define B7_FLOOR(a,b) (a-(a%b))

#define B7_NUM_REPS  10

dim3 B7_threads;
dim3 B7_grid;

int B7_size_x, B7_size_y;
float *B7_h_idata;
float *B7_h_odata;
float *B7_d_idata;
float *B7_d_odata;
const char *B7_kernelName;
void(*B7_kernel)(float *, float *, int, int);
size_t B7_mem_size;
float *B7_transposeGold;
float *B7_gold;

__global__ void copy(float *odata, float *idata, int width, int height)
{
	int xIndex = blockIdx.x * B7_TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * B7_TILE_DIM + threadIdx.y;

	int index = xIndex + width * yIndex;

	for (int i = 0; i<B7_TILE_DIM; i += B7_BLOCK_ROWS)
	{
		odata[index + i * width] = idata[index + i * width];
	}

}

void B7_computeTransposeGold(float *gold, float *idata)
{
	for (int y = 0; y < B7_size_y; ++y)
	{
		for (int x = 0; x < B7_size_x; ++x)
		{
			gold[(x * B7_size_y) + y] = idata[(y * B7_size_x) + x];
		}
	}
}

void B7_getParams(int argc, char **argv, cudaDeviceProp &deviceProp, int max_tile_dim)
{
	// set matrix size (if (x,y) dim of matrix is not square, then this will have to be modified
	if (checkCmdLineFlag(argc, (const char **)argv, "CdimX"))
	{
		B7_size_x = getCmdLineArgumentInt(argc, (const char **)argv, "CdimX");

		if (B7_size_x > max_tile_dim)
		{
			printf("> MatrixSize X = %d is greater than the recommended size = %d\n", B7_size_x, max_tile_dim);
		}
		else
		{
			printf("> MatrixSize X = %d\n", B7_size_x);
		}
	}
	else
	{
		B7_size_x = max_tile_dim;
		B7_size_x = B7_FLOOR(B7_size_x, 512);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "CdimY"))
	{
		B7_size_y = getCmdLineArgumentInt(argc, (const char **)argv, "CdimY");

		if (B7_size_y > max_tile_dim)
		{
			printf("> MatrixSize Y = %d is greater than the recommended size = %d\n", B7_size_y, max_tile_dim);
		}
		else
		{
			printf("> MatrixSize Y = %d\n", B7_size_y);
		}
	}
	else
	{
		B7_size_y = max_tile_dim;

		// If this is SM12 hardware, we want to round down to a multiple of 512
		if ((deviceProp.major == 1 && deviceProp.minor >= 2) || deviceProp.major > 1)
		{
			B7_size_y = B7_FLOOR(B7_size_y, 512);
		}
		else     // else for SM10,SM11 we round down to a multiple of 384
		{
			B7_size_y = B7_FLOOR(B7_size_y, 384);
		}
	}
}


void
B7_showHelp()
{
	printf("\n%s : Command line options\n", B7_sSDKsample);
	printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
	printf("> The default matrix size can be overridden with these parameters\n");
	printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
	printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}

void B7_Malloc(int argc, char** argv)
{
	int MATRIX_SIZE_X = 1024;
	int MATRIX_SIZE_Y = 1024;
	if (checkCmdLineFlag(argc, (const char **)argv, "CdimX"))
	{
		MATRIX_SIZE_X = getCmdLineArgumentInt(argc, (const char **)argv, "CdimX");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "CdimY"))
	{
		MATRIX_SIZE_Y = getCmdLineArgumentInt(argc, (const char **)argv, "CdimY");
	}
	printf("Enter Matrix_Size_X & Matrix_Size_Y (TransposeCopy Benchmark):\n");
	int MAX_TILES = (B7_FLOOR(MATRIX_SIZE_X, 512) * B7_FLOOR(MATRIX_SIZE_Y, 512)) / (B7_TILE_DIM *B7_TILE_DIM);
	// Start logs

	printf("%s Starting...\n\n", B7_sSDKsample);

	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	{
		B7_showHelp();
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

	max_matrix_dim = B7_FLOOR((int)(floor(sqrt(total_tiles))* B7_TILE_DIM), matrix_size_test);

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
	B7_getParams(argc, argv, deviceProp, max_matrix_dim);

	if (B7_size_x != B7_size_y)
	{
		printf("\n[%s] does not support non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n", B7_sSDKsample, B7_size_x, B7_size_y);
		exit(EXIT_FAILURE);
	}

	if (B7_size_x%B7_TILE_DIM != 0 || B7_size_y % B7_TILE_DIM != 0)
	{
		printf("[%s] Matrix size must be integral multiple of tile size\nExiting...\n\n", B7_sSDKsample);
		exit(EXIT_FAILURE);
	}

	// execution configuration parameters
	dim3 grid(B7_size_x / B7_TILE_DIM, B7_size_y / B7_TILE_DIM), threads(B7_TILE_DIM, B7_BLOCK_ROWS);
	B7_grid = grid;
	B7_threads = threads;

	if (grid.x < 1 || grid.y < 1)
	{
		printf("[%s] grid size computation incorrect in test \nExiting...\n\n", B7_sSDKsample);
		exit(EXIT_FAILURE);
	}


	// size of memory required to store the matrix
	B7_mem_size = static_cast<size_t>(sizeof(float) * B7_size_x*B7_size_y);

	if (2 * B7_mem_size > deviceProp.totalGlobalMem)
	{
		printf("Input matrix size is larger than the available device memory!\n");
		printf("Please choose a smaller size matrix\n");
		exit(EXIT_FAILURE);
	}

	// allocate host memory

	cudaMallocHost((void **)&B7_h_idata, B7_mem_size);
	cudaMallocHost((void **)&B7_h_odata, B7_mem_size);
	cudaMallocHost((void **)&B7_transposeGold, B7_mem_size);
	if (B7_h_idata == NULL || B7_h_odata == NULL || B7_transposeGold == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// allocate device memory
	checkCudaErrors(cudaMalloc((void **)&B7_d_idata, B7_mem_size));
	checkCudaErrors(cudaMalloc((void **)&B7_d_odata, B7_mem_size));

	// initialize host data
	for (int i = 0; i < (B7_size_x*B7_size_y); ++i)
	{
		B7_h_idata[i] = (float)i;
	}
	B7_computeTransposeGold(B7_transposeGold, B7_h_idata);
	B7_kernel = &copy;
	B7_kernelName = "simple copy       ";

	B7_gold = B7_h_idata;
	// Clear error status
	checkCudaErrors(cudaGetLastError());

}
void B7_H2D(cudaStream_t Stream)
{
	cudaError_t err = cudaSuccess;
	printf("Copy input data from the host memory to the CUDA device\n");
	// copy host data to device
	err = cudaMemcpyAsync(B7_d_idata, B7_h_idata, B7_mem_size, cudaMemcpyHostToDevice, Stream);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector h_idata from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void B7_Kernel(cudaStream_t Stream)
{
	B7_kernel << <B7_grid, B7_threads, 0, Stream >> >(B7_d_odata, B7_d_idata, B7_size_x, B7_size_y);
}
void B7_D2H(cudaStream_t Stream)
{
	checkCudaErrors(cudaMemcpyAsync(B7_h_odata, B7_d_odata, B7_mem_size, cudaMemcpyDeviceToHost, Stream));
	printf("Copy output data from the CUDA device to the host memory\n");

}

void B7_CL_Mem(cudaStream_t Stream)
{
	cudaStreamSynchronize(Stream);
	printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
		B7_size_x, B7_size_y, B7_size_x / B7_TILE_DIM, B7_size_y / B7_TILE_DIM, B7_TILE_DIM, B7_TILE_DIM, B7_TILE_DIM, B7_BLOCK_ROWS);
	bool success = true;
	bool res = compareData(B7_gold, B7_h_odata, B7_size_x*B7_size_y, 0.01f, 0.0f);

	if (res == false)
	{
		printf("*** %s kernel FAILED ***\n", B7_kernelName);
		success = false;
	}
	printf("Test Passed!\n");
	cudaFree(B7_d_idata);
	cudaFree(B7_d_odata);
	cudaFreeHost(B7_h_idata);
	cudaFreeHost(B7_h_odata);
	cudaFreeHost(B7_transposeGold);
	if (!success)
	{
		printf("Test failed!\n");
		exit(EXIT_FAILURE);
	}

}

