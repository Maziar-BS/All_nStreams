const char *B11_sSDKsample = "Transpose";
#define B11_TILE_DIM    16
#define B11_BLOCK_ROWS  16
int B11_MUL_FACTOR = B11_TILE_DIM;

#define B11_FLOOR(a,b) (a-(a%b))

#define B11_NUM_REPS  10

dim3 B11_threads;
dim3 B11_grid;

int B11_size_x, B11_size_y;
float *B11_h_idata;
float *B11_h_odata;
float *B11_d_idata;
float *B11_d_odata;
const char *B11_kernelName;
void(*B11_kernel)(float *, float *, int, int);
size_t B11_mem_size;
float *B11_transposeGold;
float *B11_gold;

__global__ void transposeDiagonal(float *odata, float *idata, int width, int height)
{
	__shared__ float tile[B11_TILE_DIM][B11_TILE_DIM + 1];

	int blockIdx_x, blockIdx_y;

	// do diagonal reordering
	if (width == height)
	{
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	}
	else
	{
		int bid = blockIdx.x + gridDim.x*blockIdx.y;
		blockIdx_y = bid % gridDim.y;
		blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
	}

	// from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
	// and similarly for y

	int xIndex = blockIdx_x * B11_TILE_DIM + threadIdx.x;
	int yIndex = blockIdx_y * B11_TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;

	xIndex = blockIdx_y * B11_TILE_DIM + threadIdx.x;
	yIndex = blockIdx_x * B11_TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;

	for (int i = 0; i<B11_TILE_DIM; i += B11_BLOCK_ROWS)
	{
		tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
	}

	__syncthreads();

	for (int i = 0; i<B11_TILE_DIM; i += B11_BLOCK_ROWS)
	{
		odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
	}
}

void B11_computeTransposeGold(float *gold, float *idata)
{
	for (int y = 0; y < B11_size_y; ++y)
	{
		for (int x = 0; x < B11_size_x; ++x)
		{
			gold[(x * B11_size_y) + y] = idata[(y * B11_size_x) + x];
		}
	}
}

void B11_getParams(int argc, char **argv, cudaDeviceProp &deviceProp, int max_tile_dim)
{
	// set matrix size (if (x,y) dim of matrix is not square, then this will have to be modified
	if (checkCmdLineFlag(argc, (const char **)argv, "TDdimX"))
	{
		B11_size_x = getCmdLineArgumentInt(argc, (const char **)argv, "TDdimX");

		if (B11_size_x > max_tile_dim)
		{
			printf("> MatrixSize X = %d is greater than the recommended size = %d\n", B11_size_x, max_tile_dim);
		}
		else
		{
			printf("> MatrixSize X = %d\n", B11_size_x);
		}
	}
	else
	{
		B11_size_x = max_tile_dim;
		B11_size_x = B11_FLOOR(B11_size_x, 512);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "TDdimY"))
	{
		B11_size_y = getCmdLineArgumentInt(argc, (const char **)argv, "TDdimY");

		if (B11_size_y > max_tile_dim)
		{
			printf("> MatrixSize Y = %d is greater than the recommended size = %d\n", B11_size_y, max_tile_dim);
		}
		else
		{
			printf("> MatrixSize Y = %d\n", B11_size_y);
		}
	}
	else
	{
		B11_size_y = max_tile_dim;

		// If this is SM12 hardware, we want to round down to a multiple of 512
		if ((deviceProp.major == 1 && deviceProp.minor >= 2) || deviceProp.major > 1)
		{
			B11_size_y = B11_FLOOR(B11_size_y, 512);
		}
		else     // else for SM10,SM11 we round down to a multiple of 384
		{
			B11_size_y = B11_FLOOR(B11_size_y, 384);
		}
	}
}


void
B11_showHelp()
{
	printf("\n%s : Command line options\n", B11_sSDKsample);
	printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
	printf("> The default matrix size can be overridden with these parameters\n");
	printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
	printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}

void B11_Malloc(int argc, char** argv)
{
	int MATRIX_SIZE_X = 1024;
	int MATRIX_SIZE_Y = 1024;
	if (checkCmdLineFlag(argc, (const char **)argv, "TDdimX"))
	{
		MATRIX_SIZE_X = getCmdLineArgumentInt(argc, (const char **)argv, "TDdimX");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "TDdimY"))
	{
		MATRIX_SIZE_Y = getCmdLineArgumentInt(argc, (const char **)argv, "TDdimY");
	}
	printf("Enter Matrix_Size_X & Matrix_Size_Y (TransposeDiagonal Benchmark):\n");
	int MAX_TILES = (B11_FLOOR(MATRIX_SIZE_X, 512) * B11_FLOOR(MATRIX_SIZE_Y, 512)) / (B11_TILE_DIM *B11_TILE_DIM);
	// Start logs

	printf("%s Starting...\n\n", B11_sSDKsample);

	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	{
		B11_showHelp();
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

	max_matrix_dim = B11_FLOOR((int)(floor(sqrt(total_tiles))* B11_TILE_DIM), matrix_size_test);

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
	B11_getParams(argc, argv, deviceProp, max_matrix_dim);

	if (B11_size_x != B11_size_y)
	{
		printf("\n[%s] does not support non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n", B11_sSDKsample, B11_size_x, B11_size_y);
		exit(EXIT_FAILURE);
	}

	if (B11_size_x%B11_TILE_DIM != 0 || B11_size_y % B11_TILE_DIM != 0)
	{
		printf("[%s] Matrix size must be integral multiple of tile size\nExiting...\n\n", B11_sSDKsample);
		exit(EXIT_FAILURE);
	}

	// execution configuration parameters
	dim3 grid(B11_size_x / B11_TILE_DIM, B11_size_y / B11_TILE_DIM), threads(B11_TILE_DIM, B11_BLOCK_ROWS);
	B11_grid = grid;
	B11_threads = threads;

	if (grid.x < 1 || grid.y < 1)
	{
		printf("[%s] grid size computation incorrect in test \nExiting...\n\n", B11_sSDKsample);
		exit(EXIT_FAILURE);
	}


	// size of memory required to store the matrix
	B11_mem_size = static_cast<size_t>(sizeof(float) * B11_size_x*B11_size_y);

	if (2 * B11_mem_size > deviceProp.totalGlobalMem)
	{
		printf("Input matrix size is larger than the available device memory!\n");
		printf("Please choose a smaller size matrix\n");
		exit(EXIT_FAILURE);
	}

	// allocate host memory

	cudaMallocHost((void **)&B11_h_idata, B11_mem_size);
	cudaMallocHost((void **)&B11_h_odata, B11_mem_size);
	cudaMallocHost((void **)&B11_transposeGold, B11_mem_size);
	if (B11_h_idata == NULL || B11_h_odata == NULL || B11_transposeGold == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// allocate device memory
	checkCudaErrors(cudaMalloc((void **)&B11_d_idata, B11_mem_size));
	checkCudaErrors(cudaMalloc((void **)&B11_d_odata, B11_mem_size));

	// initialize host data
	for (int i = 0; i < (B11_size_x*B11_size_y); ++i)
	{
		B11_h_idata[i] = (float)i;
	}
	B11_computeTransposeGold(B11_transposeGold, B11_h_idata);
	B11_kernel = &transposeDiagonal;
	B11_kernelName = "diagonal          ";

	B11_gold = B11_transposeGold;

	// Clear error status
	checkCudaErrors(cudaGetLastError());

}
void B11_H2D(cudaStream_t Stream)
{
	cudaError_t err = cudaSuccess;
	printf("Copy input data from the host memory to the CUDA device\n");
	// copy host data to device
	err = cudaMemcpyAsync(B11_d_idata, B11_h_idata, B11_mem_size, cudaMemcpyHostToDevice, Stream);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector h_idata from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void B11_Kernel(cudaStream_t Stream)
{
	B11_kernel << <B11_grid, B11_threads, 0, Stream >> >(B11_d_odata, B11_d_idata, B11_size_x, B11_size_y);
}
void B11_D2H(cudaStream_t Stream)
{
	checkCudaErrors(cudaMemcpyAsync(B11_h_odata, B11_d_odata, B11_mem_size, cudaMemcpyDeviceToHost, Stream));
	printf("Copy output data from the CUDA device to the host memory\n");

}

void B11_CL_Mem(cudaStream_t Stream)
{
	cudaStreamSynchronize(Stream);
	printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
		B11_size_x, B11_size_y, B11_size_x / B11_TILE_DIM, B11_size_y / B11_TILE_DIM, B11_TILE_DIM, B11_TILE_DIM, B11_TILE_DIM, B11_BLOCK_ROWS);
	bool success = true;
	bool res = compareData(B11_gold, B11_h_odata, B11_size_x*B11_size_y, 0.01f, 0.0f);

	if (res == false)
	{
		printf("*** %s kernel FAILED ***\n", B11_kernelName);
		success = false;
	}
	printf("Test Passed!\n");
	cudaFree(B11_d_idata);
	cudaFree(B11_d_odata);
	cudaFreeHost(B11_h_idata);
	cudaFreeHost(B11_h_odata);
	cudaFreeHost(B11_transposeGold);
	if (!success)
	{
		printf("Test failed!\n");
		exit(EXIT_FAILURE);
	}

}
