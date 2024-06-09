#ifdef B14_RD_WG_SIZE_0_0
#define B14_MAXBLOCKSIZE B14_RD_WG_SIZE_0_0
#elif defined(B14_RD_WG_SIZE_0)
#define B14_MAXBLOCKSIZE B14_RD_WG_SIZE_0
#elif defined(B14_RD_WG_SIZE)
#define B14_MAXBLOCKSIZE B14_RD_WG_SIZE
#else
#define B14_MAXBLOCKSIZE 512
#endif

//2D defines. Go from specific to general                                                
#ifdef B14_RD_WG_SIZE_1_0
#define B14_BLOCK_SIZE_XY B14_RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define B14_BLOCK_SIZE_XY B14_RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define B14_BLOCK_SIZE_XY B14_RD_WG_SIZE
#else
#define B14_BLOCK_SIZE_XY 4
#endif

int B14_Size;
float *B14_a, *B14_b, *B14_finalVec;
float *B14_m;

FILE *B14_fp;

unsigned int B14_totalKernelTime = 0;

float *B14_m_cuda, *B14_a_cuda, *B14_b_cuda;
int B14_verbose = 1;

dim3 B14_dimBlock, B14_dimGrid, B14_dimBlockXY, B14_dimGridXY;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size) {
	int i, j;
	float lamda = -0.01;
	std::vector<float> coe(2 * size - 1);
	float coe_i = 0.0;

	for (i = 0; i < size; i++)
	{
		coe_i = 10 * exp(lamda*i);
		j = size - 1 + i;
		coe[j] = coe_i;
		j = size - 1 - i;
		coe[j] = coe_i;
	}


	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			m[i*size + j] = coe[size - 1 - i + j];
		}
	}


}

/*------------------------------------------------------
** PrintDeviceProperties
**-----------------------------------------------------
*/
void PrintDeviceProperties() {
	cudaDeviceProp deviceProp;
	int nDevCount = 0;

	cudaGetDeviceCount(&nDevCount);
	printf("Total Device found: %d", nDevCount);
	for (int nDeviceIdx = 0; nDeviceIdx < nDevCount; ++nDeviceIdx)
	{
		memset(&deviceProp, 0, sizeof(deviceProp));
		if (cudaSuccess == cudaGetDeviceProperties(&deviceProp, nDeviceIdx))
		{
			printf("\nDevice Name \t\t - %s ", deviceProp.name);
			printf("\n**************************************");
			printf("\nTotal Global Memory\t\t\t - %lu KB", deviceProp.totalGlobalMem / 1024);
			printf("\nShared memory available per block \t - %lu KB", deviceProp.sharedMemPerBlock / 1024);
			printf("\nNumber of registers per thread block \t - %d", deviceProp.regsPerBlock);
			printf("\nWarp size in threads \t\t\t - %d", deviceProp.warpSize);
			printf("\nMemory Pitch \t\t\t\t - %zu bytes", deviceProp.memPitch);
			printf("\nMaximum threads per block \t\t - %d", deviceProp.maxThreadsPerBlock);
			printf("\nMaximum Thread Dimension (block) \t - %d %d %d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
			printf("\nMaximum Thread Dimension (grid) \t - %d %d %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
			printf("\nTotal constant memory \t\t\t - %zu bytes", deviceProp.totalConstMem);
			printf("\nCUDA ver \t\t\t\t - %d.%d", deviceProp.major, deviceProp.minor);
			printf("\nClock rate \t\t\t\t - %d KHz", deviceProp.clockRate);
			printf("\nTexture Alignment \t\t\t - %zu bytes", deviceProp.textureAlignment);
			printf("\nDevice Overlap \t\t\t\t - %s", deviceProp.deviceOverlap ? "Allowed" : "Not Allowed");
			printf("\nNumber of Multi processors \t\t - %d\n\n", deviceProp.multiProcessorCount);
		}
		else
			printf("\n%s", cudaGetErrorString(cudaGetLastError()));
	}
}

void InitMat(float *ary, int nrow, int ncol)
{
	int i, j;

	for (i = 0; i<nrow; i++) {
		for (j = 0; j<ncol; j++) {
			fscanf(B14_fp, "%f", ary + B14_Size * i + j);
		}
	}
}

/*------------------------------------------------------
** PrintMat() -- Print the contents of the matrix
**------------------------------------------------------
*/
void PrintMat(float *ary, int nrow, int ncol)
{
	int i, j;

	for (i = 0; i<nrow; i++) {
		for (j = 0; j<ncol; j++) {
			printf("%8.2f ", *(ary + B14_Size * i + j));
		}
		printf("\n");
	}
	printf("\n");
}

/*------------------------------------------------------
** InitAry() -- Initialize the array (vector) by reading
** data from the data file
**------------------------------------------------------
*/
void InitAry(float *ary, int ary_size)
{
	int i;

	for (i = 0; i<ary_size; i++) {
		fscanf(B14_fp, "%f", &ary[i]);
	}
}
/*------------------------------------------------------
** InitProblemOnce -- Initialize all of matrices and
** vectors by opening a data file specified by the user.
**
** We used dynamic array *a, *b, and *m to allocate
** the memory storages.
**------------------------------------------------------
*/
void InitProblemOnce(char *filename)
{
	//char *filename = argv[1];

	//printf("Enter the data file name: ");
	//scanf("%s", filename);
	//printf("The file name is: %s\n", filename);

	B14_fp = fopen(filename, "r");

	fscanf(B14_fp, "%d", &B14_Size);

	B14_a = (float *)malloc(B14_Size * B14_Size * sizeof(float));

	InitMat(B14_a, B14_Size, B14_Size);
	//printf("The input matrix a is:\n");
	//PrintMat(a, Size, Size);
	B14_b = (float *)malloc(B14_Size * sizeof(float));

	InitAry(B14_b, B14_Size);
	//printf("The input array b is:\n");
	//PrintAry(b, Size);

	B14_m = (float *)malloc(B14_Size * B14_Size * sizeof(float));
}

/*------------------------------------------------------
** InitPerRun() -- Initialize the contents of the
** multipier matrix **m
**------------------------------------------------------
*/
void InitPerRun()
{
	int i;
	for (i = 0; i<B14_Size*B14_Size; i++)
		*(B14_m + i) = 0.0;
}

/*-------------------------------------------------------
** Fan1() -- Calculate multiplier matrix
** Pay attention to the index.  Index i give the range
** which starts from 0 to range-1.  The real values of
** the index should be adjust and related with the value
** of t which is defined on the ForwardSub().
**-------------------------------------------------------
*/
__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t)
{
	//if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
	//printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);

	if (threadIdx.x + blockIdx.x * blockDim.x >= Size - 1 - t) return;
	*(m_cuda + Size * (blockDim.x*blockIdx.x + threadIdx.x + t + 1) + t) = *(a_cuda + Size * (blockDim.x*blockIdx.x + threadIdx.x + t + 1) + t) / *(a_cuda + Size * t + t);
}

/*-------------------------------------------------------
** Fan2() -- Modify the matrix A into LUD
**-------------------------------------------------------
*/

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda, int Size, int j1, int t)
{
	if (threadIdx.x + blockIdx.x * blockDim.x >= Size - 1 - t) return;
	if (threadIdx.y + blockIdx.y * blockDim.y >= Size - t) return;

	int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);

	a_cuda[Size*(xidx + 1 + t) + (yidx + t)] -= m_cuda[Size*(xidx + 1 + t) + t] * a_cuda[Size*t + (yidx + t)];
	//a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
	if (yidx == 0) {
		//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
		//printf("xidx:%d,yidx:%d\n",xidx,yidx);
		b_cuda[xidx + 1 + t] -= m_cuda[Size*(xidx + 1 + t) + (yidx + t)] * b_cuda[t];
	}
}

/*------------------------------------------------------
** BackSub() -- Backward substitution
**------------------------------------------------------
*/

void BackSub()
{
	// create a new vector to hold the final answer
	B14_finalVec = (float *)malloc(B14_Size * sizeof(float));
	// solve "bottom up"
	int i, j;
	for (i = 0; i<B14_Size; i++) {
		B14_finalVec[B14_Size - i - 1] = B14_b[B14_Size - i - 1];
		for (j = 0; j<i; j++)
		{
			B14_finalVec[B14_Size - i - 1] -= *(B14_a + B14_Size * (B14_Size - i - 1) + (B14_Size - j - 1)) * B14_finalVec[B14_Size - j - 1];
		}
		B14_finalVec[B14_Size - i - 1] = B14_finalVec[B14_Size - i - 1] / *(B14_a + B14_Size * (B14_Size - i - 1) + (B14_Size - i - 1));
	}
}

/*------------------------------------------------------
** PrintAry() -- Print the contents of the array (vector)
**------------------------------------------------------
*/
void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i = 0; i<ary_size; i++) {
		printf("%.2f ", ary[i]);
	}
	printf("\n\n");
}
//void checkCUDAError(const char *msg)
//{
//	cudaError_t err = cudaGetLastError();
//	if (cudaSuccess != err)
//	{
//		fprintf(stderr, "Cuda error: %s: %s.\n", msg,
//			cudaGetErrorString(err));
//		exit(EXIT_FAILURE);
//	}
//}


void B14_Malloc(int argc, char** argv)
{
	printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", B14_MAXBLOCKSIZE, B14_BLOCK_SIZE_XY, B14_BLOCK_SIZE_XY);

	int j;
	if (argc < 2) {
		printf("Usage: gaussian -f filename / -s size [-q]\n\n");
		printf("-q (quiet) suppresses printing the matrix and result values.\n");
		printf("-f (filename) path of input file\n");
		printf("-s (size) size of matrix. Create matrix and rhs in this program \n");
		printf("The first line of the file contains the dimension of the matrix, n.");
		printf("The second line of the file is a newline.\n");
		printf("The next n lines contain n tab separated values for the matrix.");
		printf("The next line of the file is a newline.\n");
		printf("The next line of the file is a 1xn vector with tab separated values.\n");
		printf("The next line of the file is a newline. (optional)\n");
		printf("The final line of the file is the pre-computed solution. (optional)\n");
		printf("Example: matrix4.txt:\n");
		printf("4\n");
		printf("\n");
		printf("-0.6	-0.5	0.7	0.3\n");
		printf("-0.3	-0.9	0.3	0.7\n");
		printf("-0.4	-0.5	-0.3	-0.8\n");
		printf("0.0	-0.1	0.2	0.9\n");
		printf("\n");
		printf("-0.85	-0.68	0.24	-0.53\n");
		printf("\n");
		printf("0.7	0.0	-0.4	-0.5\n");
		exit(0);
	}

	//PrintDeviceProperties();
	//char filename[100];
	//sprintf(filename,"matrices/matrix%d.txt",size);

	if (checkCmdLineFlag(argc, (const char **)argv, "GSize"))
	{
		B14_Size = getCmdLineArgumentInt(argc, (const char **)argv, "GSize");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "GQuiet"))
	{
		B14_verbose = getCmdLineArgumentInt(argc, (const char **)argv, "GQuiet");
	}
	printf("Create matrix internally in parse, size = %d \n", B14_Size);

	cudaMallocHost((void **)&B14_a, B14_Size * B14_Size * sizeof(float));
	create_matrix(B14_a, B14_Size);

	cudaMallocHost((void **)&B14_b, B14_Size * sizeof(float));
	for (j = 0; j< B14_Size; j++)
		B14_b[j] = 1.0;

	cudaMallocHost((void **)&B14_m, B14_Size * B14_Size * sizeof(float));

	InitProblemOnce("C:\\Users\\Admin\\Documents\\Visual Studio 2015\\Projects\\Gaussian_nStreams\\Gaussian_nStreams\\gaussian\\matrix1024.txt");
	//InitProblemOnce(filename);
	InitPerRun();

	// allocate memory on GPU
	cudaMalloc((void **)&B14_m_cuda, B14_Size * B14_Size * sizeof(float));

	cudaMalloc((void **)&B14_a_cuda, B14_Size * B14_Size * sizeof(float));

	cudaMalloc((void **)&B14_b_cuda, B14_Size * sizeof(float));

	int block_size, grid_size;

	block_size = B14_MAXBLOCKSIZE;
	grid_size = (B14_Size / block_size) + (!(B14_Size%block_size) ? 0 : 1);
	//printf("1d grid size: %d\n",grid_size);


	dim3 dimBlock(block_size);
	dim3 dimGrid(grid_size);
	B14_dimBlock = dimBlock;
	B14_dimGrid = dimGrid;

	//dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	int blockSize2d, gridSize2d;
	blockSize2d = B14_BLOCK_SIZE_XY;
	gridSize2d = (B14_Size / blockSize2d) + (!(B14_Size%blockSize2d ? 0 : 1));

	dim3 dimBlockXY(blockSize2d, blockSize2d);
	dim3 dimGridXY(gridSize2d, gridSize2d);
	B14_dimBlockXY = dimBlockXY;
	B14_dimGridXY = dimGridXY;
}

void B14_H2D(cudaStream_t Stream)
{
	// copy memory to GPU
	cudaMemcpyAsync(B14_m_cuda, B14_m, B14_Size * B14_Size * sizeof(float), cudaMemcpyHostToDevice, Stream);
	cudaMemcpyAsync(B14_a_cuda, B14_a, B14_Size * B14_Size * sizeof(float), cudaMemcpyHostToDevice, Stream);
	cudaMemcpyAsync(B14_b_cuda, B14_b, B14_Size * sizeof(float), cudaMemcpyHostToDevice, Stream);
}

void B14_Kernel(cudaStream_t Stream)
{
	for (int t = 0; t<(B14_Size - 1); t++) {
		Fan1 << <B14_dimGrid, B14_dimBlock, 0, Stream >> >(B14_m_cuda, B14_a_cuda, B14_Size, t);
		Fan2 << <B14_dimGridXY, B14_dimBlockXY, 0, Stream >> >(B14_m_cuda, B14_a_cuda, B14_b_cuda, B14_Size, B14_Size - t, t);
		//checkCUDAError("Fan2");
	}
}

void B14_D2H(cudaStream_t Stream)
{
	// copy memory back to CPU
	cudaMemcpyAsync(B14_m, B14_m_cuda, B14_Size * B14_Size * sizeof(float), cudaMemcpyDeviceToHost, Stream);
	cudaMemcpyAsync(B14_a, B14_a_cuda, B14_Size * B14_Size * sizeof(float), cudaMemcpyDeviceToHost, Stream);
	cudaMemcpyAsync(B14_b, B14_b_cuda, B14_Size * sizeof(float), cudaMemcpyDeviceToHost, Stream);
}

void B14_CL_Mem(cudaStream_t Stream)
{
	cudaStreamSynchronize(Stream);
	if (B14_verbose) {
		printf("Matrix m is: \n");
		PrintMat(B14_m, B14_Size, B14_Size);

		printf("Matrix a is: \n");
		PrintMat(B14_a, B14_Size, B14_Size);

		printf("Array b is: \n");
		PrintAry(B14_b, B14_Size);
	}
	BackSub();
	if (B14_verbose) {
		printf("The final solution is: \n");
		PrintAry(B14_finalVec, B14_Size);
	}
	cudaFree(B14_m_cuda);
	cudaFree(B14_a_cuda);
	cudaFree(B14_b_cuda);
	free(B14_m);
	free(B14_a);
	free(B14_b);
}
