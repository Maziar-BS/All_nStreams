
int imageW = 3072;
int imageH = 3072;
int iterations = 16;

float
*h_Kernel,
*h_Input,
*h_Buffer,
*h_OutputCPU,
*h_OutputGPU;

float
*d_Input,
*d_Output,
*d_Buffer;

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[KERNEL_LENGTH];

extern "C" void setConvolutionKernel(cudaStream_t Stream_Num)
{
	cudaMemcpyToSymbolAsync(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float), 0, cudaMemcpyHostToDevice, Stream_Num);
}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

__global__ void convolutionRowsKernel(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	int pitch
	)
{
	__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

	//Offset to the left halo edge
	const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
	const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

	d_Src += baseY * pitch + baseX;
	d_Dst += baseY * pitch + baseX;

	//Load main data
#pragma unroll

	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
	}

	//Load left halo
#pragma unroll

	for (int i = 0; i < ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Load right halo
#pragma unroll

	for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
	}

	//Compute and store results
	__syncthreads();
#pragma unroll

	for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
	{
		float sum = 0;

#pragma unroll

		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
		{
			sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
		}

		d_Dst[i * ROWS_BLOCKDIM_X] = sum;
	}
}

extern "C" void convolutionRowsGPU(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	cudaStream_t Stream_Num
	)
{
	assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
	assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
	assert(imageH % ROWS_BLOCKDIM_Y == 0);

	dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
	dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

	convolutionRowsKernel << <blocks, threads, 0, Stream_Num >> >(
		d_Dst,
		d_Src,
		imageW,
		imageH,
		imageW
		);
	getLastCudaError("convolutionRowsKernel() execution failed\n");
}

#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

__global__ void convolutionColumnsKernel(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	int pitch
	)
{
	__shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

	//Offset to the upper halo edge
	const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
	const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
	d_Src += baseY * pitch + baseX;
	d_Dst += baseY * pitch + baseX;

	//Main data
#pragma unroll

	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
	}

	//Upper halo
#pragma unroll

	for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Lower halo
#pragma unroll

	for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
	{
		s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
	}

	//Compute and store results
	__syncthreads();
#pragma unroll

	for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
	{
		float sum = 0;
#pragma unroll

		for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
		{
			sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
		}

		d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
	}
}

extern "C" void convolutionColumnsGPU(
	float *d_Dst,
	float *d_Src,
	int imageW,
	int imageH,
	cudaStream_t Stream_Num
	)
{
	assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
	assert(imageW % COLUMNS_BLOCKDIM_X == 0);
	assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

	dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
	dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

	convolutionColumnsKernel << <blocks, threads, 0, Stream_Num >> >(
		d_Dst,
		d_Src,
		imageW,
		imageH,
		imageW
		);
	getLastCudaError("convolutionColumnsKernel() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(
	float *h_Dst,
	float *h_Src,
	float *h_Kernel,
	int imageW,
	int imageH,
	int kernelR
	)
{
	for (int y = 0; y < imageH; y++)
		for (int x = 0; x < imageW; x++)
		{
			float sum = 0;

			for (int k = -kernelR; k <= kernelR; k++)
			{
				int d = x + k;

				if (d >= 0 && d < imageW)
					sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k];
			}

			h_Dst[y * imageW + x] = sum;
		}
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionColumnCPU(
	float *h_Dst,
	float *h_Src,
	float *h_Kernel,
	int imageW,
	int imageH,
	int kernelR
	)
{
	for (int y = 0; y < imageH; y++)
		for (int x = 0; x < imageW; x++)
		{
			float sum = 0;

			for (int k = -kernelR; k <= kernelR; k++)
			{
				int d = y + k;

				if (d >= 0 && d < imageH)
					sum += h_Src[d * imageW + x] * h_Kernel[kernelR - k];
			}

			h_Dst[y * imageW + x] = sum;
		}
}

void B4_Malloc(int argc, char** argv)
{
	printf("[%s] - Starting...\n", argv[0]);

	if (checkCmdLineFlag(argc, (const char **)argv, "iw"))
	{
		imageW = getCmdLineArgumentInt(argc, (const char **)argv, "iw");
	}

	// height of Matrix A
	if (checkCmdLineFlag(argc, (const char **)argv, "ih"))
	{
		imageH = getCmdLineArgumentInt(argc, (const char **)argv, "ih");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "iter"))
	{
		iterations = getCmdLineArgumentInt(argc, (const char **)argv, "iter");
	}
	//StopWatchInterface *hTimer = NULL;

	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
	//findCudaDevice(argc, (const char **)argv);

	//sdkCreateTimer(&hTimer);

	printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
	printf("Allocating and initializing host arrays...\n");
	//h_Kernel = (float *)malloc(KERNEL_LENGTH * sizeof(float));
	cudaMallocHost((void**)&h_Kernel, KERNEL_LENGTH * sizeof(float));
	//h_Input = (float *)malloc(imageW * imageH * sizeof(float));
	cudaMallocHost((void**)&h_Input, imageW * imageH * sizeof(float));
	//h_Buffer = (float *)malloc(imageW * imageH * sizeof(float));
	cudaMallocHost((void**)&h_Buffer, imageW * imageH * sizeof(float));
	h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
	//h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
	cudaMallocHost((void**)&h_OutputGPU, imageW * imageH * sizeof(float));
	srand(200);

	for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
	{
		h_Kernel[i] = (float)(rand() % 16);
	}

	for (unsigned i = 0; i < imageW * imageH; i++)
	{
		h_Input[i] = (float)(rand() % 16);
	}

	printf("Allocating and initializing CUDA arrays...\n");
	checkCudaErrors(cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float)));

}
void B4_H2D(cudaStream_t Stream)
{
	setConvolutionKernel(Stream);
	checkCudaErrors(cudaMemcpyAsync(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice, Stream));
}

void B4_Kernel(cudaStream_t Stream)
{
	for (int i = -1; i < iterations; i++)
	{
		//i == -1 -- warmup iteration
		/*if (i == 0)
		{
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);
		}*/

		convolutionRowsGPU(
			d_Buffer,
			d_Input,
			imageW,
			imageH,
			Stream
			);

		convolutionColumnsGPU(
			d_Output,
			d_Buffer,
			imageW,
			imageH,
			Stream
			);
	}
}
void B4_D2H(cudaStream_t Stream)
{
	printf("\nReading back GPU results...\n\n");
	checkCudaErrors(cudaMemcpyAsync(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost, Stream));
}

void B4_CL_Mem(cudaStream_t Stream)
{
	checkCudaErrors(cudaStreamSynchronize(Stream));
	printf("Checking the results...\n");
	printf(" ...running convolutionRowCPU()\n");
	convolutionRowCPU(
		h_Buffer,
		h_Input,
		h_Kernel,
		imageW,
		imageH,
		KERNEL_RADIUS
		);

	printf(" ...running convolutionColumnCPU()\n");
	convolutionColumnCPU(
		h_OutputCPU,
		h_Buffer,
		h_Kernel,
		imageW,
		imageH,
		KERNEL_RADIUS
		);
	printf(" ...comparing the results\n");
	double sum = 0, delta = 0;

	for (unsigned i = 0; i < imageW * imageH; i++)
	{
		delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
		sum += h_OutputCPU[i] * h_OutputCPU[i];
	}

	double L2norm = sqrt(delta / sum);
	printf(" ...Relative L2 norm: %E\n\n", L2norm);
	printf("Shutting down...\n");
	checkCudaErrors(cudaFree(d_Buffer));
	checkCudaErrors(cudaFree(d_Output));
	checkCudaErrors(cudaFree(d_Input));
	cudaFreeHost(h_OutputGPU);
	free(h_OutputCPU);
	cudaFreeHost(h_Buffer);
	cudaFreeHost(h_Input);
	cudaFreeHost(h_Kernel);
	if (L2norm > 1e-6)
	{
		printf("Test failed!\n");
		exit(EXIT_FAILURE);
	}

	printf("Test passed\n");
}