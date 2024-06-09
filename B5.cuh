
__device__ inline float cndGPU(float d)
{
	const float       A1 = 0.31938153f;
	const float       A2 = -0.356563782f;
	const float       A3 = 1.781477937f;
	const float       A4 = -1.821255978f;
	const float       A5 = 1.330274429f;
	const float RSQRT2PI = 0.39894228040143267793994605993438f;

	float
		K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

	float
		cnd = RSQRT2PI * __expf(-0.5f * d * d) *
		(K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

	if (d > 0)
		cnd = 1.0f - cnd;

	return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
	float &CallResult,
	float &PutResult,
	float S, //Stock price
	float X, //Option strike
	float T, //Option years
	float R, //Riskless rate
	float V  //Volatility rate
)
{
	float sqrtT, expRT;
	float d1, d2, CNDD1, CNDD2;

	sqrtT = __fdividef(1.0F, rsqrtf(T));
	d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
	d2 = d1 - V * sqrtT;

	CNDD1 = cndGPU(d1);
	CNDD2 = cndGPU(d2);

	//Calculate Call and Put simultaneously
	expRT = __expf(-R * T);
	CallResult = S * CNDD1 - X * expRT * CNDD2;
	PutResult = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__launch_bounds__(128)
__global__ void BlackScholesGPU(
	float2 * __restrict d_CallResult,
	float2 * __restrict d_PutResult,
	float2 * __restrict d_StockPrice,
	float2 * __restrict d_OptionStrike,
	float2 * __restrict d_OptionYears,
	float Riskfree,
	float Volatility,
	int optN
)
{
	////Thread index
	//const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
	////Total number of threads in execution grid
	//const int THREAD_N = blockDim.x * gridDim.x;

	const int opt = blockDim.x * blockIdx.x + threadIdx.x;

	// Calculating 2 options per thread to increase ILP (instruction level parallelism)
	if (opt < (optN / 2))
	{
		float callResult1, callResult2;
		float putResult1, putResult2;
		BlackScholesBodyGPU(
			callResult1,
			putResult1,
			d_StockPrice[opt].x,
			d_OptionStrike[opt].x,
			d_OptionYears[opt].x,
			Riskfree,
			Volatility
		);
		BlackScholesBodyGPU(
			callResult2,
			putResult2,
			d_StockPrice[opt].y,
			d_OptionStrike[opt].y,
			d_OptionYears[opt].y,
			Riskfree,
			Volatility
		);
		d_CallResult[opt] = make_float2(callResult1, callResult2);
		d_PutResult[opt] = make_float2(putResult1, putResult2);
	}
}

/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
static double CND(double d)
{
	const double       A1 = 0.31938153;
	const double       A2 = -0.356563782;
	const double       A3 = 1.781477937;
	const double       A4 = -1.821255978;
	const double       A5 = 1.330274429;
	const double RSQRT2PI = 0.39894228040143267793994605993438;

	double
		K = 1.0 / (1.0 + 0.2316419 * fabs(d));

	double
		cnd = RSQRT2PI * exp(-0.5 * d * d) *
		(K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

	if (d > 0)
		cnd = 1.0 - cnd;

	return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
static void BlackScholesBodyCPU(
	float &callResult,
	float &putResult,
	float Sf, //Stock price
	float Xf, //Option strike
	float Tf, //Option years
	float Rf, //Riskless rate
	float Vf  //Volatility rate
)
{
	double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

	double sqrtT = sqrt(T);
	double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
	double    d2 = d1 - V * sqrtT;
	double CNDD1 = CND(d1);
	double CNDD2 = CND(d2);

	//Calculate Call and Put simultaneously
	double expRT = exp(-R * T);
	callResult = (float)(S * CNDD1 - X * expRT * CNDD2);
	putResult = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}


////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
	float *h_CallResult,
	float *h_PutResult,
	float *h_StockPrice,
	float *h_OptionStrike,
	float *h_OptionYears,
	float Riskfree,
	float Volatility,
	int optN
)
{
	for (int opt = 0; opt < optN; opt++)
		BlackScholesBodyCPU(
			h_CallResult[opt],
			h_PutResult[opt],
			h_StockPrice[opt],
			h_OptionStrike[opt],
			h_OptionYears[opt],
			Riskfree,
			Volatility
		);
}

/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
* This sample evaluates fair call and put prices for a
* given set of European options by Black-Scholes formula.
* See supplied whitepaper for more explanations.
*/


#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
	float *h_CallResult,
	float *h_PutResult,
	float *h_StockPrice,
	float *h_OptionStrike,
	float *h_OptionYears,
	float Riskfree,
	float Volatility,
	int optN
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
int OPT_N = 40000000;
const int  NUM_ITERATIONS = 1;


int          OPT_SZ = OPT_N * sizeof(float);
const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )

float
//Results calculated by CPU for reference
*h_CallResultCPU,
*h_PutResultCPU,
//CPU copy of GPU results
*h_CallResultGPU,
*h_PutResultGPU,
//CPU instance of input data
*h_StockPrice,
*h_OptionStrike,
*h_OptionYears;

//'d_' prefix - GPU (device) memory space
float
//Results calculated by GPU
*d_CallResult,
*d_PutResult,
//GPU instance of input data
*d_StockPrice,
*d_OptionStrike,
*d_OptionYears;

double
B5_delta, B5_ref, B5_sum_delta, B5_sum_ref, B5_max_delta, B5_L1norm, B5_gpuTime;


////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(
	float *h_CallResult,
	float *h_PutResult,
	float *h_StockPrice,
	float *h_OptionStrike,
	float *h_OptionYears,
	float Riskfree,
	float Volatility,
	int optN
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
void B5_Malloc(int argc, char** argv)
{
	if (checkCmdLineFlag(argc, (const char **)argv, "OPT_N"))
	{
		OPT_N = getCmdLineArgumentInt(argc, (const char **)argv, "OPT_N");
	}
	OPT_SZ = OPT_N * sizeof(float);
	// Start logs
	printf("[%s] - Starting...\n", argv[0]);

	//'h_' prefix - CPU (host) memory space

	//StopWatchInterface *hTimer = NULL;
	int i;

	findCudaDevice(argc, (const char **)argv);

	//sdkCreateTimer(&hTimer);

	printf("Initializing data...\n");
	printf("...allocating CPU memory for options.\n");
	h_CallResultCPU = (float *)malloc(OPT_SZ);
	h_PutResultCPU = (float *)malloc(OPT_SZ);
	cudaMallocHost((void **)&h_CallResultGPU, OPT_SZ);
	cudaMallocHost((void **)&h_PutResultGPU, OPT_SZ);
	cudaMallocHost((void **)&h_StockPrice, OPT_SZ);
	cudaMallocHost((void **)&h_OptionStrike, OPT_SZ);
	cudaMallocHost((void **)&h_OptionYears, OPT_SZ);

	printf("...allocating GPU memory for options.\n");
	checkCudaErrors(cudaMalloc((void **)&d_CallResult, OPT_SZ));
	checkCudaErrors(cudaMalloc((void **)&d_PutResult, OPT_SZ));
	checkCudaErrors(cudaMalloc((void **)&d_StockPrice, OPT_SZ));
	checkCudaErrors(cudaMalloc((void **)&d_OptionStrike, OPT_SZ));
	checkCudaErrors(cudaMalloc((void **)&d_OptionYears, OPT_SZ));

	printf("...generating input data in CPU mem.\n");
	srand(5347);

	//Generate options set
	for (i = 0; i < OPT_N; i++)
	{
		h_CallResultCPU[i] = 0.0f;
		h_PutResultCPU[i] = -1.0f;
		h_StockPrice[i] = RandFloat(5.0f, 30.0f);
		h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
		h_OptionYears[i] = RandFloat(0.25f, 10.0f);
	}

	printf("...copying input data to GPU mem.\n");
	//Copy options data to GPU memory for further processing
	;
	printf("Data init done.\n\n");


	printf("Executing Black-Scholes GPU kernel (%i iterations)...\n", NUM_ITERATIONS);
}
void B5_H2D(cudaStream_t Stream)
{
	printf("Copy input data from the host memory to the CUDA device\n");
	checkCudaErrors(cudaMemcpyAsync(d_StockPrice, h_StockPrice, OPT_SZ, cudaMemcpyHostToDevice, Stream));
	checkCudaErrors(cudaMemcpyAsync(d_OptionStrike, h_OptionStrike, OPT_SZ, cudaMemcpyHostToDevice, Stream));
	checkCudaErrors(cudaMemcpyAsync(d_OptionYears, h_OptionYears, OPT_SZ, cudaMemcpyHostToDevice, Stream));
}

void B5_Kernel(cudaStream_t Stream)
{
	for (int i = 0; i < NUM_ITERATIONS; i++)
	{
		BlackScholesGPU << <DIV_UP((OPT_N / 2), 128), 128/*480, 128*/, 0, Stream >> >(
			(float2 *)d_CallResult,
			(float2 *)d_PutResult,
			(float2 *)d_StockPrice,
			(float2 *)d_OptionStrike,
			(float2 *)d_OptionYears,
			RISKFREE,
			VOLATILITY,
			OPT_N
			);
		getLastCudaError("BlackScholesGPU() execution failed\n");
	}
}
void B5_D2H(cudaStream_t Stream)
{
	checkCudaErrors(cudaMemcpyAsync(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost, Stream));
	checkCudaErrors(cudaMemcpyAsync(h_PutResultGPU, d_PutResult, OPT_SZ, cudaMemcpyDeviceToHost, Stream));
}

void B5_CL_Mem(cudaStream_t Stream)
{
	checkCudaErrors(cudaStreamSynchronize(Stream));
	printf("\nReading back GPU results...\n");
	//Read back GPU results to compare them to CPU results



	printf("Checking the results...\n");
	printf("...running CPU calculations.\n\n");
	//Calculate options values on CPU
	BlackScholesCPU(
		h_CallResultCPU,
		h_PutResultCPU,
		h_StockPrice,
		h_OptionStrike,
		h_OptionYears,
		RISKFREE,
		VOLATILITY,
		OPT_N
	);

	printf("Comparing the results...\n");
	//Calculate max absolute difference and L1 distance
	//between CPU and GPU results
	B5_sum_delta = 0;
	B5_sum_ref = 0;
	B5_max_delta = 0;

	for (int i = 0; i < OPT_N; i++)
	{
		B5_ref = h_CallResultCPU[i];
		B5_delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

		if (B5_delta > B5_max_delta)
		{
			B5_max_delta = B5_delta;
		}

		B5_sum_delta += B5_delta;
		B5_sum_ref += fabs(B5_ref);
	}

	B5_L1norm = B5_sum_delta / B5_sum_ref;
	printf("L1 norm: %E\n", B5_L1norm);
	printf("Max absolute error: %E\n\n", B5_max_delta);

	printf("Shutting down...\n");
	printf("...releasing GPU memory.\n");

	printf("...releasing CPU memory.\n");
	checkCudaErrors(cudaFree(d_OptionYears));
	checkCudaErrors(cudaFree(d_OptionStrike));
	checkCudaErrors(cudaFree(d_StockPrice));
	checkCudaErrors(cudaFree(d_PutResult));
	checkCudaErrors(cudaFree(d_CallResult));
	cudaFreeHost(h_OptionYears);
	cudaFreeHost(h_OptionStrike);
	cudaFreeHost(h_StockPrice);
	cudaFreeHost(h_PutResultGPU);
	cudaFreeHost(h_CallResultGPU);
	free(h_PutResultCPU);
	free(h_CallResultCPU);

	printf("Shutdown done.\n");

	printf("\n[BlackScholes] - Test Summary\n");

	if (B5_L1norm > 1e-6)
	{
		printf("Test failed!\n");
		exit(EXIT_FAILURE);
	}

	printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n\n");
	printf("Test passed\n");
}
