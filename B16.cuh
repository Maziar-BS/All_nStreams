
#ifdef RD_WG_SIZE_0_0                                                            
#define BLOCK_SIZE RD_WG_SIZE_0_0                                        
#elif defined(RD_WG_SIZE_0)                                                      
#define BLOCK_SIZE RD_WG_SIZE_0                                          
#elif defined(RD_WG_SIZE)                                                        
#define BLOCK_SIZE RD_WG_SIZE                                            
#else                                                                                    
#define BLOCK_SIZE 16                                                            
#endif                                                                                   

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

float *FilesavingTemp, *FilesavingPower, *MatrixOut;
float *MatrixTemp[2], *MatrixPower;
int hotspot_size;

int total_iterations, num_iterations;
int src = 1, dst = 0;
int col, row, borderCols, borderRows;
float Cap, Rx, Ry, Rz, step;
dim3 B16_Grid;
dim3 B16_Block;

void
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

void writeoutput(int grid_rows, int grid_cols, char *file) {

	int i, j, index = 0;
	FILE *fp;
	char str[STR_SIZE];

	if ((fp = fopen(file, "w")) == 0)
		printf("The file was not opened\n");


	for (i = 0; i < grid_rows; i++)
		for (j = 0; j < grid_cols; j++)
		{

			sprintf(str, "%d\t%g\n", index, MatrixOut[i*grid_cols + j]);
			fputs(str, fp);
			index++;
		}

	fclose(fp);
}


void readinput_t(int grid_rows, int grid_cols, char *file) {

	int i, j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if ((fp = fopen(file, "r")) == 0)
		printf("The file was not opened\n");


	for (i = 0; i <= grid_rows - 1; i++)
		for (j = 0; j <= grid_cols - 1; j++)
		{
			fgets(str, STR_SIZE, fp);
			if (feof(fp))
				fatal("not enough lines in file");
			//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
			if ((sscanf(str, "%f", &val) != 1))
				fatal("invalid file format");
			FilesavingTemp[i*grid_cols + j] = val;
		}

	fclose(fp);

}

void readinput_p(int grid_rows, int grid_cols, char *file) {

	int i, j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if ((fp = fopen(file, "r")) == 0)
		printf("The file was not opened\n");


	for (i = 0; i <= grid_rows - 1; i++)
		for (j = 0; j <= grid_cols - 1; j++)
		{
			fgets(str, STR_SIZE, fp);
			if (feof(fp))
				fatal("not enough lines in file");
			//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i - 1)*(grid_cols - 2) + j - 1)))
			if ((sscanf(str, "%f", &val) != 1))
				fatal("invalid file format");
			FilesavingPower[i*grid_cols + j] = val;
		}

	fclose(fp);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define B16_MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void calculate_temp(int iteration,  //number of iteration
	float *power,   //power input
	float *temp_src,    //temperature input/output
	float *temp_dst,    //temperature input/output
	int grid_cols,  //Col of grid
	int grid_rows,  //Row of grid
	int border_cols,  // border offset 
	int border_rows,  // border offset
	float Cap,      //Capacitance
	float Rx,
	float Ry,
	float Rz,
	float step,
	float time_elapsed) {

	__shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

	float amb_temp = 80.0;
	float step_div_Cap;
	float Rx_1, Ry_1, Rz_1;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	step_div_Cap = step / Cap;

	Rx_1 = 1 / Rx;
	Ry_1 = 1 / Ry;
	Rz_1 = 1 / Rz;

	// each block finally computes result for a small block
	// after N iterations. 
	// it is the non-overlapping small blocks that cover 
	// all the input data

	// calculate the small block size
	int small_block_rows = BLOCK_SIZE - iteration * 2;//EXPAND_RATE
	int small_block_cols = BLOCK_SIZE - iteration * 2;//EXPAND_RATE

													  // calculate the boundary for the block according to 
													  // the boundary of its small block
	int blkY = small_block_rows * by - border_rows;
	int blkX = small_block_cols * bx - border_cols;
	int blkYmax = blkY + BLOCK_SIZE - 1;
	int blkXmax = blkX + BLOCK_SIZE - 1;

	// calculate the global thread coordination
	int yidx = blkY + ty;
	int xidx = blkX + tx;

	// load data if it is within the valid input range
	int loadYidx = yidx, loadXidx = xidx;
	int index = grid_cols * loadYidx + loadXidx;

	if (IN_RANGE(loadYidx, 0, grid_rows - 1) && IN_RANGE(loadXidx, 0, grid_cols - 1)) {
		temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
		power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
	}
	__syncthreads();

	// effective range within this block that falls within 
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validYmin = (blkY < 0) ? -blkY : 0;
	int validYmax = (blkYmax > grid_rows - 1) ? BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) : BLOCK_SIZE - 1;
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > grid_cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) : BLOCK_SIZE - 1;

	int N = ty - 1;
	int S = ty + 1;
	int W = tx - 1;
	int E = tx + 1;

	N = (N < validYmin) ? validYmin : N;
	S = (S > validYmax) ? validYmax : S;
	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool computed;
	for (int i = 0; i<iteration; i++) {
		computed = false;
		if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && \
			IN_RANGE(ty, i + 1, BLOCK_SIZE - i - 2) && \
			IN_RANGE(tx, validXmin, validXmax) && \
			IN_RANGE(ty, validYmin, validYmax)) {
			computed = true;
			temp_t[ty][tx] = temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
				(temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 +
				(temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 +
				(amb_temp - temp_on_cuda[ty][tx]) * Rz_1);

		}
		__syncthreads();
		if (i == iteration - 1)
			break;
		if (computed)	 //Assign the computation range
			temp_on_cuda[ty][tx] = temp_t[ty][tx];
		__syncthreads();
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the 
	// small block perform the calculation and switch on ``computed''
	if (computed) {
		temp_dst[index] = temp_t[ty][tx];
	}
}

void compute_tran_temp(int col, int row)
{
	float grid_height = chip_height / row;
	float grid_width = chip_width / col;
	Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	step = PRECISION / max_slope;
}

void run(int argc, char** argv)
{
	int grid_rows, grid_cols;
	char *tfile, *pfile, *ofile;
	total_iterations = 60;
	int pyramid_height = 1; // number of iterations

	if (checkCmdLineFlag(argc, (const char **)argv, "GRC"))
	{
		grid_rows = getCmdLineArgumentInt(argc, (const char **)argv, "GRC");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "GRC"))
	{
		grid_cols = getCmdLineArgumentInt(argc, (const char **)argv, "GRC");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "PH"))
	{
		pyramid_height = getCmdLineArgumentInt(argc, (const char **)argv, "PH");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "TI"))
	{
		total_iterations = getCmdLineArgumentInt(argc, (const char **)argv, "TI");
	}

	tfile = "F:\\Coding\\All_nStreams\\All_nStreams\\All_nStreams\\Data\\temp_1024";
	pfile = "F:\\Coding\\All_nStreams\\All_nStreams\\All_nStreams\\Data\\power_1024";
	ofile = "F:\\Coding\\All_nStreams\\All_nStreams\\All_nStreams\\Data\\out";

	/*if (checkCmdLineFlag(argc, (const char **)argv, "TF"))
	{
	*tfile = getCmdLineArgumentInt(argc, (const char **)argv, "TF");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "PF"))
	{
	*pfile = getCmdLineArgumentInt(argc, (const char **)argv, "PF");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "OF"))
	{
	*ofile = getCmdLineArgumentInt(argc, (const char **)argv, "OF");
	}*/
	hotspot_size = grid_rows * grid_cols;
	num_iterations = pyramid_height;
	/* --------------- pyramid parameters --------------- */
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
	borderCols = (pyramid_height)*EXPAND_RATE / 2;
	borderRows = (pyramid_height)*EXPAND_RATE / 2;
	int smallBlockCol = BLOCK_SIZE - (pyramid_height)*EXPAND_RATE;
	int smallBlockRow = BLOCK_SIZE - (pyramid_height)*EXPAND_RATE;
	int blockCols = grid_cols / smallBlockCol + ((grid_cols%smallBlockCol == 0) ? 0 : 1);
	int blockRows = grid_rows / smallBlockRow + ((grid_rows%smallBlockRow == 0) ? 0 : 1);

	cudaMallocHost((void **)&FilesavingTemp, hotspot_size * sizeof(float));

	cudaMallocHost((void **)&FilesavingPower, hotspot_size * sizeof(float));

	cudaMallocHost((void **)&MatrixOut, hotspot_size * sizeof(float));

	if (!FilesavingPower || !FilesavingTemp || !MatrixOut)
		fatal("unable to allocate memory");

	printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n", \
		pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);

	readinput_t(grid_rows, grid_cols, tfile);
	readinput_p(grid_rows, grid_cols, pfile);


	cudaMalloc((void**)&MatrixTemp[0], sizeof(float)*hotspot_size);
	cudaMalloc((void**)&MatrixTemp[1], sizeof(float)*hotspot_size);
	cudaMalloc((void**)&MatrixPower, sizeof(float)*hotspot_size);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(blockCols, blockRows);
	B16_Block = dimBlock;
	B16_Grid = dimGrid;
	printf("Start computing the transient temperature\n");
	compute_tran_temp(grid_cols, grid_rows);
	printf("Ending simulation\n");
	writeoutput(grid_rows, grid_cols, ofile);
}

void B16_Malloc(int argc, char** argv)
{
	printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
	run(argc, argv);
}

void B16_H2D(cudaStream_t Stream)
{
	printf("Copy input data from the host memory to the CUDA device\n");

	cudaMemcpyAsync(MatrixTemp[0], FilesavingTemp, sizeof(float)*hotspot_size, cudaMemcpyHostToDevice, Stream);
	cudaMemcpyAsync(MatrixPower, FilesavingPower, sizeof(float)*hotspot_size, cudaMemcpyHostToDevice, Stream);
}

void B16_Kernel(cudaStream_t Stream)
{
	float t;
	float time_elapsed;
	time_elapsed = 0.001;

	for (t = 0; t < total_iterations; t += num_iterations) {
		int temp = src;
		src = dst;
		dst = temp;
		calculate_temp << <B16_Grid, B16_Block, 0, Stream >> >(B16_MIN(num_iterations, total_iterations - t), MatrixPower, MatrixTemp[src], MatrixTemp[dst], \
			col, row, borderCols, borderRows, Cap, Rx, Ry, Rz, step, time_elapsed);
	}

}
void B16_D2H(cudaStream_t Stream)
{
	cudaMemcpyAsync(MatrixOut, MatrixTemp[dst], sizeof(float)*hotspot_size, cudaMemcpyDeviceToHost, Stream);
}

void B16_CL_Mem(cudaStream_t Stream)
{
	cudaStreamSynchronize(Stream);
	cudaFree(MatrixPower);
	cudaFree(MatrixTemp[0]);
	cudaFree(MatrixTemp[1]);
	cudaFreeHost(FilesavingPower);
	cudaFreeHost(FilesavingTemp);
	cudaFreeHost(MatrixOut);
	printf("The execution is completed.    ");
}
