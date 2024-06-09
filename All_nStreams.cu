#include "All_Benchmarks.cuh"


void CL_Mem(int argc, char** argv, cudaStream_t Stream, int B)
{
	switch (B)
	{
	case 1:
		B1_CL_Mem(Stream);
		break;
	case 2:
		B2_CL_Mem(Stream);
		break;
	case 3:
		B3_CL_Mem(Stream);
		break;
	case 4:
		B4_CL_Mem(Stream);
		break;
	case 5:
		B5_CL_Mem(Stream);
		break;
	case 6:
		B6_CL_Mem(Stream);
		break;
	case 7:
		B7_CL_Mem(Stream);
		break;
	case 8:
		B8_CL_Mem(Stream);
		break;
	case 9:
		B9_CL_Mem(Stream);
		break;
	case 10:
		B10_CL_Mem(Stream);
		break;
	case 11:
		B11_CL_Mem(Stream);
		break;
	case 12:
		B12_CL_Mem(Stream);
		break;
	case 13:
		B13_CL_Mem(Stream);
		break;
	case 14:
		B14_CL_Mem(Stream);
		break;
	case 15:
		B15_CL_Mem(Stream);
		break;
	case 16:
		B16_CL_Mem(Stream);
		break;
	}
}

void D2H(int argc, char** argv, cudaStream_t Stream, int B)
{
	switch (B)
	{
	case 1:
		B1_D2H(Stream);
		break;
	case 2:
		B2_D2H(Stream);
		break;
	case 3:
		B3_D2H(Stream);
		break;
	case 4:
		B4_D2H(Stream);
		break;
	case 5:
		B5_D2H(Stream);
		break;
	case 6:
		B6_D2H(Stream);
		break;
	case 7:
		B7_D2H(Stream);
		break;
	case 8:
		B8_D2H(Stream);
		break;
	case 9:
		B9_D2H(Stream);
		break;
	case 10:
		B10_D2H(Stream);
		break;
	case 11:
		B11_D2H(Stream);
		break;
	case 12:
		B12_D2H(Stream);
		break;
	case 13:
		B13_D2H(Stream);
		break;
	case 14:
		B14_D2H(Stream);
		break;
	case 15:
		B15_D2H(Stream);
		break;
	case 16:
		B16_D2H(Stream);
		break;
	}
}

void Kernel_Invocation(int argc, char** argv, cudaStream_t Stream, int B)
{
	switch (B)
	{
	case 1:
		B1_Kernel(Stream);
		break;
	case 2:
		B2_Kernel(Stream);
		break;
	case 3:
		B3_Kernel(Stream);
		break;
	case 4:
		B4_Kernel(Stream);
		break;
	case 5:
		B5_Kernel(Stream);
		break;
	case 6:
		B6_Kernel(Stream);
		break;
	case 7:
		B7_Kernel(Stream);
		break;
	case 8:
		B8_Kernel(Stream);
		break;
	case 9:
		B9_Kernel(Stream);
		break;
	case 10:
		B10_Kernel(Stream);
		break;
	case 11:
		B11_Kernel(Stream);
		break;
	case 12:
		B12_Kernel(Stream);
		break;
	case 13:
		B13_Kernel(Stream);
		break;
	case 14:
		B14_Kernel(Stream);
		break;
	case 15:
		B15_Kernel(Stream);
		break;
	case 16:
		B16_Kernel(Stream);
		break;
	}
}

void H2D(int argc, char** argv, cudaStream_t Stream, int B)
{
	switch (B)
	{
	case 1:
		B1_H2D(Stream);
		break;
	case 2:
		B2_H2D(Stream);
		break;
	case 3:
		B3_H2D(Stream);
		break;
	case 4:
		B4_H2D(Stream);
		break;
	case 5:
		B5_H2D(Stream);
		break;
	case 6:
		B6_H2D(Stream);
		break;
	case 7:
		B7_H2D(Stream);
		break;
	case 8:
		B8_H2D(Stream);
		break;
	case 9:
		B9_H2D(Stream);
		break;
	case 10:
		B10_H2D(Stream);
		break;
	case 11:
		B11_H2D(Stream);
		break;
	case 12:
		B12_H2D(Stream);
		break;
	case 13:
		B13_H2D(Stream);
		break;
	case 14:
		B14_H2D(Stream);
		break;
	case 15:
		B15_H2D(Stream);
		break;
	case 16:
		B16_H2D(Stream);
		break;
	}
}

void Allocation(int argc, char** argv, int B)
{
	switch (B)
	{
	case 1:
		B1_Malloc(argc, argv);
		break;
	case 2:
		B2_Malloc(argc, argv);
		break;
	case 3:
		B3_Malloc(argc, argv);
		break;
	case 4:
		B4_Malloc(argc, argv);
		break;
	case 5:
		B5_Malloc(argc, argv);
		break;
	case 6:
		B6_Malloc(argc, argv);
		break;
	case 7:
		B7_Malloc(argc, argv);
		break;
	case 8:
		B8_Malloc(argc, argv);
		break;
	case 9:
		B9_Malloc(argc, argv);
		break;
	case 10:
		B10_Malloc(argc, argv);
		break;
	case 11:
		B11_Malloc(argc, argv);
		break;
	case 12:
		B12_Malloc(argc, argv);
		break;
	case 13:
		B13_Malloc(argc, argv);
		break;
	case 14:
		B14_Malloc(argc, argv);
		break;
	case 15:
		B15_Malloc(argc, argv);
		break;
	case 16:
		B16_Malloc(argc, argv);
		break;
	}
}

void nStreams_Invocation(int argc, char** argv, int *S, int NumofStreams, cudaStream_t Stream[6])
{
	for (int i = 0; i < NumofStreams; i++)
	{
		switch (S[i])
		{
		case 1:
			Allocation(argc, argv, 1);
			break;
		case 2:
			Allocation(argc, argv, 2);
			break;
		case 3:
			Allocation(argc, argv, 3);
			break;
		case 4:
			Allocation(argc, argv, 4);
			break;
		case 5:
			Allocation(argc, argv, 5);
			break;
		case 6:
			Allocation(argc, argv, 6);
			break;
		case 7:
			Allocation(argc, argv, 7);
			break;
		case 8:
			Allocation(argc, argv, 8);
			break;
		case 9:
			Allocation(argc, argv, 9);
			break;
		case 10:
			Allocation(argc, argv, 10);
			break;
		case 11:
			Allocation(argc, argv, 11);
			break;
		case 12:
			Allocation(argc, argv, 12);
			break;
		case 13:
			Allocation(argc, argv, 13);
			break;
		case 14:
			Allocation(argc, argv, 14);
			break;
		case 15:
			Allocation(argc, argv, 15);
			break;
		case 16:
			Allocation(argc, argv, 16);
			break;
		}
	}
	/*switch (S2)
	{
		case 1:
			Allocation(argc, argv, 1);
			break;
		case 2:
			Allocation(argc, argv, 2);
			break;
		case 3:
			Allocation(argc, argv, 3);
			break;
		case 4:
			Allocation(argc, argv, 4);
			break;
		case 5:
			Allocation(argc, argv, 5);
			break;
		case 6:
			Allocation(argc, argv, 6);
			break;
		case 7:
			Allocation(argc, argv, 7);
			break;
		case 8:
			Allocation(argc, argv, 8);
			break;
		case 9:
			Allocation(argc, argv, 9);
			break;
		case 10:
			Allocation(argc, argv, 10);
			break;
		case 11:
			Allocation(argc, argv, 11);
			break;
		case 12:
			Allocation(argc, argv, 12);
			break;
		case 13:
			Allocation(argc, argv, 13);
			break;
		case 14:
			Allocation(argc, argv, 14);
			break;
		case 15:
			Allocation(argc, argv, 15);
			break;
		case 16:
			Allocation(argc, argv, 16);
			break;
	}
	switch (S3)
	{
		case 1:
			Allocation(argc, argv, 1);
			break;
		case 2:
			Allocation(argc, argv, 2);
			break;
		case 3:
			Allocation(argc, argv, 3);
			break;
		case 4:
			Allocation(argc, argv, 4);
			break;
		case 5:
			Allocation(argc, argv, 5);
			break;
		case 6:
			Allocation(argc, argv, 6);
			break;
		case 7:
			Allocation(argc, argv, 7);
			break;
		case 8:
			Allocation(argc, argv, 8);
			break;
		case 9:
			Allocation(argc, argv, 9);
			break;
		case 10:
			Allocation(argc, argv, 10);
			break;
		case 11:
			Allocation(argc, argv, 11);
			break;
		case 12:
			Allocation(argc, argv, 12);
			break;
		case 13:
			Allocation(argc, argv, 13);
			break;
		case 14:
			Allocation(argc, argv, 14);
			break;
		case 15:
			Allocation(argc, argv, 15);
			break;
		case 16:
			Allocation(argc, argv, 16);
			break;
	}*/
	// H2D phase
	for (int i = 0; i < NumofStreams; i++)
	{
		switch (S[i])
		{
		case 1:
			H2D(argc, argv, Stream[i], 1);
			break;
		case 2:
			H2D(argc, argv, Stream[i], 2);
			break;
		case 3:
			H2D(argc, argv, Stream[i], 3);
			break;
		case 4:
			H2D(argc, argv, Stream[i], 4);
			break;
		case 5:
			H2D(argc, argv, Stream[i], 5);
			break;
		case 6:
			H2D(argc, argv, Stream[i], 6);
			break;
		case 7:
			H2D(argc, argv, Stream[i], 7);
			break;
		case 8:
			H2D(argc, argv, Stream[i], 8);
			break;
		case 9:
			H2D(argc, argv, Stream[i], 9);
			break;
		case 10:
			H2D(argc, argv, Stream[i], 10);
			break;
		case 11:
			H2D(argc, argv, Stream[i], 11);
			break;
		case 12:
			H2D(argc, argv, Stream[i], 12);
			break;
		case 13:
			H2D(argc, argv, Stream[i], 13);
			break;
		case 14:
			H2D(argc, argv, Stream[i], 14);
			break;
		case 15:
			H2D(argc, argv, Stream[i], 15);
			break;
		case 16:
			H2D(argc, argv, Stream[i], 16);
			break;
		}
	}
	/*switch (S1)
	{
		case 1:
			H2D(argc, argv, Stream[0], 1);
			break;
		case 2:
			H2D(argc, argv, Stream[0], 2);
			break;
		case 3:
			H2D(argc, argv, Stream[0], 3);
			break;
		case 4:
			H2D(argc, argv, Stream[0], 4);
			break;
		case 5:
			H2D(argc, argv, Stream[0], 5);
			break;
		case 6:
			H2D(argc, argv, Stream[0], 6);
			break;
		case 7:
			H2D(argc, argv, Stream[0], 7);
			break;
		case 8:
			H2D(argc, argv, Stream[0], 8);
			break;
		case 9:
			H2D(argc, argv, Stream[0], 9);
			break;
		case 10:
			H2D(argc, argv, Stream[0], 10);
			break;
		case 11:
			H2D(argc, argv, Stream[0], 11);
			break;
		case 12:
			H2D(argc, argv, Stream[0], 12);
			break;
		case 13:
			H2D(argc, argv, Stream[0], 13);
			break;
		case 14:
			H2D(argc, argv, Stream[0], 14);
			break;
		case 15:
			H2D(argc, argv, Stream[0], 15);
			break;
		case 16:
			H2D(argc, argv, Stream[0], 16);
			break;
	}
	switch (S2)
	{
		case 1:
			H2D(argc, argv, Stream[1], 1);
			break;
		case 2:
			H2D(argc, argv, Stream[1], 2);
			break;
		case 3:
			H2D(argc, argv, Stream[1], 3);
			break;
		case 4:
			H2D(argc, argv, Stream[1], 4);
			break;
		case 5:
			H2D(argc, argv, Stream[1], 5);
			break;
		case 6:
			H2D(argc, argv, Stream[1], 6);
			break;
		case 7:
			H2D(argc, argv, Stream[1], 7);
			break;
		case 8:
			H2D(argc, argv, Stream[1], 8);
			break;
		case 9:
			H2D(argc, argv, Stream[1], 9);
			break;
		case 10:
			H2D(argc, argv, Stream[1], 10);
			break;
		case 11:
			H2D(argc, argv, Stream[1], 11);
			break;
		case 12:
			H2D(argc, argv, Stream[1], 12);
			break;
		case 13:
			H2D(argc, argv, Stream[1], 13);
			break;
		case 14:
			H2D(argc, argv, Stream[1], 14);
			break;
		case 15:
			H2D(argc, argv, Stream[1], 15);
			break;
		case 16:
			H2D(argc, argv, Stream[1], 16);
			break;
	}
	switch (S3)
	{
		case 1:
			H2D(argc, argv, Stream[2], 1);
			break;
		case 2:
			H2D(argc, argv, Stream[2], 2);
			break;
		case 3:
			H2D(argc, argv, Stream[2], 3);
			break;
		case 4:
			H2D(argc, argv, Stream[2], 4);
			break;
		case 5:
			H2D(argc, argv, Stream[2], 5);
			break;
		case 6:
			H2D(argc, argv, Stream[2], 6);
			break;
		case 7:
			H2D(argc, argv, Stream[2], 7);
			break;
		case 8:
			H2D(argc, argv, Stream[2], 8);
			break;
		case 9:
			H2D(argc, argv, Stream[2], 9);
			break;
		case 10:
			H2D(argc, argv, Stream[2], 10);
			break;
		case 11:
			H2D(argc, argv, Stream[2], 11);
			break;
		case 12:
			H2D(argc, argv, Stream[2], 12);
			break;
		case 13:
			H2D(argc, argv, Stream[2], 13);
			break;
		case 14:
			H2D(argc, argv, Stream[2], 14);
			break;
		case 15:
			H2D(argc, argv, Stream[2], 15);
			break;
		case 16:
			H2D(argc, argv, Stream[2], 16);
			break;
	}*/
	// Kernel Executions
	for (int i = 0; i < NumofStreams; i++)
	{
		switch (S[i])
		{
		case 1:
			Kernel_Invocation(argc, argv, Stream[i], 1);
			break;
		case 2:
			Kernel_Invocation(argc, argv, Stream[i], 2);
			break;
		case 3:
			Kernel_Invocation(argc, argv, Stream[i], 3);
			break;
		case 4:
			Kernel_Invocation(argc, argv, Stream[i], 4);
			break;
		case 5:
			Kernel_Invocation(argc, argv, Stream[i], 5);
			break;
		case 6:
			Kernel_Invocation(argc, argv, Stream[i], 6);
			break;
		case 7:
			Kernel_Invocation(argc, argv, Stream[i], 7);
			break;
		case 8:
			Kernel_Invocation(argc, argv, Stream[i], 8);
			break;
		case 9:
			Kernel_Invocation(argc, argv, Stream[i], 9);
			break;
		case 10:
			Kernel_Invocation(argc, argv, Stream[i], 10);
			break;
		case 11:
			Kernel_Invocation(argc, argv, Stream[i], 11);
			break;
		case 12:
			Kernel_Invocation(argc, argv, Stream[i], 12);
			break;
		case 13:
			Kernel_Invocation(argc, argv, Stream[i], 13);
			break;
		case 14:
			Kernel_Invocation(argc, argv, Stream[i], 14);
			break;
		case 15:
			Kernel_Invocation(argc, argv, Stream[i], 15);
			break;
		case 16:
			Kernel_Invocation(argc, argv, Stream[i], 16);
			break;
		}
	}
	/*switch (S1)
	{
		case 1:
			Kernel_Invocation(argc, argv, Stream[0], 1);
			break;
		case 2:
			Kernel_Invocation(argc, argv, Stream[0], 2);
			break;
		case 3:
			Kernel_Invocation(argc, argv, Stream[0], 3);
			break;
		case 4:
			Kernel_Invocation(argc, argv, Stream[0], 4);
			break;
		case 5:
			Kernel_Invocation(argc, argv, Stream[0], 5);
			break;
		case 6:
			Kernel_Invocation(argc, argv, Stream[0], 6);
			break;
		case 7:
			Kernel_Invocation(argc, argv, Stream[0], 7);
			break;
		case 8:
			Kernel_Invocation(argc, argv, Stream[0], 8);
			break;
		case 9:
			Kernel_Invocation(argc, argv, Stream[0], 9);
			break;
		case 10:
			Kernel_Invocation(argc, argv, Stream[0], 10);
			break;
		case 11:
			Kernel_Invocation(argc, argv, Stream[0], 11);
			break;
		case 12:
			Kernel_Invocation(argc, argv, Stream[0], 12);
			break;
		case 13:
			Kernel_Invocation(argc, argv, Stream[0], 13);
			break;
		case 14:
			Kernel_Invocation(argc, argv, Stream[0], 14);
			break;
		case 15:
			Kernel_Invocation(argc, argv, Stream[0], 15);
			break;
		case 16:
			Kernel_Invocation(argc, argv, Stream[0], 16);
			break;
	}
	switch (S2)
	{
		case 1:
			Kernel_Invocation(argc, argv, Stream[1], 1);
			break;
		case 2:
			Kernel_Invocation(argc, argv, Stream[1], 2);
			break;
		case 3:
			Kernel_Invocation(argc, argv, Stream[1], 3);
			break;
		case 4:
			Kernel_Invocation(argc, argv, Stream[1], 4);
			break;
		case 5:
			Kernel_Invocation(argc, argv, Stream[1], 5);
			break;
		case 6:
			Kernel_Invocation(argc, argv, Stream[1], 6);
			break;
		case 7:
			Kernel_Invocation(argc, argv, Stream[1], 7);
			break;
		case 8:
			Kernel_Invocation(argc, argv, Stream[1], 8);
			break;
		case 9:
			Kernel_Invocation(argc, argv, Stream[1], 9);
			break;
		case 10:
			Kernel_Invocation(argc, argv, Stream[1], 10);
			break;
		case 11:
			Kernel_Invocation(argc, argv, Stream[1], 11);
			break;
		case 12:
			Kernel_Invocation(argc, argv, Stream[1], 12);
			break;
		case 13:
			Kernel_Invocation(argc, argv, Stream[1], 13);
			break;
		case 14:
			Kernel_Invocation(argc, argv, Stream[1], 14);
			break;
		case 15:
			Kernel_Invocation(argc, argv, Stream[1], 15);
			break;
		case 16:
			Kernel_Invocation(argc, argv, Stream[1], 16);
			break;
	}
	switch (S3)
	{
		case 1:
			Kernel_Invocation(argc, argv, Stream[2], 1);
			break;
		case 2:
			Kernel_Invocation(argc, argv, Stream[2], 2);
			break;
		case 3:
			Kernel_Invocation(argc, argv, Stream[2], 3);
			break;
		case 4:
			Kernel_Invocation(argc, argv, Stream[2], 4);
			break;
		case 5:
			Kernel_Invocation(argc, argv, Stream[2], 5);
			break;
		case 6:
			Kernel_Invocation(argc, argv, Stream[2], 6);
			break;
		case 7:
			Kernel_Invocation(argc, argv, Stream[2], 7);
			break;
		case 8:
			Kernel_Invocation(argc, argv, Stream[2], 8);
			break;
		case 9:
			Kernel_Invocation(argc, argv, Stream[2], 9);
			break;
		case 10:
			Kernel_Invocation(argc, argv, Stream[2], 10);
			break;
		case 11:
			Kernel_Invocation(argc, argv, Stream[2], 11);
			break;
		case 12:
			Kernel_Invocation(argc, argv, Stream[2], 12);
			break;
		case 13:
			Kernel_Invocation(argc, argv, Stream[2], 13);
			break;
		case 14:
			Kernel_Invocation(argc, argv, Stream[2], 14);
			break;
		case 15:
			Kernel_Invocation(argc, argv, Stream[2], 15);
			break;
		case 16:
			Kernel_Invocation(argc, argv, Stream[2], 16);
			break;
	}*/
	// D2H Phase
	for (int i = 0; i < NumofStreams; i++)
	{
		switch (S[i])
		{
		case 1:
			D2H(argc, argv, Stream[i], 1);
			break;
		case 2:
			D2H(argc, argv, Stream[i], 2);
			break;
		case 3:
			D2H(argc, argv, Stream[i], 3);
			break;
		case 4:
			D2H(argc, argv, Stream[i], 4);
			break;
		case 5:
			D2H(argc, argv, Stream[i], 5);
			break;
		case 6:
			D2H(argc, argv, Stream[i], 6);
			break;
		case 7:
			D2H(argc, argv, Stream[i], 7);
			break;
		case 8:
			D2H(argc, argv, Stream[i], 8);
			break;
		case 9:
			D2H(argc, argv, Stream[i], 9);
			break;
		case 10:
			D2H(argc, argv, Stream[i], 10);
			break;
		case 11:
			D2H(argc, argv, Stream[i], 11);
			break;
		case 12:
			D2H(argc, argv, Stream[i], 12);
			break;
		case 13:
			D2H(argc, argv, Stream[i], 13);
			break;
		case 14:
			D2H(argc, argv, Stream[i], 14);
			break;
		case 15:
			D2H(argc, argv, Stream[i], 15);
			break;
		case 16:
			D2H(argc, argv, Stream[i], 16);
			break;
		}
	}
	/*switch (S1)
	{
		case 1:
			D2H(argc, argv, Stream[0], 1);
			break;
		case 2:
			D2H(argc, argv, Stream[0], 2);
			break;
		case 3:
			D2H(argc, argv, Stream[0], 3);
			break;
		case 4:
			D2H(argc, argv, Stream[0], 4);
			break;
		case 5:
			D2H(argc, argv, Stream[0], 5);
			break;
		case 6:
			D2H(argc, argv, Stream[0], 6);
			break;
		case 7:
			D2H(argc, argv, Stream[0], 7);
			break;
		case 8:
			D2H(argc, argv, Stream[0], 8);
			break;
		case 9:
			D2H(argc, argv, Stream[0], 9);
			break;
		case 10:
			D2H(argc, argv, Stream[0], 10);
			break;
		case 11:
			D2H(argc, argv, Stream[0], 11);
			break;
		case 12:
			D2H(argc, argv, Stream[0], 12);
			break;
		case 13:
			D2H(argc, argv, Stream[0], 13);
			break;
		case 14:
			D2H(argc, argv, Stream[0], 14);
			break;
		case 15:
			D2H(argc, argv, Stream[0], 15);
			break;
		case 16:
			D2H(argc, argv, Stream[0], 16);
			break;
	}
	switch (S2)
	{
		case 1:
			D2H(argc, argv, Stream[1], 1);
			break;
		case 2:
			D2H(argc, argv, Stream[1], 2);
			break;
		case 3:
			D2H(argc, argv, Stream[1], 3);
			break;
		case 4:
			D2H(argc, argv, Stream[1], 4);
			break;
		case 5:
			D2H(argc, argv, Stream[1], 5);
			break;
		case 6:
			D2H(argc, argv, Stream[1], 6);
			break;
		case 7:
			D2H(argc, argv, Stream[1], 7);
			break;
		case 8:
			D2H(argc, argv, Stream[1], 8);
			break;
		case 9:
			D2H(argc, argv, Stream[1], 9);
			break;
		case 10:
			D2H(argc, argv, Stream[1], 10);
			break;
		case 11:
			D2H(argc, argv, Stream[1], 11);
			break;
		case 12:
			D2H(argc, argv, Stream[1], 12);
			break;
		case 13:
			D2H(argc, argv, Stream[1], 13);
			break;
		case 14:
			D2H(argc, argv, Stream[1], 14);
			break;
		case 15:
			D2H(argc, argv, Stream[1], 15);
			break;
		case 16:
			D2H(argc, argv, Stream[1], 16);
			break;
	}
	switch (S3)
	{
		case 1:
			D2H(argc, argv, Stream[2], 1);
			break;
		case 2:
			D2H(argc, argv, Stream[2], 2);
			break;
		case 3:
			D2H(argc, argv, Stream[2], 3);
			break;
		case 4:
			D2H(argc, argv, Stream[2], 4);
			break;
		case 5:
			D2H(argc, argv, Stream[2], 5);
			break;
		case 6:
			D2H(argc, argv, Stream[2], 6);
			break;
		case 7:
			D2H(argc, argv, Stream[2], 7);
			break;
		case 8:
			D2H(argc, argv, Stream[2], 8);
			break;
		case 9:
			D2H(argc, argv, Stream[2], 9);
			break;
		case 10:
			D2H(argc, argv, Stream[2], 10);
			break;
		case 11:
			D2H(argc, argv, Stream[2], 11);
			break;
		case 12:
			D2H(argc, argv, Stream[2], 12);
			break;
		case 13:
			D2H(argc, argv, Stream[2], 13);
			break;
		case 14:
			D2H(argc, argv, Stream[2], 14);
			break;
		case 15:
			D2H(argc, argv, Stream[2], 15);
			break;
		case 16:
			D2H(argc, argv, Stream[2], 16);
			break;
		}*/
		// Delete Allocated Space
	for (int i = 0; i < NumofStreams; i++)
	{
		switch (S[i])
		{
		case 1:
			CL_Mem(argc, argv, Stream[i], 1);
			break;
		case 2:
			CL_Mem(argc, argv, Stream[i], 2);
			break;
		case 3:
			CL_Mem(argc, argv, Stream[i], 3);
			break;
		case 4:
			CL_Mem(argc, argv, Stream[i], 4);
			break;
		case 5:
			CL_Mem(argc, argv, Stream[i], 5);
			break;
		case 6:
			CL_Mem(argc, argv, Stream[i], 6);
			break;
		case 7:
			CL_Mem(argc, argv, Stream[i], 7);
			break;
		case 8:
			CL_Mem(argc, argv, Stream[i], 8);
			break;
		case 9:
			CL_Mem(argc, argv, Stream[i], 9);
			break;
		case 10:
			CL_Mem(argc, argv, Stream[i], 10);
			break;
		case 11:
			CL_Mem(argc, argv, Stream[i], 11);
			break;
		case 12:
			CL_Mem(argc, argv, Stream[i], 12);
			break;
		case 13:
			CL_Mem(argc, argv, Stream[i], 13);
			break;
		case 14:
			CL_Mem(argc, argv, Stream[i], 14);
			break;
		case 15:
			CL_Mem(argc, argv, Stream[i], 15);
			break;
		case 16:
			CL_Mem(argc, argv, Stream[i], 16);
			break;

		}
	}
	/*switch (S1)
	{
		case 1:
			CL_Mem(argc, argv, Stream[0], 1);
			break;
		case 2:
			CL_Mem(argc, argv, Stream[0], 2);
			break;
		case 3:
			CL_Mem(argc, argv, Stream[0], 3);
			break;
		case 4:
			CL_Mem(argc, argv, Stream[0], 4);
			break;
		case 5:
			CL_Mem(argc, argv, Stream[0], 5);
			break;
		case 6:
			CL_Mem(argc, argv, Stream[0], 6);
			break;
		case 7:
			CL_Mem(argc, argv, Stream[0], 7);
			break;
		case 8:
			CL_Mem(argc, argv, Stream[0], 8);
			break;
		case 9:
			CL_Mem(argc, argv, Stream[0], 9);
			break;
		case 10:
			CL_Mem(argc, argv, Stream[0], 10);
			break;
		case 11:
			CL_Mem(argc, argv, Stream[0], 11);
			break;
		case 12:
			CL_Mem(argc, argv, Stream[0], 12);
			break;
		case 13:
			CL_Mem(argc, argv, Stream[0], 13);
			break;
		case 14:
			CL_Mem(argc, argv, Stream[0], 14);
			break;
		case 15:
			CL_Mem(argc, argv, Stream[0], 15);
			break;
		case 16:
			CL_Mem(argc, argv, Stream[0], 16);
			break;

	}
	switch (S2)
	{
		case 1:
			CL_Mem(argc, argv, Stream[1], 1);
			break;
		case 2:
			CL_Mem(argc, argv, Stream[1], 2);
			break;
		case 3:
			CL_Mem(argc, argv, Stream[1], 3);
			break;
		case 4:
			CL_Mem(argc, argv, Stream[1], 4);
			break;
		case 5:
			CL_Mem(argc, argv, Stream[1], 5);
			break;
		case 6:
			CL_Mem(argc, argv, Stream[1], 6);
			break;
		case 7:
			CL_Mem(argc, argv, Stream[1], 7);
			break;
		case 8:
			CL_Mem(argc, argv, Stream[1], 8);
			break;
		case 9:
			CL_Mem(argc, argv, Stream[1], 9);
			break;
		case 10:
			CL_Mem(argc, argv, Stream[1], 10);
			break;
		case 11:
			CL_Mem(argc, argv, Stream[1], 11);
			break;
		case 12:
			CL_Mem(argc, argv, Stream[1], 12);
			break;
		case 13:
			CL_Mem(argc, argv, Stream[1], 13);
			break;
		case 14:
			CL_Mem(argc, argv, Stream[1], 14);
			break;
		case 15:
			CL_Mem(argc, argv, Stream[1], 15);
			break;
		case 16:
			CL_Mem(argc, argv, Stream[1], 16);
			break;
	}
	switch (S3)
	{
		case 1:
			CL_Mem(argc, argv, Stream[2], 1);
			break;
		case 2:
			CL_Mem(argc, argv, Stream[2], 2);
			break;
		case 3:
			CL_Mem(argc, argv, Stream[2], 3);
			break;
		case 4:
			CL_Mem(argc, argv, Stream[2], 4);
			break;
		case 5:
			CL_Mem(argc, argv, Stream[2], 5);
			break;
		case 6:
			CL_Mem(argc, argv, Stream[2], 6);
			break;
		case 7:
			CL_Mem(argc, argv, Stream[2], 7);
			break;
		case 8:
			CL_Mem(argc, argv, Stream[2], 8);
			break;
		case 9:
			CL_Mem(argc, argv, Stream[2], 9);
			break;
		case 10:
			CL_Mem(argc, argv, Stream[2], 10);
			break;
		case 11:
			CL_Mem(argc, argv, Stream[2], 11);
			break;
		case 12:
			CL_Mem(argc, argv, Stream[2], 12);
			break;
		case 13:
			CL_Mem(argc, argv, Stream[2], 13);
			break;
		case 14:
			CL_Mem(argc, argv, Stream[2], 14);
			break;
		case 15:
			CL_Mem(argc, argv, Stream[2], 15);
			break;
		case 16:
			CL_Mem(argc, argv, Stream[2], 16);
			break;
	}*/
}

void nStream_Benchmarks(int argc, char** argv)
{
	
	int Mode = 0;
	int SI = 0;
	int EI = 0;
	if (checkCmdLineFlag(argc, (const char **)argv, "Mode"))
	{
		Mode = getCmdLineArgumentInt(argc, (const char **)argv, "Mode");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "SI"))
	{
		SI = getCmdLineArgumentInt(argc, (const char **)argv, "SI");
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "EI"))
	{
		EI = getCmdLineArgumentInt(argc, (const char **)argv, "EI");
	}
	int *S = new int[Mode];
	cudaStream_t Stream[6];
	cudaStreamCreate(&Stream[0]);
	cudaStreamCreate(&Stream[1]);
	cudaStreamCreate(&Stream[2]);
	cudaStreamCreate(&Stream[3]);
	cudaStreamCreate(&Stream[4]);
	cudaStreamCreate(&Stream[5]);
	//Manual
	switch (Mode)
	{
	case 0:
		if (checkCmdLineFlag(argc, (const char **)argv, "S1"))
		{
			S[0] = getCmdLineArgumentInt(argc, (const char **)argv, "S1");
		}
		if (checkCmdLineFlag(argc, (const char **)argv, "S2"))
		{
			S[1] = getCmdLineArgumentInt(argc, (const char **)argv, "S2");
		}
		if (checkCmdLineFlag(argc, (const char **)argv, "S3"))
		{
			S[2] = getCmdLineArgumentInt(argc, (const char **)argv, "S3");
		}
		printf("\n************************* S1=%i, S2=%i, S3=%i *************************\n", S[0], S[1], S[2]);
		nStreams_Invocation(argc, argv, S, Mode, Stream);
		break;
	case 1:
		break;
	case 2:
		for (int i = SI; i <= EI; i++)
		{
			for (int j = SI; j <= EI; j++)
			{
				if (i != j)
				{
					S[0] = i;
					S[1] = j;
					printf("\n************************* S1=%i, S2=%i *************************\n", i, j);
					nStreams_Invocation(argc, argv, S, Mode, Stream);
					cudaDeviceSynchronize();
				}
			}
		}
		break;
	case 3:
		for (int i = SI; i <= EI; i++)
		{
			for (int j = SI; j <= EI; j++)
			{
				for (int k = SI; k <= EI; k++)
				{
					if ((i != j) && (i != k) && (j != k))
					{
						S[0] = i;
						S[1] = j;
						S[2] = k;
						printf("\n************************* S1=%i, S2=%i, S3=%i *************************\n", i, j, k);
						nStreams_Invocation(argc, argv, S, Mode, Stream);
						cudaDeviceSynchronize();
					}
				}
			}
		}
		break;
	case 4:
		for (int i = SI; i <= EI; i++)
		{
			for (int j = SI; j <= EI; j++)
			{
				for (int k = SI; k <= EI; k++)
				{
					for (int x = SI; x <= EI; x++)
					{
						if ((i != j) && (i != k) && (i != x) && (j != k) && (j != x) && (k != x))
						{
							S[0] = i;
							S[1] = j;
							S[2] = k;
							S[3] = x;
							printf("\n************************* S1=%i, S2=%i, S3=%i, S4=%i *************************\n", i, j, k, x);
							nStreams_Invocation(argc, argv, S, Mode, Stream);
							cudaDeviceSynchronize();
						}
					}
				}

			}
		}
		break;
	case 5:
		for (int i = SI; i <= EI; i++)
		{
			for (int j = SI; j <= EI; j++)
			{
				for (int k = SI; k <= EI; k++)
				{
					for (int x = SI; x <= EI; x++)
					{
						for (int y = SI; y <= EI; y++)
						{
							if ((i != j) && (i != k) && (i != x) && (i != y) && (j != k) && (j != x) && (j != y) && (k != x) && (k != y) && (x != y))
							{
								S[0] = i;
								S[1] = j;
								S[2] = k;
								S[3] = x;
								S[4] = y;
								printf("\n************************* S1=%i, S2=%i, S3=%i, S4=%i, S5=%i *************************\n", i, j, k, x, y);
								nStreams_Invocation(argc, argv, S, Mode, Stream);
								cudaDeviceSynchronize();
							}
						}
					}
				}
			}
		}
		break;
	case 6:
		for (int i = SI; i <= EI; i++)
		{
			for (int j = SI; j <= EI; j++)
			{
				for (int k = SI; k <= EI; k++)
				{
					for (int x = SI; x <= EI; x++)
					{
						for (int y = SI; y <= EI; y++)
						{
							for (int z = SI; z <= EI; z++)
							{
								if ((i != j) && (i != k) && (i != x) && (i != y) && (i != z) && (j != k) && (j != x) && (j != y) && (j != z) && (k != x) && (k != y) && (k != z) && (x != y) && (x != z) && (y != z))
								{
									S[0] = i;
									S[1] = j;
									S[2] = k;
									S[3] = x;
									S[4] = y;
									S[5] = z;
									printf("\n************************* S1=%i, S2=%i, S3=%i, S4=%i, S5=%i, S6=%i *************************\n", i, j, k, x, y, z);
									nStreams_Invocation(argc, argv, S, Mode, Stream);
									cudaDeviceSynchronize();
								}
							}
						}
					}
				}
			}
		}
		break;
	default:
		break;
	}
	/*if (Mode == 0)
	{

	}
	if (Mode == 3)
	{

	}*/
	cudaStreamDestroy(Stream[0]);
	cudaStreamDestroy(Stream[1]);
	cudaStreamDestroy(Stream[2]);
	cudaStreamDestroy(Stream[3]);
	cudaStreamDestroy(Stream[4]);
	cudaStreamDestroy(Stream[5]);
}

int main(int argc, char **argv)
{
	cudaDeviceReset();
	nStream_Benchmarks(argc, argv);
	exit(EXIT_SUCCESS);
}
