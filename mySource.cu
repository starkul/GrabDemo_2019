#include <stdio.h>  
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  

#include <iostream>
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include "device_functions.h"
//#include "sm_20_atomic_functions.h"
using namespace std;
// �������
__global__ void TwoPoint_Correction(unsigned short *gpu_img, int Length, int Width, float *pTP_Gain, float *pTP_Bias)
{
	for (int i = 0; i < 768; i++)
		for (int j = 0; j < 1024; j++)
			gpu_img[j + i * 1024] = unsigned short(pTP_Gain[j + i * 1024] * gpu_img[j + i * 1024] + pTP_Bias[j + i * 1024]);

}

// ä�����
__global__ void Blind_On_Correction(unsigned short *gpu_img, int Length, int Width, unsigned short *pBlind_Ram)
{
	float mean = 0; int count = 0;
	for (int i = 0; i < 768; i++)
		for (int j = 0; j < 1024; j++)
			if (pBlind_Ram[j + i * 1024] == 1) //��ΪäԪ���򽫸õ��Ϊ��Χ���ƽ��
			{
				mean = 0; count = 0;//��Ե��������
				if (i > 0)
				{
					count++;
					mean += gpu_img[j + (i-1) * 1024];
				}
				if (i < 768)
				{
					count++;
					mean += gpu_img[j + (i + 1) * 1024];
				}
				if (j > 0)
				{
					count++;
					mean += gpu_img[(j - 1) + i * 1024];
				}
				if (j < 1024)
				{
					count++;
					mean += gpu_img[(j + 1) + i * 1024];
				}
				gpu_img[j + i * 1024] = unsigned short(mean/count);
			}

}
//---------ֱ��ͼ����----------
__global__ void Histogram_Enhancement(unsigned short *gpu_img)
{
		//----------------------���������------------------------
		unsigned short Histogram_Count[65536] = { 0 };   //ֱ��ͼ��ǿ��
		float pHistogram_Enhancement[65536] = {};

		for (int i = 0; i < 768; i++)
			for (int j = 0; j < 1024; j++)
			{
				Histogram_Count[gpu_img[j + i * 1024]] = Histogram_Count[gpu_img[j + i * 1024]] + 1;
				//int k = Histogram_Count[gpu_img[j + i * 1024]];
				//k = 1;
			}

		//------------
		float sum = 0;

		//-----------����ֱ��ͼ��--------------
		for (int i = 0; i < 65536; i++)
		{
			sum = sum + Histogram_Count[i];
			pHistogram_Enhancement[i] = sum / 768 / 1024;
		}


		for (int i = 0; i < 768; i++)
			for (int j = 0; j < 1024; j++)
			{
				unsigned short k = gpu_img[j + i * 1024];
				gpu_img[j + i * 1024] =  65535 * pHistogram_Enhancement[k];
				//gpu_img[j + i * 1024] =  //pHistogram[0]*gpu_img[j + i * 1024];
				 //gpu_img[j + i * 1024] =  pHistogram[30001]*65535;
				//int k = pHistogram[gpu_img[j * 1024]];
			}
	
	//for (int i = 0; i < 768; i++)
	//	for (int j = 0; j < 1024; j++)
	//	{
	//		gpu_img[j + i * 1024] = 0;
	//	}

}

// ��д��ҪGPU����Ĺ��ܺ���
extern "C"
cudaError_t Image_Solution(unsigned short *Image,int Width,int Length,float *pTP_Gain, float *pTP_Bias,int TP_On,int Blind_On, unsigned short *pBlind_Ram,int Histogram_On)  //  pTP_Gain,pTP_Biasָ�������������----TP_On��ʾ�Ƿ����������
{
	int size = 1; //���ú��߳���
	unsigned short *dev_img = 0;
	float *dev_pTP_Gain = 0, *dev_pTP_Bias = 0;
	unsigned short *dev_pBlind_Ram = 0;
	cudaError_t cudaStatus;

	//size = pHistogram[65535];
	//size = pHistogram[20000];
	//size = pHistogram[10000];
	// Choose which GPU to run on, change this on a multi-GPU system.  
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//---------------------------ͼ���ڴ濪��---------------------------------------------------
	// ���ٴ��ͼ����ڴ�    .  
	cudaStatus = cudaMalloc((void**)&dev_img, Length * Width * sizeof(unsigned short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// ��ͼ���ڴ�浽CUDA�ڴ���  
	cudaStatus = cudaMemcpy(dev_img, Image, Length * Width * sizeof(unsigned short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// Launch a kernel on the GPU with one thread for each element.  
	//�˴�Ϊ���� GPU�Ĳ��������������ַ
	////�˺����ĵ��ã�ע��<<<1,1>>>����һ��1�������̸߳���ֻ��һ���߳̿飻�ڶ���1������һ���߳̿���ֻ��һ���̡߳�
	//Solution_Kernel << <1, size >> >(dev_img, Length, Width, pTP_Gain, pTP_Bias);

	//----------------------------��������н������ڴ濪��---------------------------------------
	if (TP_On > 0)
	{
		cudaStatus = cudaMalloc((void**)&dev_pTP_Gain, Length * Width * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Gain cudaMalloc failed!");
			goto Error;
		}
		// ��ͼ���ڴ�浽CUDA�ڴ���  
		cudaStatus = cudaMemcpy(dev_pTP_Gain, pTP_Gain, Length * Width * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Gain cudaMemcpy failed!");
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&dev_pTP_Bias, Length * Width * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Bias cudaMalloc failed!");
			goto Error;
		}
		// ��ͼ���ڴ�浽CUDA�ڴ���  
		cudaStatus = cudaMemcpy(dev_pTP_Bias, pTP_Gain, Length * Width * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Bias cudaMemcpy failed!");
			goto Error;
		}
		TwoPoint_Correction << <1, size >> >(dev_img, Length, Width, dev_pTP_Gain, dev_pTP_Bias);
	}

	//____________________________________________________________________________

	//--------------------------------äԪ����ʵ��--------------------------------
	if (Blind_On > 0)
	{
		cudaStatus = cudaMalloc((void**)&dev_pBlind_Ram, Length * Width * sizeof(unsigned short));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Gain cudaMalloc failed!");
			goto Error;
		}
		// ��ͼ���ڴ�浽CUDA�ڴ���  
		cudaStatus = cudaMemcpy(dev_pBlind_Ram, pBlind_Ram, Length * Width * sizeof(unsigned short), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Gain cudaMemcpy failed!");
			goto Error;
		}

		Blind_On_Correction << <1, size >> >(dev_img, Length, Width, dev_pBlind_Ram);
	}

	//--------------------------------ֱ��ͼ��ǿʵ��--------------------------------
	if (Histogram_On > 0)
	{
		Histogram_Enhancement << <1, 1 >> >(dev_img);
	}

	//---------------------------------------------------------------------------------------
	// Check for any errors launching the kernel  
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns  
	// any errors encountered during the launch.  
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.  
	cudaStatus = cudaMemcpy(Image, dev_img, Length * Width * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//Ӧ��ÿ�ζ���Ҫ����ڴ��
	//cudaFree(dev_img);
Error:
	cudaFree(dev_img);
	cudaFree(dev_pTP_Gain); cudaFree(dev_pTP_Bias);
	cudaFree(dev_pBlind_Ram);
	return cudaStatus;
}

//------------------------------------------------------------------------------
//                      GPU ��������
//------------------------------------------------------------------------------

//----------------------ֱ��ͼ��ǿ----------------------------
__global__ void Create_Histogram1(unsigned short *gpu_img, unsigned int * dev_Histogram)
{
	//ͳ��ֱ��ͼ
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//dev_Histogram[gpu_img[i]] += 1;
	unsigned short value =  gpu_img[gpu_img[i]];  //  ??? ͳ��ֱ��ͼֻ��Ҫȡ��ǰ�ĻҶ�ֵ�ɣ�
	//unsigned short value = gpu_img[i];
	//ԭ�Ӳ�������������
	atomicAdd(&dev_Histogram[value], 1);
}

__global__ void Clear_Histogram(unsigned int* dev_Histogram)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	dev_Histogram[i] = 0;
}

__global__ void Create_Histogram2(unsigned short* dev_img, unsigned int* dev_Histogram, int Height, int Width)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned short count = dev_Histogram_Float[dev_img[i]];
	dev_img[i] = double(dev_Histogram[dev_img[i]]) / Height / Width * 65535;

	//double tmp = double(dev_Histogram[dev_img[i]]) / 768 / 1024;
	//dev_img[i] =  tmp * 65535 ;

	//dev_img[i] = unsigned short(dev_Histogram[dev_img[i]] / 16 * 65535 / 49152 + dev_Histogram[dev_img[i]] % 16);
	//dev_img[i] = unsigned short(dev_Histogram[dev_img[i]] / Height * 65535 / Width + dev_Histogram[dev_img[i]] % Height * 65535 / Width);
}
__global__ void Create_Histogram3(unsigned short *gpu_img, float* dev_Histogram_float)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//gpu_img[i] = dev_Histogram[gpu_img[i]] / 1024 / 768 ;
	gpu_img[i] =  65535 * dev_Histogram_float[gpu_img[i]];
	//gpu_img[i] = 0;
}

//------------------------��������------------------------
__global__ void Linear_1(unsigned short *gpu_img, unsigned short* dev_Histogram)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;


	if (gpu_img[i] > dev_Histogram[0])  //���ֵ�洢
		dev_Histogram[0] = gpu_img[i];
	if (gpu_img[i] < dev_Histogram[1])  //��Сֵ�洢
		dev_Histogram[1] = gpu_img[i];


		
/*
//����û����
for (int i=0;i<1024*768;i++)
if (gpu_img[i] > dev_Histogram[0])  //���ֵ�洢
dev_Histogram[0] = gpu_img[i];
*/

}
__global__ void Find_Max(unsigned short *g_idata, unsigned short *g_odata, unsigned short *gate_index)
{
	__shared__ int sdata[1024];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned int i = blockIdx.x * gate_index[1] + threadIdx.x;

	if(tid < *gate_index)
		sdata[tid] = g_idata[i];
	else 
		sdata[tid] = 0;

	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			if (sdata[tid] < sdata[tid + s])
				sdata[tid] = sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}
__global__ void Find_Min(unsigned short *g_idata, unsigned short *g_odata, unsigned short *gate_index)
{
	__shared__ int sdata[1024];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned int i = blockIdx.x * 768 + threadIdx.x;

	if (tid < *gate_index)
		sdata[tid] = g_idata[i];
	else
		sdata[tid] = 65535;

	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			if (sdata[tid] > sdata[tid + s])
				sdata[tid] = sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}
__global__ void Linear_2(unsigned short *gpu_img, unsigned short* dev_Histogram)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	gpu_img[i] = unsigned short (float(65535 / (dev_Histogram[0] - dev_Histogram[1]))*(gpu_img[i] - dev_Histogram[1]));  //��������
	//gpu_img[i] = unsigned short(65535 *(gpu_img[i] - dev_Histogram[1]) / (dev_Histogram[0] - dev_Histogram[1]));  //��������
}
//--------------------------------ֱ��ͼ��ǿʵ��--------------------------------
extern "C"
cudaError_t GPU_Histogram_Enhancement(unsigned short *Image, unsigned int *Histogram,float *Histogram_Float,unsigned short *dev_img, unsigned int* dev_Histogram, float* dev_Histogram_Float, int Height, int Width)  //  pTP_Gain,pTP_Biasָ�������������----TP_On��ʾ�Ƿ����������
{

	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dev_img, Image, Height * Width * sizeof(unsigned short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	unsigned short *dev_index = 0;
/*	cudaStatus = cudaMalloc((void**)&dev_index, 2 * sizeof(unsigned short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "p_index cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(dev_index, &Width, 2 * sizeof(unsigned short), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_index+1, &Height, 2 * sizeof(unsigned short), cudaMemcpyHostToDevice);
*/
	unsigned int tmp[65536];
	unsigned int *ptmp = &tmp[0];
	unsigned short tmp_short[300000];
	unsigned short tmp_float[65536];
	tmp[0] = 0;

/*	//����ͼ��Image�Ƿ��롣
	for (int i = 0; i < Height * Width; i++)
	{
		if (tmp[0] < Image[i])
			tmp[0] = Image[i];
		if (tmp[1] > Image[i])
			tmp[1] = Image[i];
	}
*/
/*	// ����ͼ���Ƿ���GPU
	cudaMemcpy(Image, dev_img, 2 * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	*/

	//--------------------------------ֱ��ͼ��ǿʵ��--------------------------------
	//dim3 threadsPerBlock(1024);
	//dim3 numBlocks(768);
	dim3 threadsPerBlock(Width);
	dim3 numBlocks(Height);
	// ֱ��ͼ����

	Clear_Histogram << <256, 256 >> > (dev_Histogram);
	cudaStatus = cudaMemcpy(ptmp, dev_Histogram, 65536 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(tmp_short, dev_img, 65536 * sizeof(unsigned short), cudaMemcpyDeviceToHost);

	Create_Histogram1 << <numBlocks, threadsPerBlock >> > (dev_img, dev_Histogram);  //1.ͳ�Ƹ����Ҷ�ֵ������
	cudaStatus = cudaMemcpy(ptmp, dev_Histogram, 65536 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(Histogram, dev_Histogram, 65536 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	long tValue = Height * Width;
	//Histogram[0]�Ѿ���ֵ��
	//�۲�ptmp���ݣ����ͳ���Ƿ���ȷ��
	for (int i = 1; i < 65536; i++)
	{
		Histogram[i] = Histogram[i - 1] + Histogram[i];
//		tmp_float[i] = Histogram[i] / tValue;
		tmp[i] = Histogram[i];
	}
	cudaStatus = cudaMemcpy(dev_Histogram, Histogram, 65536 * sizeof(unsigned int), cudaMemcpyHostToDevice);
	//��Ҫ���ݷֱ��ʵ�����ǿ����
	Create_Histogram2 << <numBlocks, threadsPerBlock >> > (dev_img, dev_Histogram, Height, Width);
	cudaStatus = cudaMemcpy(tmp_short, dev_img,300000 * sizeof(unsigned short), cudaMemcpyDeviceToHost);
/*	Histogram_Float[0] = Histogram[0] / tValue;
	for (int i = 1; i < 65536; i++)
	{
		Histogram_Float[i] = Histogram_Float[i - 1] + Histogram[i] / tValue;
		tmp_float[i] = Histogram_Float[i];
	}

	cudaStatus = cudaMemcpy(dev_Histogram_Float, Histogram_Float, 65536 * sizeof(float), cudaMemcpyHostToDevice);
*/
	unsigned short max = 0;
	for (int i = 0; i < 300000; i++)
		if (max < tmp_short[i])
			max = tmp_short[i];

	//cudaMemcpy(ptmp, dev_Histogram, 65536 * sizeof(long), cudaMemcpyDeviceToHost);
	//Create_Histogram3 << <numBlocks, threadsPerBlock >> > (dev_img, dev_Histogram_float);
	//Histogram_Enhancement << <1, 1 >> >(dev_img);  //����

	//-------------��������------------------

	//�����ֵ��Сֵ
/*	cudaMemcpy(dev_Histogram, ptmp, 2 * sizeof(unsigned short), cudaMemcpyHostToDevice); //���±�ֵ
	//Linear_1 << <numBlocks, threadsPerBlock >> > (dev_img, dev_Histogram);
	Find_Max <<< Height, Width >>> (dev_img, dev_Histogram,&dev_index[0]);  //0 --- width  \\  1------height
	Find_Max <<< 1, 1024 >>> (dev_Histogram, dev_Histogram, &dev_index[1]);
	cudaMemcpy(ptmp, dev_Histogram, 1 * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	Find_Min << < Height, Width >> > (dev_img, dev_Histogram,&dev_index[0]);
	Find_Min << < 1, 1024 >> > (dev_Histogram, dev_Histogram,&dev_index[1]);
	cudaMemcpy(ptmp+1, dev_Histogram+1, 1 * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_Histogram, ptmp, 2 * sizeof(unsigned short), cudaMemcpyHostToDevice); //���±�ֵ

	Linear_2 << <numBlocks, threadsPerBlock >> > (dev_img, dev_Histogram);
	*/
	//---------------------------------------------------------------------------------------
	// Check for any errors launching the kernel  
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns  
	// any errors encountered during the launch.  
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.  
	cudaStatus = cudaMemcpy(Image, dev_img, Height * Width * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	//Ӧ��ÿ�ζ���Ҫ����ڴ��
	//cudaFree(dev_img);

	return cudaStatus;
}

//*****************************************************************************************

//----------------------�������������-------------------------
__global__ void GPU_TwoPoint_Helper(unsigned short *gpu_img, float* dev_pTP_Gain, float* dev_pTP_Bias)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	gpu_img[i] = unsigned short(dev_pTP_Gain[i] * gpu_img[i] + dev_pTP_Bias[i]);
}
//-----------------------��������ӿں���--------------------------
extern "C"
cudaError_t GPU_TwoPoint_Correction(unsigned short *Image, unsigned short *dev_img, float* dev_pTP_Gain, float* dev_pTP_Bias, int Height, int Width)  //  pTP_Gain,pTP_Biasָ�������������----TP_On��ʾ�Ƿ����������
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dev_img, Image, Height * Width * sizeof(unsigned short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	//--------------------------------�����������--------------------------------
	dim3 threadsPerBlock(Width);
	dim3 numBlocks(Height);
	GPU_TwoPoint_Helper << <numBlocks, threadsPerBlock >> > (dev_img, dev_pTP_Gain, dev_pTP_Bias);
	//TwoPoint_Correction << <1, 1 >> >(dev_img, Height, Width, dev_pTP_Gain, dev_pTP_Bias);
	//---------------------------------------------------------------------------------------
	// Check for any errors launching the kernel  
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns  
	// any errors encountered during the launch.  
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.  
	cudaStatus = cudaMemcpy(Image, dev_img, Height * Width * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	//Ӧ��ÿ�ζ���Ҫ����ڴ��
	//cudaFree(dev_img);

	return cudaStatus;
}

//*****************************************************************************************

//----------------------äԪ����������-------------------------
__global__ void GPU_Blind__Helper(unsigned short *gpu_img, unsigned short* dev_pBlind_Ram, int Height, int Width)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	if (dev_pBlind_Ram[i*blockDim.x+j] == 1) //����1��ʾΪäԪ
	{
		//��äԪ�滻Ϊ��Χ�ĸ����ص�ƽ��
		unsigned short n = 0; double sum = 0;
		if (j > 0)
		{
			sum += gpu_img[i*blockDim.x + j - 1]; n++;
		}
		if (j < Width)
		{
			sum += gpu_img[i*blockDim.x + j + 1]; n++;
		}
		if (i > 0)
		{
			sum += gpu_img[(i-1)*blockDim.x + j + 1]; n++;
		}
		if (i < Height)
		{
			sum += gpu_img[(i+1)*blockDim.x + j + 1]; n++;
		}
		if (n != 0)
			gpu_img[i*blockDim.x + j] = sum / n;
	}
}
//-----------------------äԪ����--------------------------
extern "C"
cudaError_t GPU_Blind_Correction(unsigned short *Image, unsigned short *dev_img, unsigned short *dev_pBlind_Ram, int Height, int Width)  //  pTP_Gain,pTP_Biasָ�������������----TP_On��ʾ�Ƿ����������
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dev_img, Image, Height * Width * sizeof(unsigned short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	//--------------------------------�����������--------------------------------
	dim3 threadsPerBlock(Width);
	dim3 numBlocks(Height);
	GPU_Blind__Helper << <numBlocks, threadsPerBlock >> > (dev_img, dev_pBlind_Ram, Height, Width);
	//Blind_On_Correction << <1, 1 >> > (dev_img, Height, Width, dev_pBlind_Ram);
	
/*	// ���Դ���
	unsigned short *ptmp = new unsigned short[768*1024];
	cudaMemcpy(ptmp, dev_pBlind_Ram, 768 * 1024 * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	unsigned short kk = ptmp[700 * 1024 + 980];
	kk = ptmp[700 * 1024 + 981];
*/
	//---------------------------------------------------------------------------------------
	// Check for any errors launching the kernel  
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns  
	// any errors encountered during the launch.  
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.  
	cudaStatus = cudaMemcpy(Image, dev_img, Height * Width * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}

	//Ӧ��ÿ�ζ���Ҫ����ڴ��
	//cudaFree(dev_img);

	return cudaStatus;
}

