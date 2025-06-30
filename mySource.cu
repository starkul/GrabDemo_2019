#include <stdio.h>  
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  

#include <iostream>
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include "device_functions.h"
//#include "sm_20_atomic_functions.h"
using namespace std;
// 两点矫正
__global__ void TwoPoint_Correction(unsigned short *gpu_img, int Length, int Width, float *pTP_Gain, float *pTP_Bias)
{
	for (int i = 0; i < 768; i++)
		for (int j = 0; j < 1024; j++)
			gpu_img[j + i * 1024] = unsigned short(pTP_Gain[j + i * 1024] * gpu_img[j + i * 1024] + pTP_Bias[j + i * 1024]);

}

// 盲点矫正
__global__ void Blind_On_Correction(unsigned short *gpu_img, int Length, int Width, unsigned short *pBlind_Ram)
{
	float mean = 0; int count = 0;
	for (int i = 0; i < 768; i++)
		for (int j = 0; j < 1024; j++)
			if (pBlind_Ram[j + i * 1024] == 1) //若为盲元，则将该点变为周围点的平均
			{
				mean = 0; count = 0;//边缘条件设置
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
//---------直方图均衡----------
__global__ void Histogram_Enhancement(unsigned short *gpu_img)
{
		//----------------------构建均衡表------------------------
		unsigned short Histogram_Count[65536] = { 0 };   //直方图增强表
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

		//-----------生成直方图表--------------
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

// 编写需要GPU处理的功能函数
extern "C"
cudaError_t Image_Solution(unsigned short *Image,int Width,int Length,float *pTP_Gain, float *pTP_Bias,int TP_On,int Blind_On, unsigned short *pBlind_Ram,int Histogram_On)  //  pTP_Gain,pTP_Bias指向两点矫正参数----TP_On表示是否开启两点矫正
{
	int size = 1; //调用核线程数
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

	//---------------------------图像内存开辟---------------------------------------------------
	// 开辟存放图像的内存    .  
	cudaStatus = cudaMalloc((void**)&dev_img, Length * Width * sizeof(unsigned short));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// 将图像内存存到CUDA内存中  
	cudaStatus = cudaMemcpy(dev_img, Image, Length * Width * sizeof(unsigned short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	// Launch a kernel on the GPU with one thread for each element.  
	//此处为设置 GPU的操作函数，传入地址
	////核函数的调用，注意<<<1,1>>>，第一个1，代表线程格里只有一个线程块；第二个1，代表一个线程块里只有一个线程。
	//Solution_Kernel << <1, size >> >(dev_img, Length, Width, pTP_Gain, pTP_Bias);

	//----------------------------两点矫正中矫正表内存开辟---------------------------------------
	if (TP_On > 0)
	{
		cudaStatus = cudaMalloc((void**)&dev_pTP_Gain, Length * Width * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Gain cudaMalloc failed!");
			goto Error;
		}
		// 将图像内存存到CUDA内存中  
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
		// 将图像内存存到CUDA内存中  
		cudaStatus = cudaMemcpy(dev_pTP_Bias, pTP_Gain, Length * Width * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Bias cudaMemcpy failed!");
			goto Error;
		}
		TwoPoint_Correction << <1, size >> >(dev_img, Length, Width, dev_pTP_Gain, dev_pTP_Bias);
	}

	//____________________________________________________________________________

	//--------------------------------盲元矫正实现--------------------------------
	if (Blind_On > 0)
	{
		cudaStatus = cudaMalloc((void**)&dev_pBlind_Ram, Length * Width * sizeof(unsigned short));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Gain cudaMalloc failed!");
			goto Error;
		}
		// 将图像内存存到CUDA内存中  
		cudaStatus = cudaMemcpy(dev_pBlind_Ram, pBlind_Ram, Length * Width * sizeof(unsigned short), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Gain cudaMemcpy failed!");
			goto Error;
		}

		Blind_On_Correction << <1, size >> >(dev_img, Length, Width, dev_pBlind_Ram);
	}

	//--------------------------------直方图增强实现--------------------------------
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

	//应该每次都需要清空内存吧
	//cudaFree(dev_img);
Error:
	cudaFree(dev_img);
	cudaFree(dev_pTP_Gain); cudaFree(dev_pTP_Bias);
	cudaFree(dev_pBlind_Ram);
	return cudaStatus;
}

//------------------------------------------------------------------------------
//                      GPU 并行运算
//------------------------------------------------------------------------------

//----------------------直方图增强----------------------------
__global__ void Create_Histogram1(unsigned short *gpu_img, unsigned int * dev_Histogram)
{
	//统计直方图
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//dev_Histogram[gpu_img[i]] += 1;
	unsigned short value =  gpu_img[gpu_img[i]];  //  ??? 统计直方图只需要取当前的灰度值吧？
	//unsigned short value = gpu_img[i];
	//原子操作，否则会出错。
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

//------------------------线性拉伸------------------------
__global__ void Linear_1(unsigned short *gpu_img, unsigned short* dev_Histogram)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;


	if (gpu_img[i] > dev_Histogram[0])  //最大值存储
		dev_Histogram[0] = gpu_img[i];
	if (gpu_img[i] < dev_Histogram[1])  //最小值存储
		dev_Histogram[1] = gpu_img[i];


		
/*
//串行没问题
for (int i=0;i<1024*768;i++)
if (gpu_img[i] > dev_Histogram[0])  //最大值存储
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

	gpu_img[i] = unsigned short (float(65535 / (dev_Histogram[0] - dev_Histogram[1]))*(gpu_img[i] - dev_Histogram[1]));  //线性拉伸
	//gpu_img[i] = unsigned short(65535 *(gpu_img[i] - dev_Histogram[1]) / (dev_Histogram[0] - dev_Histogram[1]));  //线性拉伸
}
//--------------------------------直方图增强实现--------------------------------
extern "C"
cudaError_t GPU_Histogram_Enhancement(unsigned short *Image, unsigned int *Histogram,float *Histogram_Float,unsigned short *dev_img, unsigned int* dev_Histogram, float* dev_Histogram_Float, int Height, int Width)  //  pTP_Gain,pTP_Bias指向两点矫正参数----TP_On表示是否开启两点矫正
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

/*	//测试图像Image是否传入。
	for (int i = 0; i < Height * Width; i++)
	{
		if (tmp[0] < Image[i])
			tmp[0] = Image[i];
		if (tmp[1] > Image[i])
			tmp[1] = Image[i];
	}
*/
/*	// 测试图像是否传入GPU
	cudaMemcpy(Image, dev_img, 2 * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	*/

	//--------------------------------直方图增强实现--------------------------------
	//dim3 threadsPerBlock(1024);
	//dim3 numBlocks(768);
	dim3 threadsPerBlock(Width);
	dim3 numBlocks(Height);
	// 直方图均衡

	Clear_Histogram << <256, 256 >> > (dev_Histogram);
	cudaStatus = cudaMemcpy(ptmp, dev_Histogram, 65536 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(tmp_short, dev_img, 65536 * sizeof(unsigned short), cudaMemcpyDeviceToHost);

	Create_Histogram1 << <numBlocks, threadsPerBlock >> > (dev_img, dev_Histogram);  //1.统计各个灰度值的数量
	cudaStatus = cudaMemcpy(ptmp, dev_Histogram, 65536 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(Histogram, dev_Histogram, 65536 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	long tValue = Height * Width;
	//Histogram[0]已经赋值了
	//观察ptmp数据，检查统计是否正确？
	for (int i = 1; i < 65536; i++)
	{
		Histogram[i] = Histogram[i - 1] + Histogram[i];
//		tmp_float[i] = Histogram[i] / tValue;
		tmp[i] = Histogram[i];
	}
	cudaStatus = cudaMemcpy(dev_Histogram, Histogram, 65536 * sizeof(unsigned int), cudaMemcpyHostToDevice);
	//需要根据分辨率调整增强参数
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
	//Histogram_Enhancement << <1, 1 >> >(dev_img);  //串行

	//-------------线性拉伸------------------

	//求最大值最小值
/*	cudaMemcpy(dev_Histogram, ptmp, 2 * sizeof(unsigned short), cudaMemcpyHostToDevice); //更新表值
	//Linear_1 << <numBlocks, threadsPerBlock >> > (dev_img, dev_Histogram);
	Find_Max <<< Height, Width >>> (dev_img, dev_Histogram,&dev_index[0]);  //0 --- width  \\  1------height
	Find_Max <<< 1, 1024 >>> (dev_Histogram, dev_Histogram, &dev_index[1]);
	cudaMemcpy(ptmp, dev_Histogram, 1 * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	Find_Min << < Height, Width >> > (dev_img, dev_Histogram,&dev_index[0]);
	Find_Min << < 1, 1024 >> > (dev_Histogram, dev_Histogram,&dev_index[1]);
	cudaMemcpy(ptmp+1, dev_Histogram+1, 1 * sizeof(unsigned short), cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_Histogram, ptmp, 2 * sizeof(unsigned short), cudaMemcpyHostToDevice); //更新表值

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

	//应该每次都需要清空内存吧
	//cudaFree(dev_img);

	return cudaStatus;
}

//*****************************************************************************************

//----------------------两点矫正处理函数-------------------------
__global__ void GPU_TwoPoint_Helper(unsigned short *gpu_img, float* dev_pTP_Gain, float* dev_pTP_Bias)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	gpu_img[i] = unsigned short(dev_pTP_Gain[i] * gpu_img[i] + dev_pTP_Bias[i]);
}
//-----------------------两点矫正接口函数--------------------------
extern "C"
cudaError_t GPU_TwoPoint_Correction(unsigned short *Image, unsigned short *dev_img, float* dev_pTP_Gain, float* dev_pTP_Bias, int Height, int Width)  //  pTP_Gain,pTP_Bias指向两点矫正参数----TP_On表示是否开启两点矫正
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dev_img, Image, Height * Width * sizeof(unsigned short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	//--------------------------------两点矫正调用--------------------------------
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

	//应该每次都需要清空内存吧
	//cudaFree(dev_img);

	return cudaStatus;
}

//*****************************************************************************************

//----------------------盲元矫正处理函数-------------------------
__global__ void GPU_Blind__Helper(unsigned short *gpu_img, unsigned short* dev_pBlind_Ram, int Height, int Width)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	if (dev_pBlind_Ram[i*blockDim.x+j] == 1) //等于1表示为盲元
	{
		//将盲元替换为周围四个像素的平均
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
//-----------------------盲元矫正--------------------------
extern "C"
cudaError_t GPU_Blind_Correction(unsigned short *Image, unsigned short *dev_img, unsigned short *dev_pBlind_Ram, int Height, int Width)  //  pTP_Gain,pTP_Bias指向两点矫正参数----TP_On表示是否开启两点矫正
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dev_img, Image, Height * Width * sizeof(unsigned short), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return cudaStatus;
	}
	//--------------------------------两点矫正调用--------------------------------
	dim3 threadsPerBlock(Width);
	dim3 numBlocks(Height);
	GPU_Blind__Helper << <numBlocks, threadsPerBlock >> > (dev_img, dev_pBlind_Ram, Height, Width);
	//Blind_On_Correction << <1, 1 >> > (dev_img, Height, Width, dev_pBlind_Ram);
	
/*	// 测试代码
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

	//应该每次都需要清空内存吧
	//cudaFree(dev_img);

	return cudaStatus;
}

