#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

__constant__ float color_weight[4 * 256];
__constant__ float space_weight[1024];

USHORT *dImage = NULL;   //original image
USHORT *dTemp = NULL;   //temp array for iterations
size_t pitch;

texture<uchar4, 2, cudaReadModeElementType> rgbaTex;//声明纹理参照系


__device__ float colorLenGaussian(uchar4 a, uchar4 b)
{
	//若想达到漫画效果，就注释掉sqrt,使颜色距离变大
	USHORT mod = (USHORT)sqrt(((float)b.x - (float)a.x) * ((float)b.x - (float)a.x) +
		((float)b.y - (float)a.y) * ((float)b.y - (float)a.y) +
		((float)b.z - (float)a.z) * ((float)b.z - (float)a.z) +
		((float)b.w - (float)a.w) * ((float)b.w - (float)a.w));

	return color_weight[mod];
}
__device__ uint rgbaFloatToInt(float4 rgba)
{
	rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(fabs(rgba.y));
	rgba.z = __saturatef(fabs(rgba.z));
	rgba.w = __saturatef(fabs(rgba.w));
	return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}
__device__ float4 rgbaIntToFloat(uint c)
{
	float4 rgba;
	rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
	rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;  //  /255.0f;
	rgba.z = ((c >> 16) & 0xff) * 0.003921568627f; //  /255.0f;
	rgba.w = ((c >> 24) & 0xff) * 0.003921568627f; //  /255.0f;
	return rgba;
}
//column pass using coalesced global memory reads
__global__ void
d_bilateral_filter(USHORT *od, int w, int h, int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= w || y >= h)
	{
		return;
	}

	float sum = 0.0f;
	float factor = 0.0f;;
	uchar4 t = { 0, 0, 0, 0 };
	float tw = 0.f, tx = 0.f, ty = 0.f, tz = 0.f;
	uchar4 center = tex2D(rgbaTex, x, y);
	//t = center;
	int posIndex = 0;
	for (int i = -r; i <= r; i++)
	{
		for (int j = -r; j <= r; j++)
		{
			uchar4 curPix = { 0, 0, 0, 0 };
			USHORT d = (USHORT)sqrt((double)i*i + (double)j*j);
			if (d>r)
				continue;

			if (x + j<0 || y + i<0 || x + j>w - 1 || y + i>h - 1)
			{
				factor = 0;
			}
			else
			{
				curPix = tex2D(rgbaTex, x + j, y + i);
				factor = space_weight[d] *     //domain factor
					colorLenGaussian(curPix, center);             //range factor
			}


			tw += factor * (float)curPix.w;
			tx += factor * (float)curPix.x;
			ty += factor * (float)curPix.y;
			tz += factor * (float)curPix.z;
			sum += factor;
		}
	}
	t.w = (UCHAR)(tw / sum);
	t.x = (UCHAR)(tx / sum);
	t.y = (UCHAR)(ty / sum);
	t.z = (UCHAR)(tz / sum);
	od[y * w + x] = (USHORT)(((UINT)t.w) << 24 | ((UINT)t.z) << 16 | ((UINT)t.y) << 8 | ((UINT)t.x));

}

extern "C"
void updateGaussian(float sigma_color, float sigma_space, int radius)
{
	if (sigma_color <= 0)
		sigma_color = 1;
	if (sigma_space <= 0)
		sigma_space = 1;
	double gauss_color_coeff = -0.5 / (sigma_color*sigma_color);
	double gauss_space_coeff = -0.5 / (sigma_space*sigma_space);

	float color_gaussian[4 * 256];
	float space_gaussian[1024];

	for (int i = 0; i<256 * 4; i++)
	{
		color_gaussian[i] = (float)std::exp(i*i*gauss_color_coeff);
		space_gaussian[i] = (float)std::exp(i*i*gauss_space_coeff);
		//if(i>100) color_gaussian[i] = 0.0f; //漫画效果
	}
	// 	for(int i = -radius,int maxk=0;i<radius;i++)
	// 		for(int j=-radius;j<radius;j++)
	// 		{
	// 			double r = sqrt((double)i*i + (double)j*j);
	// 			 if( r > radius )
	//                 continue;  
	// 			space_gaussian[maxk++] = (float)std::exp(r*r*gauss_space_coeff); 
	// 			//space_ofs[maxk++] = (int)(i*temp.step + j*4);  
	// 		}

	checkCudaErrors(cudaMemcpyToSymbol(color_weight, color_gaussian, sizeof(float)*(4 * 256)));
	checkCudaErrors(cudaMemcpyToSymbol(space_weight, space_gaussian, sizeof(float)*(1024)));
}

//---------------------初始化内存---------------------
extern "C"
void initTexture(int width, int height, USHORT *hImage)
{
	// copy image data to array
	//  cudaMallocPitch是将内存安装对齐的方式进行开辟内存，所以使用的时候需要返回pitch值，表示当前存储每行有pitch列，访问时使用a[pitch*i+j]代替原来的a[row*i+j]
	checkCudaErrors(cudaMallocPitch(&dImage, &pitch, sizeof(USHORT)*width, height));
	checkCudaErrors(cudaMallocPitch(&dTemp, &pitch, sizeof(USHORT)*width, height));
	checkCudaErrors(cudaMemcpy2D(dImage, pitch, hImage, sizeof(USHORT)*width,
		sizeof(USHORT)*width, height, cudaMemcpyHostToDevice));
}

//---------------------释放内存----------------------------
extern "C"
void freeTextures()
{
	checkCudaErrors(cudaFree(dImage));
	checkCudaErrors(cudaFree(dTemp));
}

// RGBA version
extern "C"
double bilateralFilterRGBA(USHORT *dDest,
	int width, int height,
	int radius, int iterations,
	StopWatchInterface *timer)
{
	// var for kernel computation timing
	double dKernelTime;

	// Bind the array to the texture
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
	checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dImage, desc, width, height, pitch));

	for (int i = 0; i<iterations; i++)
	{
		// sync host and start kernel computation timer
		dKernelTime = 0.0;
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);

		dim3 blockSize(16, 16);
		dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);

		d_bilateral_filter << < gridSize, blockSize >> >(dDest, width, height, radius);

		// sync host and stop computation timer
		checkCudaErrors(cudaDeviceSynchronize());
		dKernelTime += sdkGetTimerValue(&timer);

	}

	return ((dKernelTime / 1000.) / (double)iterations);
}

//-----------------------GPU接口实现--------------------
//GPU_Bilatetal_Filter()
//{
//
//}