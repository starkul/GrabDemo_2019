#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <iostream>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

__global__ void hwcToNchwKernel(const float* src, float* dst, 
                                int height, int width, int channels,
                                float mean0, float mean1, float mean2,
                                float std0, float std1, float std2) 
{
    // 计算全局索引
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalPixels = height * width;
    
    if (idx < totalPixels) {
        // 计算像素位置
        const int y = idx / width;
        const int x = idx % width;
        
        // 原始数据索引 (HWC布局)
        const int hwcIdx = (y * width + x) * channels;
        
        // 处理每个通道
        for (int c = 0; c < channels; c++) {
            // NCHW布局索引
            const int nchwIdx = c * totalPixels + idx;
            
            // 应用标准化
            float value = src[hwcIdx + c];
            
            // 应用通道特定的标准化
            float normalized;
            if (c == 0) {
                normalized = (value - mean0) / std0;
            } else if (c == 1) {
                normalized = (value - mean1) / std1;
            } else {
                normalized = (value - mean2) / std2;
            }
            
            // 转换为FP16
            dst[nchwIdx] = (normalized);
        }
    }
}

void convertHWCtoNCHW(cv::cuda::GpuMat& hwcInput, float* nchwOutput, 
                     cudaStream_t& stream,
                     const float mean[3], const float std[3])
{
    const int height = hwcInput.rows;
    const int width = hwcInput.cols;
    const int channels = hwcInput.channels();
    const int totalPixels = height * width;
    
    // 计算线程块和网格大小
    const int blockSize = 256;
    const int gridSize = (totalPixels + blockSize - 1) / blockSize;
    
    // 调用核函数
    hwcToNchwKernel<<<gridSize, blockSize, 0, stream>>>(
        reinterpret_cast<const float*>(hwcInput.data), 
        nchwOutput, 
        height, 
        width, 
        channels,
        mean[0], mean[1], mean[2],
        std[0], std[1], std[2]
    );
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel error: " + std::string(cudaGetErrorString(err)));
    }
}