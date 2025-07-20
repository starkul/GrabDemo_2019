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
__global__ void convert_HWC_to_NCHW_and_normalize_kernel(
    const float* src,
    size_t src_step,
    float* dst,
    int width,
    int height,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // 1. 正确计算源地址 (HWC)
    // src_step 是以字节为单位的，所以要先转成 char*
    const float* p_src_pixel = (const float*)((const char*)src + y * src_step) + x * 3;

    // 2. 读取 RGB 值 (假设输入已经是RGB)
    float r = p_src_pixel[0];
    float g = p_src_pixel[1];
    float b = p_src_pixel[2];

    // 3. 在寄存器中完成标准化
    r = (r - mean_r) / std_r;
    g = (g - mean_g) / std_g;
    b = (b - mean_b) / std_b;

    // 4. 直接写入到目标 NCHW 格式的正确位置
    size_t channel_offset = (size_t)height * width;
    dst[y * width + x] = r;                       // 写入 R 通道平面
    dst[channel_offset + y * width + x] = g;      // 写入 G 通道平面
    dst[2 * channel_offset + y * width + x] = b;  // 写入 B 通道平面
}

// C++ 包装函数的实现
void convert_HWC_to_NCHW_and_normalize(
    const cv::cuda::GpuMat& src, 
    float* dst, 
    int width, 
    int height, 
    const float* mean,
    const float* std,
    cudaStream_t stream)
{
    dim3 block(16, 16); // 线程块大小
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    convert_HWC_to_NCHW_and_normalize_kernel<<<grid, block, 0, stream>>>(
        (const float*)src.data,
        src.step,
        dst,
        width,
        height,
        mean[0], mean[1], mean[2], // 将mean/std直接作为值传递，效率更高
        std[0], std[1], std[2]
    );
}