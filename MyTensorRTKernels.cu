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
    // ����ȫ������
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalPixels = height * width;
    
    if (idx < totalPixels) {
        // ��������λ��
        const int y = idx / width;
        const int x = idx % width;
        
        // ԭʼ�������� (HWC����)
        const int hwcIdx = (y * width + x) * channels;
        
        // ����ÿ��ͨ��
        for (int c = 0; c < channels; c++) {
            // NCHW��������
            const int nchwIdx = c * totalPixels + idx;
            
            // Ӧ�ñ�׼��
            float value = src[hwcIdx + c];
            
            // Ӧ��ͨ���ض��ı�׼��
            float normalized;
            if (c == 0) {
                normalized = (value - mean0) / std0;
            } else if (c == 1) {
                normalized = (value - mean1) / std1;
            } else {
                normalized = (value - mean2) / std2;
            }
            
            // ת��ΪFP16
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
    
    // �����߳̿�������С
    const int blockSize = 256;
    const int gridSize = (totalPixels + blockSize - 1) / blockSize;
    
    // ���ú˺���
    hwcToNchwKernel<<<gridSize, blockSize, 0, stream>>>(
        reinterpret_cast<const float*>(hwcInput.data), 
        nchwOutput, 
        height, 
        width, 
        channels,
        mean[0], mean[1], mean[2],
        std[0], std[1], std[2]
    );
    
    // ������
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel error: " + std::string(cudaGetErrorString(err)));
    }
}