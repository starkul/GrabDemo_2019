#pragma once
#include <NvInfer.h> 
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <iostream>
#include <vector> 
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

struct Detection {
    float x, y, width, height;
    float confidence;
    int classId;
};

struct PreprocessParams {
    cv::Size originalSize;    // 原始图像尺寸
    float scaleRatio;         // 缩放比例
    cv::Point paddingOffset;  // 填充偏移量
};
enum class ModelType {
    Unknown,
    SuperResolution_IINet,
    DepthEstimation_DepthAnything,
    ObjectDetection_YOLOv8
};
class MyTensorRT
{
public:
    MyTensorRT(const std::string& enginePath, bool enableFP16 = false);
    ~MyTensorRT();
    // 新增：设置当前模型类型的公共方法
    void setModelType(ModelType type);
    void inference(int batchSize = 1);
    void saveEngine(const std::string& path);
    nvinfer1::DataType getInputDataType() const;
    nvinfer1::Dims  getInputDims();
    nvinfer1::Dims getOutputDims(int index);
    void preprocessImage_Detect(const cv::Mat& inputImage);
    std::vector<Detection> postprocessOutput(int batchSize);
    std::vector<Detection> postprocessOutputYOLOV8(int batchSize);
    std::string getClassName(int classId) const;
    void warmup(int numIterations);
    /**
 * @brief 专为深度估计模型设计的预处理函数.
 * @param inputImage 输入的原始图像 (BGR格式).
 */
    void preprocessImage_Depth(const cv::Mat& inputImage);

    /**
     * @brief 专为深度估计模型设计的后处理函数.
     * @return 返回一个可视化的彩色深度图 (cv::Mat, BGR格式).
     */
    cv::Mat postprocessOutput_Depth(int batchSize);
    // ===================================================
    void preprocessImage_Super(const cv::Mat& inputImage);
    cv::Mat postprocessOutput_Super(int batchSize);
    void preprocessImage_LightField(const cv::Mat& nine_grid_image, int angular_resolution);
private:
    void loadEngine(const std::string& path);
    void setupMemory();
    int64_t volume(const nvinfer1::Dims& d);
    std::vector<Detection> applyNMS(const std::vector<Detection>& candidates, float iouThreshold);
    float calculateIOU(const Detection& a, const Detection& b);
    void validateMemoryAccess();
    int m_upscaleFactor;
private:
    nvinfer1::IRuntime* m_runtime = nullptr;
    nvinfer1::ICudaEngine* m_engine = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;
    cudaStream_t m_stream;
    float* m_inputBuffer = nullptr;
    std::vector<float*> m_outputBuffers;  // 支持多个输出
    nvinfer1::Dims m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;  // 支持多个输出维度
    bool m_enableFP16;
    PreprocessParams m_lastPreprocessParams;
    cv::Size m_lastOriginalImageSize; // 用于存储最后一次处理的原始图像尺寸
    //std::vector<half> m_outputDataFP16;
    std::vector<float> m_outputDataFP32;
    std::vector<const char*> m_inputNames;   // 输入张量名称
    std::vector<const char*> m_outputNames;  // 输出张量名称
    ModelType m_modelType; // 记录自己的模型类型
    // GPU加速预处理相关
    /*cv::cuda::GpuMat    m_gpu_input;
    cv::cuda::GpuMat    m_gpu_resized;
    cv::cuda::GpuMat    m_gpu_padded;
    cv::cuda::GpuMat    m_gpu_rgb;
    cv::cuda::GpuMat    m_gpu_normalized;
    std::vector<cv::cuda::GpuMat> m_gpu_channels;*/
    int m_inputCount = 0;
    // 批量处理支持
    //int m_maxBatchSize;
    //std::vector<half> m_batchPreprocessedData;

    // 日志器 
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING)
                std::cerr << "[TensorRT] " << msg << std::endl;
        }
    } m_logger;
};