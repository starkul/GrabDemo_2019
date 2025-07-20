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
    cv::Size originalSize;    // ԭʼͼ��ߴ�
    float scaleRatio;         // ���ű���
    cv::Point paddingOffset;  // ���ƫ����
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
    // ���������õ�ǰģ�����͵Ĺ�������
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
 * @brief רΪ��ȹ���ģ����Ƶ�Ԥ������.
 * @param inputImage �����ԭʼͼ�� (BGR��ʽ).
 */
    void preprocessImage_Depth(const cv::Mat& inputImage);

    /**
     * @brief רΪ��ȹ���ģ����Ƶĺ�����.
     * @return ����һ�����ӻ��Ĳ�ɫ���ͼ (cv::Mat, BGR��ʽ).
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
    std::vector<float*> m_outputBuffers;  // ֧�ֶ�����
    nvinfer1::Dims m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;  // ֧�ֶ�����ά��
    bool m_enableFP16;
    PreprocessParams m_lastPreprocessParams;
    cv::Size m_lastOriginalImageSize; // ���ڴ洢���һ�δ����ԭʼͼ��ߴ�
    //std::vector<half> m_outputDataFP16;
    std::vector<float> m_outputDataFP32;
    std::vector<const char*> m_inputNames;   // ������������
    std::vector<const char*> m_outputNames;  // �����������
    ModelType m_modelType; // ��¼�Լ���ģ������
    // GPU����Ԥ�������
    /*cv::cuda::GpuMat    m_gpu_input;
    cv::cuda::GpuMat    m_gpu_resized;
    cv::cuda::GpuMat    m_gpu_padded;
    cv::cuda::GpuMat    m_gpu_rgb;
    cv::cuda::GpuMat    m_gpu_normalized;
    std::vector<cv::cuda::GpuMat> m_gpu_channels;*/
    int m_inputCount = 0;
    // ��������֧��
    //int m_maxBatchSize;
    //std::vector<half> m_batchPreprocessedData;

    // ��־�� 
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING)
                std::cerr << "[TensorRT] " << msg << std::endl;
        }
    } m_logger;
};