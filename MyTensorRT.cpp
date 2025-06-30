#include "MyTensorRT.h"
#include <fstream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <afx.h>         // MFC核心头文件
#include <atlstr.h>      // ATL字符串类（如果不使用MFC）

//#define DEBUG

void convertHWCtoNCHW(cv::cuda::GpuMat& hwcInput, float* nchwOutput,
    cudaStream_t& stream,
    const float mean[3], const float std[3]);

MyTensorRT::MyTensorRT(const std::string& enginePath, bool enableFP16) : m_enableFP16(enableFP16)
{
    try {
        loadEngine(enginePath);
        setupMemory();
    }
    catch (const std::exception& e) {
        CString errorMsg;
        errorMsg.Format(_T("TensorRT初始化失败:\n%hs"), e.what());
        MessageBox(NULL, errorMsg, _T("致命错误"), MB_OK | MB_ICONERROR);
        throw;
    }
}

MyTensorRT::~MyTensorRT()
{
    // 释放 GPU 内存
    if (m_inputBuffer) cudaFree(m_inputBuffer);

    for (auto buffer : m_outputBuffers) {
        if (buffer) cudaFree(buffer);
    }

    // TensorRT资源释放
    if (m_context) delete m_context;
    if (m_engine) delete m_engine;
    if (m_runtime) delete m_runtime;

    // 销毁 CUDA 流
    if (m_stream) cudaStreamDestroy(m_stream);
}

void MyTensorRT::validateMemoryAccess()
{
    // 验证输入内存可访问性
    cudaError_t status = cudaMemsetAsync(m_inputBuffer, 0, 128, m_stream);
    if (status != cudaSuccess) {
        throw std::runtime_error("输入内存访问失败: " + std::string(cudaGetErrorString(status)));
    }

    // 验证输出内存可访问性
    for (auto buffer : m_outputBuffers) {
        status = cudaMemsetAsync(buffer, 0, 128, m_stream);
        if (status != cudaSuccess) {
            throw std::runtime_error("输出内存访问失败: " + std::string(cudaGetErrorString(status)));
        }
    }

    // 等待所有操作完成
    cudaStreamSynchronize(m_stream);

    // 使用cudaPointerGetAttributes验证内存类型
    struct cudaPointerAttributes attr;
    status = cudaPointerGetAttributes(&attr, m_inputBuffer);
    if (status != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        throw std::runtime_error("输入内存不是设备内存");
    }

    for (auto buffer : m_outputBuffers) {
        status = cudaPointerGetAttributes(&attr, buffer);
        if (status != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
            throw std::runtime_error("输出内存不是设备内存");
        }
    }

    // 检查内存对齐
    const uintptr_t alignment = 128;
    if (reinterpret_cast<uintptr_t>(m_inputBuffer) % alignment != 0) {
        throw std::runtime_error("输入内存未对齐");
    }

    for (auto buffer : m_outputBuffers) {
        if (reinterpret_cast<uintptr_t>(buffer) % alignment != 0) {
            throw std::runtime_error("输出内存未对齐");
        }
    }
}

// 执行推理 (支持批量输入)
void MyTensorRT::inference(int batchSize)
{
    //m_context->setInputShape(m_engine->getIOTensorName(0), m_inputDims);
    cudaStreamSynchronize(m_stream);
    std::vector<const char*> tensorNames;
    std::vector<void*> tensorAddresses;
    if (m_inputBuffer == nullptr) {
        throw std::runtime_error("输入缓冲区为空");
    }
    // 按照引擎中张量名称的顺序绑定缓冲区
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) 
    {
        const char* name = m_engine->getIOTensorName(i);
        tensorNames.push_back(name);

        // 检查是否是输入张量
        if (std::find(m_inputNames.begin(), m_inputNames.end(), name) != m_inputNames.end()) {
            tensorAddresses.push_back(m_inputBuffer);
        }
        // 检查是否是输出张量
        else if (std::find(m_outputNames.begin(), m_outputNames.end(), name) != m_outputNames.end()) {
            // 查找输出索引
            auto it = std::find(m_outputNames.begin(), m_outputNames.end(), name);
            int idx = std::distance(m_outputNames.begin(), it);

            if (idx < 0 || idx >= static_cast<int>(m_outputBuffers.size())) {
                throw std::runtime_error("输出缓冲区索引越界: " + std::string(name));
            }
            tensorAddresses.push_back(m_outputBuffers[idx]);
        }
        else {
            throw std::runtime_error("未知的张量类型: " + std::string(name));
        }
    }
    if (tensorAddresses.size() == 0)
    {
        throw std::runtime_error("张量为0: ");
        // 验证绑定数量
        if (tensorAddresses.size() != m_engine->getNbIOTensors()) {
            throw std::runtime_error("绑定数量不匹配");
        }
    }

#ifdef DEBUG
    for (int i = 0; i < tensorAddresses.size(); ++i) {
        CString msg;
        msg.Format(_T("张量 %d [%hs]: 地址 %p"),
            i, tensorNames[i], tensorAddresses[i]);
        //OutputDebugString(msg);
        MessageBox(NULL, msg, _T("张量验证"), MB_OK | MB_ICONINFORMATION);
    }
#endif
    // 使用新的executeV2接口
    bool success = m_context->executeV2(tensorAddresses.data());
    //bool success = m_context->enqueueV3(m_stream);
    if (!success) {
        cudaStreamSynchronize(m_stream);
        cudaError_t err = cudaGetLastError();
        throw std::runtime_error("推理失败: " + std::string(cudaGetErrorString(err)));
    }
    int64_t outputElements = volume(m_outputDims[0]);
    // 新增：验证输出缓冲区大小
    if (m_outputDataFP32.size() < outputElements) {
        throw std::runtime_error("输出缓冲区大小不足");
    }
    //推理结果通过之前的张量绑定自动保存在m_outputBuffers中，复制到m_outputDataFP16中用于后处理
    cudaMemcpyAsync(m_outputDataFP32.data(), m_outputBuffers[0],
        outputElements * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    // 转换FP16到FP32
    /*for (size_t i = 0; i < m_outputDataFP16.size(); ++i) {
        m_outputDataFP32[i] = __half2float(m_outputDataFP16[i]);
    }*/

#ifdef DEBUG

    if (m_outputDataFP32.size() >= 10) 
    {
        CString outputInfo;
        outputInfo.Format(_T("推理输出数据前10个值:\n"));

        // 获取输出维度
        nvinfer1::Dims outDims = getOutputDims(0);
        const int numChannels = outDims.d[1];  // 10个通道
        const int numBoxes = outDims.d[2];     // 8400个框

        // 测试1: 假设行优先(通道在后)
        outputInfo.AppendFormat(_T("\n假设行优先(框优先):\n"));
        for (int i = 0; i < 3; i++) {
            outputInfo.AppendFormat(_T("框 %d (索引 %d-%d): "), i, i * numChannels, (i + 1) * numChannels - 1);

            float x1 = m_outputDataFP32[i * numChannels + 0];
            float y1 = m_outputDataFP32[i * numChannels + 1];
            float x2 = m_outputDataFP32[i * numChannels + 2];
            float y2 = m_outputDataFP32[i * numChannels + 3];

            outputInfo.AppendFormat(_T("坐标=(%.2f,%.2f)-(%.2f,%.2f) "), x1, y1, x2, y2);

            // 查找最大类别分数
            float max_score = 0;
            int class_id = 0;
            for (int c = 0; c < 6; c++) {
                float score = m_outputDataFP32[i * numChannels + 4 + c];
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }
            outputInfo.AppendFormat(_T("类别=%d 分数=%.4f\n"), class_id, max_score);
        }

        // 测试2: 假设列优先(通道在前)
        outputInfo.AppendFormat(_T("\n假设列优先(通道优先):\n"));
        for (int i = 0; i < 3; i++) {
            outputInfo.AppendFormat(_T("框 %d (索引 %d,%d,%d,%d...): "), i, i, i + numBoxes, i + 2 * numBoxes, i + 3 * numBoxes);

            float x1 = m_outputDataFP32[i + 0 * numBoxes];
            float y1 = m_outputDataFP32[i + 1 * numBoxes];
            float x2 = m_outputDataFP32[i + 2 * numBoxes];
            float y2 = m_outputDataFP32[i + 3 * numBoxes];

            outputInfo.AppendFormat(_T("坐标=(%.2f,%.2f)-(%.2f,%.2f) "), x1, y1, x2, y2);

            // 查找最大类别分数
            float max_score = 0;
            int class_id = 0;
            for (int c = 0; c < 6; c++) {
                float score = m_outputDataFP32[i + (4 + c) * numBoxes];
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }
            outputInfo.AppendFormat(_T("类别=%d 分数=%.4f\n"), class_id, max_score);
        }

        MessageBox(NULL, outputInfo, _T("推理输出验证"), MB_OK | MB_ICONINFORMATION);
    }else {
        MessageBox(NULL, _T("输出数据不足10个元素"), _T("推理输出验证"), MB_OK | MB_ICONWARNING);
    }
#endif // DEBUG

}

// 保存优化后的引擎
void MyTensorRT::saveEngine(const std::string& path)
{
    std::ofstream outFile(path, std::ios::binary);
    auto serializedEngine = m_engine->serialize();
    outFile.write(reinterpret_cast<const char*>(serializedEngine->data()),
        serializedEngine->size());
}

// 加载引擎文件
void MyTensorRT::loadEngine(const std::string& path)
{
    try {
        //initLibNvInferPlugins(&m_logger, "");
        // 1. 打开二进制文件并验证
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Engine file not found or unreadable");
        }
        // 2. 获取文件大小并分配缓冲区
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        // 3. 读取文件内容到缓冲区
        file.read(buffer.data(), size);
        if (!file) {
            throw std::runtime_error("Failed to read engine file");
        }
        // 4. 创建TensorRT运行时环境
        m_runtime = nvinfer1::createInferRuntime(m_logger);
        if (!m_runtime) {
            throw std::runtime_error("Failed to create runtime");
        }
        // 5. 反序列化引擎数据
        m_engine = m_runtime->deserializeCudaEngine(buffer.data(), size);
        if (!m_engine) {
            throw std::runtime_error("Deserialize failed: invalid engine data");
        }
        // 6. 创建执行上下文
        m_context = m_engine->createExecutionContext();
        if (!m_context) {
            throw std::runtime_error("Execution context creation failed");
        }
        // 7. 获取输入/输出绑定数量
        const int numBindings = m_engine->getNbIOTensors();
        if (numBindings < 1) {
            throw std::runtime_error("Engine must have at least 1 binding");
        }
        // 8. 遍历绑定并提取维度信息
        m_inputCount = 0;
        for (int i = 0; i < numBindings; ++i) {
            if (m_engine->getTensorIOMode(m_engine->getIOTensorName(i)) == nvinfer1::TensorIOMode::kINPUT) {
                if (m_inputCount > 0) {
                    throw std::runtime_error("Only one input is supported");
                }
                m_inputDims = m_engine->getTensorShape(m_engine->getIOTensorName(i));
                m_inputCount++;
            }
            else {
                m_outputDims.push_back(m_engine->getTensorShape(m_engine->getIOTensorName(i)));
            }
        }
        m_inputNames.clear();
        m_outputNames.clear();

        for (int i = 0; i < numBindings; ++i) {
            const char* name = m_engine->getIOTensorName(i);
            if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                m_inputNames.push_back(name);
            }
            else {
                m_outputNames.push_back(name);
            }
        }

        // 验证输入输出数量
        if (m_inputNames.size() != 1 || m_outputNames.empty()) {
            throw std::runtime_error("引擎输入输出数量不符合预期");
        }

#ifdef DEBUG
        CString namesInfo;
        namesInfo.Format(_T("输入张量: %hs\n"), m_inputNames[0]);
        for (size_t i = 0; i < m_outputNames.size(); ++i) {
            namesInfo.AppendFormat(_T("输出张量 %d: %hs\n"), i, m_outputNames[i]);
        }
        MessageBox(NULL, namesInfo, _T("张量名称"), MB_OK);
#endif
#ifdef DEBUG
        CString dimInfo;
        dimInfo.Format(_T("引擎加载成功！\n输入维度: ["));
        for (int i = 0; i < m_inputDims.nbDims; i++) {
            CString dim;
            dim.Format(_T("%d"), m_inputDims.d[i]);
            dimInfo += dim;
            if (i < m_inputDims.nbDims - 1) dimInfo += _T(", ");
        }
        dimInfo += _T("]\n输出维度: ");

        for (int i = 0; i < m_outputDims.size(); i++) {
            dimInfo += _T("[");
            for (int j = 0; j < m_outputDims[i].nbDims; j++) {
                CString dim;
                dim.Format(_T("%d"), m_outputDims[i].d[j]);
                dimInfo += dim;
                if (j < m_outputDims[i].nbDims - 1) dimInfo += _T(", ");
            }
            dimInfo += _T("]");
            if (i < m_outputDims.size() - 1) dimInfo += _T(", ");
        }

        MessageBox(NULL, dimInfo, _T("TensorRT 引擎信息"), MB_OK | MB_ICONINFORMATION);
#endif // DEBUG


    }
    catch (const std::exception& e) {
        CString errorMsg;
        errorMsg.Format(_T("LoadEngine failed: %hs"), e.what());
        OutputDebugString(errorMsg);
        MessageBox(NULL, errorMsg, _T("TensorRT Error"), MB_OK | MB_ICONERROR);
        throw;
    }
}

// 分配GPU内存 
void MyTensorRT::setupMemory()
{
    try {
        // 计算输入元素数量和字节大小
        int64_t inputElements = volume(m_inputDims);
        size_t inputBytes = inputElements * sizeof(float);//alignedSize(inputElements, sizeof(float));

        // 分配输入内存
        cudaError_t status = cudaMalloc((void**)&m_inputBuffer, inputBytes);
        if (status != cudaSuccess) {
            throw std::runtime_error("输入缓冲区分配失败: " +
                std::string(cudaGetErrorString(status)));
        }

        // 初始化输入内存
        cudaMemset(m_inputBuffer, 0, inputBytes);

        // 分配输出内存
        m_outputBuffers.clear();
        for (const auto& dims : m_outputDims) {
            int64_t outputElements = volume(dims);
            size_t outputBytes = outputElements * sizeof(float);//alignedSize(outputElements, sizeof(half));

            float* outputBuffer = nullptr;
            status = cudaMalloc((void**)&outputBuffer, outputBytes);
            if (status != cudaSuccess) {
                throw std::runtime_error("输出缓冲区分配失败: " +
                    std::string(cudaGetErrorString(status)));
            }

            // 初始化输出内存
            cudaMemset(outputBuffer, 0, outputBytes);
            m_outputBuffers.push_back(outputBuffer);
        }

        // 初始化CPU端输出数据容器
        int64_t maxOutputElements = 0;
        for (const auto& dims : m_outputDims) {
            maxOutputElements = max(maxOutputElements, volume(dims));
        }

        //m_outputDataFP16.resize(maxOutputElements);
        m_outputDataFP32.resize(maxOutputElements);

        // 创建CUDA流
        status = cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            throw std::runtime_error("CUDA流创建失败: " +
                std::string(cudaGetErrorString(status)));
        }

        // 打印内存分配详情
#ifdef DEBUG
        CString memInfo;
        int outputElemsDebug = volume(m_outputDims[0]);
        int outputBytesDebug = volume(m_outputDims[0]) * sizeof(float);
        memInfo.Format(_T("输入内存: %lld 元素, %u 字节\n")
            _T("输出内存: %lld 元素, %u 字节"),
            inputElements, inputBytes,
            outputElemsDebug,
            outputBytesDebug);
        MessageBox(NULL, memInfo, _T("内存分配详情"), MB_OK);
#endif
        validateMemoryAccess();
    }
    catch (const std::exception& e) {
        CString errorMsg;
        errorMsg.Format(_T("SetupMemory failed: %hs"), e.what());
        MessageBox(NULL, errorMsg, _T("Error"), MB_OK | MB_ICONERROR);
        throw;
    }
}

// 计算张量体积
int64_t MyTensorRT::volume(const nvinfer1::Dims& d)
{
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; i++) {
        v *= d.d[i];
    }
    return v;
}

nvinfer1::DataType MyTensorRT::getInputDataType() const
{
    return m_engine->getTensorDataType(m_engine->getIOTensorName(0));
}

nvinfer1::Dims MyTensorRT::getInputDims()
{
    return m_inputDims;
}

nvinfer1::Dims MyTensorRT::getOutputDims(int index)
{
    if (index >= 0 && index < static_cast<int>(m_outputDims.size())) {
        return m_outputDims[index];
    }
    throw std::out_of_range("Invalid output index");
}

void MyTensorRT::preprocessImage(const cv::Mat& inputImage)
{
    // 保存原始尺寸
    m_lastPreprocessParams.originalSize = inputImage.size();
    // 1. 获取模型期望的输入维度 (640x640)
    nvinfer1::Dims dims = getInputDims();
    const int modelHeight = dims.d[2];  // 640
    const int modelWidth = dims.d[3];   // 640
    cv::cuda::GpuMat gpu_input;
    // 1. 上传到GPU
    if (gpu_input.empty()) {
        gpu_input = cv::cuda::GpuMat(inputImage);
    }
    else {
        gpu_input.upload(inputImage); // 优化上传操作
    }

    float scale_x = static_cast<float>(modelWidth) / gpu_input.cols;
    float scale_y = static_cast<float>(modelHeight) / gpu_input.rows;
    float scale = (std::min)(scale_x, scale_y);
    int newWidth = static_cast<int>(gpu_input.cols * scale);
    int newHeight = static_cast<int>(gpu_input.rows * scale);

    // 3. 计算填充
    int dw = (modelWidth - newWidth) / 2;
    int dh = (modelHeight - newHeight) / 2;

    // 保存预处理参数
    m_lastPreprocessParams.scaleRatio = scale;
    m_lastPreprocessParams.paddingOffset = cv::Point(dw, dh);
#ifdef DEBUG
    CString scaleInfo;
    scaleInfo.Format(_T("原始尺寸: %dx%d\n缩放后: %dx%d\n填充: (%d, %d)"),
        gpu_input.cols, gpu_input.rows,
        newWidth, newHeight,
        dw, dh);
    MessageBox(NULL, scaleInfo, _T("缩放参数"), MB_OK | MB_ICONINFORMATION);
#endif // DEBUG
    // 2. 缩放
    cv::cuda::GpuMat gpu_resized;
    cv::cuda::resize(gpu_input, gpu_resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

    // 3. 填充
    cv::cuda::GpuMat gpu_padded(modelHeight, modelWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::cuda::GpuMat roi = gpu_padded(cv::Rect(dw, dh, newWidth, newHeight));
    gpu_resized.copyTo(roi);

    // 4. 转换为RGB
    cv::cuda::GpuMat gpu_rgb;
    cv::cuda::cvtColor(gpu_padded, gpu_rgb, cv::COLOR_BGR2RGB);

    // 5. 转换为浮点并归一化
    cv::cuda::GpuMat gpu_normalized;

    gpu_rgb.convertTo(gpu_normalized, CV_32FC3, 1.0f / 255.0f);
    
    // 6. 标准化 - 使用核函数直接处理
    //const float mean[3] = { 0.485f, 0.456f, 0.406f };
    //const float std[3] = { 0.229f, 0.224f, 0.225f };
    const float dummy_mean[3] = { 0.0f, 0.0f, 0.0f };
    const float dummy_std[3] = { 1.0f, 1.0f, 1.0f };
    cudaStreamSynchronize(m_stream);
    // 直接调用转换核函数（包含标准化）
    convertHWCtoNCHW(gpu_normalized, m_inputBuffer, m_stream, dummy_mean, dummy_std);
    //m_inputBuffer = __half2float(halfInput);
    cudaStreamSynchronize(m_stream);
#ifdef DEBUG
    int64_t inputElements = volume(m_inputDims);
    std::vector<float> testInput(inputElements);
    cudaMemcpy(testInput.data(), m_inputBuffer,
        inputElements * sizeof(float),
        cudaMemcpyDeviceToHost);

    float minVal = FLT_MAX, maxVal = -FLT_MAX;
    for (int i = 0; i < inputElements; i++) {
        float val = (testInput[i]);
        minVal = min(minVal, val);
        maxVal = max(maxVal, val);
    }

    CString inputRange;
    inputRange.Format(_T("预处理后输入范围: %.6f ~ %.6f"), minVal, maxVal);
    MessageBox(NULL, inputRange, _T("输入验证"), MB_OK);
#endif
}
void MyTensorRT::preprocessImage_Super(const cv::Mat& inputImage)
{

}
/**
 * @brief 专为深度估计模型设计的预处理函数。
 *        它将输入图像直接缩放、转换颜色空间、归一化，并转换为CHW格式。
 * @param inputImage 输入的原始图像 (OpenCV Mat, BGR格式)。
 */
void MyTensorRT::preprocessImage_Depth(const cv::Mat& inputImage)
{
    // 1. 记录原始图像尺寸，用于后处理的放大
    m_lastOriginalImageSize = inputImage.size();
    // 1. 获取模型期望的输入维度 (例如 1x3x518x518)
    nvinfer1::Dims dims = getInputDims();
    if (dims.nbDims != 4) {
        throw std::runtime_error("Depth model requires a 4D input tensor (NCHW).");
    }
    const int modelHeight = dims.d[2];
    const int modelWidth = dims.d[3];
    // 3. 【关键】计算保持宽高比的缩放尺寸
    int original_w = inputImage.cols;
    int original_h = inputImage.rows;

    // 计算缩放比例，使得图像能被完整地放入目标框内
    float scale = (std::min)(static_cast<float>(modelWidth) / original_w, static_cast<float>(modelHeight) / original_h);
    int new_w = static_cast<int>(original_w * scale);
    int new_h = static_cast<int>(original_h * scale);
    
    // 4. 按计算出的新尺寸进行【等比例】缩放
    cv::Mat resized_img;
    // 使用INTER_CUBIC以精确匹配Python代码
    cv::resize(inputImage, resized_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    // 5. 创建一个最终尺寸的黑色“画布” (padding)
    cv::Mat padded_img(modelHeight, modelWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    // 6. 将等比例缩放后的图像“粘贴”到画布的左上角
    resized_img.copyTo(padded_img(cv::Rect(0, 0, new_w, new_h)));

    // 7. 对【填充后】的完整图像进行后续处理
    //    (BGR->RGB, 归一化, 减均值/除标准差)
    cv::Mat rgb_img;
    cv::cvtColor(padded_img, rgb_img, cv::COLOR_BGR2RGB);

    cv::Mat float_img;
    rgb_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    const float mean[3] = { 0.485f, 0.456f, 0.406f };
    const float std[3] = { 0.229f, 0.224f, 0.225f };

    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    channels[0] = (channels[0] - mean[0]) / std[0];
    channels[1] = (channels[1] - mean[1]) / std[1];
    channels[2] = (channels[2] - mean[2]) / std[2];

    // 8. 将处理好的通道数据按 CHW 格式拷贝到最终的 CPU 缓冲区
    std::vector<float> cpu_input_buffer(volume(dims));
    size_t channel_size = static_cast<size_t>(modelHeight) * modelWidth;
    memcpy(cpu_input_buffer.data(), channels[0].data, channel_size * sizeof(float));
    memcpy(cpu_input_buffer.data() + channel_size, channels[1].data, channel_size * sizeof(float));
    memcpy(cpu_input_buffer.data() + 2 * channel_size, channels[2].data, channel_size * sizeof(float));

    // 9. 将准备好的数据从 CPU 拷贝到 GPU
    cudaMemcpyAsync(m_inputBuffer, cpu_input_buffer.data(), cpu_input_buffer.size() * sizeof(float), cudaMemcpyHostToDevice, m_stream);

    cudaStreamSynchronize(m_stream);
}
inline float sigmoid(float x) 
{
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<Detection> MyTensorRT::postprocessOutputYOLOV8(int batchSize)
{
    const std::vector<float>& outputData = m_outputDataFP32;
    if (outputData.empty()) return {};

    // 获取输出维度信息 (1, num_classes+4, num_boxes)
    nvinfer1::Dims outDims = getOutputDims(0);
    const int numChannels = outDims.d[1];  // 10 (4坐标+6类别)
    const int numBoxes = outDims.d[2];     // 8400

    // 预处理参数
    float scale = m_lastPreprocessParams.scaleRatio;
    int dw = m_lastPreprocessParams.paddingOffset.x;
    int dh = m_lastPreprocessParams.paddingOffset.y;
    float originalWidth = static_cast<float>(m_lastPreprocessParams.originalSize.width);
    float originalHeight = static_cast<float>(m_lastPreprocessParams.originalSize.height);

    // 存储候选框
    std::vector<Detection> candidates;
    float maxConf = 0.0f;
    int highConfBoxIdx = -1;
    int maxConfClass = -1;
    const float* outputPtr = outputData.data();
    for (int boxIdx = 0; boxIdx < numBoxes; ++boxIdx)
    {
        //const float* det = &outputData[boxIdx * numChannels];

        // 解析坐标 (cx, cy, w, h) - 与Python一致
        float cx_padded = outputPtr[0 * numBoxes + boxIdx];
        float cy_padded = outputPtr[1 * numBoxes + boxIdx];
        float w_padded = outputPtr[2 * numBoxes + boxIdx];
        float h_padded = outputPtr[3 * numBoxes + boxIdx];

        // 计算类别置信度 (与Python一致，直接取最大值)
        float confidence = 0.0f;
        int classId = 0;
        for (int c = 0; c < numChannels - 4; ++c) {
            float score = outputPtr[(4 + c) * numBoxes + boxIdx];
            if (score > confidence) {
                confidence = score;
                classId = c;
            }
        }

        // 记录最高置信度框用于调试
        if (confidence > maxConf) {
            maxConf = confidence;
            highConfBoxIdx = boxIdx;
            maxConfClass = classId;
        }

        // 应用置信度阈值
        if (confidence < 0.25) continue;

        // 映射回原始图像坐标 (与Python一致)
        float cx_orig = (cx_padded - dw) / scale;
        float cy_orig = (cy_padded - dh) / scale;
        float w_orig = w_padded / scale;
        float h_orig = h_padded / scale;

        // 转换为xywh格式
        float x = cx_orig - w_orig / 2;
        float y = cy_orig - h_orig / 2;

        // 边界保护
        x = max(0.0f, min(x, originalWidth - 1));
        y = max(0.0f, min(y, originalHeight - 1));
        w_orig = min(w_orig, originalWidth - x);
        h_orig = min(h_orig, originalHeight - y);

        if (w_orig <= 1 || h_orig <= 1) continue;

        Detection candidate;
        candidate.x = x;
        candidate.y = y;
        candidate.width = w_orig;
        candidate.height = h_orig;
        candidate.confidence = confidence;
        candidate.classId = classId;
        candidates.push_back(candidate);
    }

#ifdef DEBUG
    // 打印最高置信度框信息
    if (highConfBoxIdx >= 0) {
        float cx = outputPtr[0 * numBoxes + highConfBoxIdx];
        float cy = outputPtr[1 * numBoxes + highConfBoxIdx];
        float w = outputPtr[2 * numBoxes + highConfBoxIdx];
        float h = outputPtr[3 * numBoxes + highConfBoxIdx];
        float classScore0 = outputPtr[4 * numBoxes + highConfBoxIdx];
        float classScore1 = outputPtr[5 * numBoxes + highConfBoxIdx];
        float classScore2 = outputPtr[6 * numBoxes + highConfBoxIdx];
        float classScore3 = outputPtr[7 * numBoxes + highConfBoxIdx];
        float classScore4 = outputPtr[8 * numBoxes + highConfBoxIdx];
        float classScore5 = outputPtr[9 * numBoxes + highConfBoxIdx];
        CString msg;
        msg.Format(_T("最高置信度框[%d]:\n  cx=%.2f, cy=%.2f, w=%.2f, h=%.2f\n  类别分数: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n  置信度: %.4f, 类别: %d"),
            highConfBoxIdx, cx, cy, w, h,
            classScore0, classScore1, classScore2, classScore3, classScore4, classScore5,
            maxConf, maxConfClass);
        MessageBox(NULL, msg, _T("后处理调试"), MB_OK);
    }
#endif

    // 应用NMS (与Python实现一致)
    return applyNMS(candidates, 0.45f);
}
cv::Mat MyTensorRT::postprocessOutput_Super(int batchSize)
{
    cv::Mat colored_depth;
    return colored_depth;
}
/**
 * @brief 专为深度估计模型设计的后处理函数。
 *        它将推理结果进行min-max归一化，并应用颜色图生成可视化结果。
 * @return 返回一个8位的3通道彩色深度图 (cv::Mat, BGR格式)，可以直接显示或保存。
 */
cv::Mat MyTensorRT::postprocessOutput_Depth(int batchSize)
{
    // `inference()`函数已将GPU输出结果拷贝到了CPU的 m_outputDataFP32 成员变量中。
    // 我们直接使用这个成员变量进行后处理。
    if (m_outputDataFP32.empty()) {
        throw std::runtime_error("Output data is empty. Did you run inference() first?");
    }

    // 1. 获取输出维度
    nvinfer1::Dims outDims = getOutputDims(0); // 得到 [1, 518, 616]

    // 2. 修改检查逻辑，以适应3维输出
    if (outDims.nbDims != 3) {
        // 更新报错信息，使其更准确
        throw std::runtime_error("Expected a 3D output format [N, H, W] for the depth map.");
    }

    // 3. 从3维张量中正确地提取高和宽
    // 对于 [N, H, W] -> N在索引0, H在索引1, W在索引2
    const int outputHeight = outDims.d[1]; // H 现在在索引 1
    const int outputWidth = outDims.d[2];  // W 现在在索引 2

    // 1. 将模型输出的【小尺寸】深度图数据转换为 cv::Mat
    cv::Mat small_depth_map(outputHeight, outputWidth, CV_32FC1, m_outputDataFP32.data());

    // 2. 【关键新增步骤】将小尺寸深度图放大回【原始图像尺寸】
    //    我们使用 cv::INTER_LINEAR 作为 PyTorch 中 "bilinear" 的高效等效插值方法
    cv::Mat full_size_depth_map;
    cv::resize(small_depth_map, full_size_depth_map, m_lastOriginalImageSize, 0, 0, cv::INTER_LINEAR);

    // 3. 对【放大后】的全尺寸深度图进行归一化和着色
    cv::Mat normalized_depth_8u;
    cv::normalize(full_size_depth_map, normalized_depth_8u, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat colored_depth;
    cv::applyColorMap(normalized_depth_8u, colored_depth, cv::COLORMAP_INFERNO);

    return colored_depth;
}
std::vector<Detection> MyTensorRT::postprocessOutput(int batchSize)
{
    const std::vector<float>& outputData = m_outputDataFP32;

    if (outputData.empty()) {
        throw std::runtime_error("输出数据为空");
    }

    // 获取输出维度信息
    nvinfer1::Dims outDims = getOutputDims(0);
    const int numDetections = outDims.d[1];  // 检测框数量,25200
    const int elementsPerDetection = outDims.d[2]; // 每个检测框包含的元素数,6

    // 获取图像预处理参数，用于将检测框坐标映射回原始图像
    float originalWidth = static_cast<float>(m_lastPreprocessParams.originalSize.width);
    float originalHeight = static_cast<float>(m_lastPreprocessParams.originalSize.height);
    float scale = m_lastPreprocessParams.scaleRatio;// 缩放比例
    int dw = m_lastPreprocessParams.paddingOffset.x;// 水平填充
    int dh = m_lastPreprocessParams.paddingOffset.y;// 垂直填充

    // 存储所有符合置信度阈值的候选检测框
    std::vector<Detection> candidates;
    //遍历所有检测框
    for (int i = 0; i < numDetections; ++i) {
        // 提取当前检测框的输出数据
        const float* det = &outputData[i * elementsPerDetection];
        // 解析检测框信息：中心点坐标、宽高、目标置信度、类别置信度
        float x_center = det[0];
        float y_center = det[1];
        float w = det[2];
        float h = det[3];
        float objConf = det[4];// 目标存在的置信度
        float clsConf = det[5];// 类别置信度

        //转换为左上右下坐标格式(xyxy)
        float x1 = x_center - w / 2;
        float y1 = y_center - h / 2;
        float x2 = x_center + w / 2;
        float y2 = y_center + h / 2;

        // 计算最终置信度：目标置信度 × 类别置信度
        float confidence = objConf * clsConf;

        // 应用置信度阈值(0.45)过滤低置信度检测框
        if (confidence < 0.01f) continue;

        // 映射回原始图像坐标
        // 1. 去除填充
        x1 = (x1 - dw) / scale;
        y1 = (y1 - dh) / scale;
        x2 = (x2 - dw) / scale;
        y2 = (y2 - dh) / scale;

        // 2. 确保在图像范围内
        x1 = (std::max)(0.0f, (std::min)(x1, originalWidth - 1));
        y1 = (std::max)(0.0f, (std::min)(y1, originalHeight - 1));
        x2 = (std::max)(0.0f, (std::min)(x2, originalWidth - 1));
        y2 = (std::max)(0.0f, (std::min)(y2, originalHeight - 1));

        // 3. 计算宽高
        float width = x2 - x1;
        float height = y2 - y1;
        // 过滤无效检测框(宽高必须为正)
        if (width <= 0 || height <= 0) continue;

        // 4. 创建检测对象
        Detection candidate;
        candidate.x = x1;// 左上角x坐标
        candidate.y = y1;// 左上角y坐标
        candidate.width = width;
        candidate.height = height;
        candidate.confidence = confidence;// 置信度
        candidate.classId = 0;  // 单类别模型,类别ID固定为0

        candidates.push_back(candidate);
    }

    // 应用非极大值抑制(NMS)算法，去除重叠的检测框(IOU阈值0.45)
    return applyNMS(candidates, 0.45f);
}

std::vector<Detection> MyTensorRT::applyNMS(const std::vector<Detection>& candidates, float iouThreshold)
{
 /**  if (candidates.empty()) return {};

    // 创建索引数组并初始化为[0,1,2,...]
    std::vector<size_t> indices(candidates.size());
    std::iota(indices.begin(), indices.end(), 0);
    // 按置信度降序排序，置信度高的检测框优先保留
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return candidates[a].confidence > candidates[b].confidence;
        });
    // 标记数组，记录每个检测框是否被抑制
    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<Detection> detections;
    // 遍历排序后的检测框
    for (size_t i = 0; i < indices.size(); ++i) {
        // 如果当前检测框已被抑制，则跳过
        if (suppressed[indices[i]]) continue;
        // 保留当前检测框（置信度最高）
        detections.push_back(candidates[indices[i]]);
        // 遍历剩余检测框，计算与当前保留框的IOU
        for (size_t j = i + 1; j < indices.size(); ++j) {
            if (suppressed[indices[j]]) continue;
            // 计算IOU(交并比)
            float iou = calculateIOU(candidates[indices[i]], candidates[indices[j]]);
            // 如果IOU超过阈值，抑制当前检测框
            if (iou > iouThreshold + 1e-6) {
                suppressed[indices[j]] = true;
            }
        }
    }

    return detections;*/
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    for (const auto& det : candidates) {
        boxes.emplace_back(det.x, det.y, det.width, det.height);
        scores.push_back(det.confidence);
        classIds.push_back(det.classId);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, 0.5f, 0.5, indices);  

    std::vector<Detection> result;
    for (int i : indices) {
        result.push_back(candidates[i]);
    }
    return result;
}

float MyTensorRT::calculateIOU(const Detection& a, const Detection& b)
{
    // 计算两个检测框的右下坐标
    float a_x2 = a.x + a.width;
    float a_y2 = a.y + a.height;
    float b_x2 = b.x + b.width;
    float b_y2 = b.y + b.height;
    // 计算交集区域的边界
    float interLeft = (std::max)(a.x, b.x);
    float interTop = (std::max)(a.y, b.y);
    float interRight = (std::min)(a_x2, b_x2);
    float interBottom = (std::min)(a_y2, b_y2);
    // 如果没有交集，IOU为0
    if (interRight < interLeft || interBottom < interTop)
        return 0.0f;
    // 计算交集面积
    float interArea = (interRight - interLeft) * (interBottom - interTop);
    // 计算并集面积 = 两个检测框面积之和 - 交集面积
    float unionArea = a.width * a.height + b.width * b.height - interArea;
    // 计算IOU = 交集面积 / 并集面积
    return (unionArea > 1e-6) ? (interArea / unionArea) : 0.0f;
}

std::string MyTensorRT::getClassName(int classId) const
{
    // 单类别模型
    std::string name = "other";
    if (classId == 0)
    {
        name = "gate";
    }
    else if (classId == 1)
    {
        name = "vehicle";
    }
    else if (classId == 2) 
    {
        name = "people";
    }
    else if (classId == 3)
    {
        name = "weapon";
    }
    else if (classId == 4)
    {
        name = "dog";
    }
    else if (classId == 5)
    {
        name = "airplane";
    }
    return name;  
}

void MyTensorRT::warmup(int numIterations)
{
    try {
        std::cout << "开始GPU预热..." << std::endl;

        // 获取模型输入尺寸
        nvinfer1::Dims dims = getInputDims();
        const int modelHeight = dims.d[2];
        const int modelWidth = dims.d[3];

        // 创建一个空白图像用于预热
        cv::Mat warmupImage = cv::Mat::zeros(modelHeight, modelWidth, CV_8UC3);

        // 执行多次预热迭代
        for (int i = 0; i < numIterations; ++i) {
            // 预处理
            preprocessImage(warmupImage);

            // 推理
            inference(1);

            // 后处理
            std::vector<Detection> dummy = postprocessOutputYOLOV8(1);

            // 确保所有操作完成
            cudaDeviceSynchronize();
        }
        // 执行多次预热迭代
        for (int i = 0; i < numIterations; ++i) {
            // 预处理
            preprocessImage_Depth(warmupImage);

            // 推理
            inference(1);

            // 后处理
            cv::Mat depth = postprocessOutput_Depth(1);

            // 确保所有操作完成
            cudaDeviceSynchronize();
        }
        // 执行多次预热迭代
        for (int i = 0; i < numIterations; ++i) {
            // 预处理
            preprocessImage_Super(warmupImage);

            // 推理
            inference(1);

            // 后处理
            cv::Mat depth = postprocessOutput_Super(1);

            // 确保所有操作完成
            cudaDeviceSynchronize();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "GPU预热失败: " << e.what() << std::endl;
        throw;
    }
}