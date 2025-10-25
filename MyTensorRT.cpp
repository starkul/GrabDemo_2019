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
void convert_HWC_to_NCHW_and_normalize(
    const cv::cuda::GpuMat& src, // 输入 (HWC)
    float* dst,                  // 输出 (NCHW)
    int width,
    int height,
    const float* mean,
    const float* std,
    cudaStream_t stream
);
MyTensorRT::MyTensorRT(const std::string& enginePath, bool enableFP16) :  m_enableFP16(enableFP16)
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

void MyTensorRT::preprocessImage_Detect(const cv::Mat& inputImage)
{
    cv::Mat processedImage;
    if (inputImage.channels() == 1) {
        // 如果输入是单通道灰度图，自动转换为3通道BGR图。
        cv::cvtColor(inputImage, processedImage, cv::COLOR_GRAY2BGR);
    }
    else if (inputImage.channels() == 3) {
        // 如果输入已经是3通道图，直接使用。
        // 用 clone() 确保我们有一个独立的副本，避免后续操作意外修改原始数据。
        processedImage = inputImage.clone();
    }
    // 保存原始尺寸
    m_lastPreprocessParams.originalSize = processedImage.size();
    // 1. 获取模型期望的输入维度 (640x640)
    nvinfer1::Dims dims = getInputDims();
    const int modelHeight = dims.d[2];  // 640
    const int modelWidth = dims.d[3];   // 640

    cv::cuda::GpuMat gpu_input;
    // 1. 上传到GPU
    if (gpu_input.empty()) {
        gpu_input = cv::cuda::GpuMat(processedImage);
    }
    else {
        gpu_input.upload(processedImage); // 优化上传操作
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
    //const float dummy_mean[3] = { 0.485f, 0.456f, 0.406f };
    //const float dummy_std[3] = { 0.229f, 0.224f, 0.225f };
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
void MyTensorRT::preprocessImage_Super(const cv::Mat& lrImage)
{
    // 1. 获取模型期望的输入维度 (例如 1x3xH_lrxW_lr)
    nvinfer1::Dims dims = getInputDims();
    if (dims.nbDims != 4) {
        throw std::runtime_error("Super-Resolution model requires a 4D input tensor (NCHW).");
    }
    const int modelHeight = dims.d[2];
    const int modelWidth = dims.d[3];

    // 2. 检查输入图像尺寸是否与模型期望的输入尺寸匹配
    //    超分模型通常对输入尺寸有严格要求。如果不匹配，可以选择报错或进行缩放/填充。
    //    这里我们先采用严格的报错策略。
    if (lrImage.rows != modelHeight || lrImage.cols != modelWidth) {
        std::string errMsg = "Input image size (" + std::to_string(lrImage.cols) + "x" + std::to_string(lrImage.rows) +
            ") does not match model's required input size (" + std::to_string(modelWidth) + "x" + std::to_string(modelHeight) + ").";
        throw std::runtime_error(errMsg);
    }

    // --- 使用 GPU 加速预处理 ---
    cv::cuda::GpuMat gpu_input(lrImage);

    // 3. BGR -> RGB
    cv::cuda::GpuMat gpu_rgb;
    cv::cuda::cvtColor(gpu_input, gpu_rgb, cv::COLOR_BGR2RGB);

    // 4. uint8 [0, 255] -> float32 [0, 1]
    cv::cuda::GpuMat gpu_normalized;
    gpu_rgb.convertTo(gpu_normalized, CV_32FC3, 1.0 / 255.0);

    // 5. HWC -> NCHW (不进行标准化)
       //    您的核函数调用方式完全符合需求，我们直接复用！
    const float dummy_mean[3] = { 0.0f, 0.0f, 0.0f };
    const float dummy_std[3] = { 1.0f, 1.0f, 1.0f };

    // 调用您现有的灵活核函数
    convertHWCtoNCHW(gpu_normalized, m_inputBuffer, m_stream, dummy_mean, dummy_std);
    // 如果没有专用核函数，可以手动在CPU完成，但这会慢一些
    // 参考您的深度估计代码中的CPU部分即可实现

    cudaStreamSynchronize(m_stream);
}
/**
 * @brief (已更新) 专为光场模型设计的预处理函数。
 *        它接收一张包含 AxA 网格视图的大图，在CPU上完成分割和裁剪，
 *        然后利用GPU加速后续的格式转换，最终生成模型所需的NCHW输入。
 * @param nine_grid_image 包含九宫格视图的输入大图 (cv::Mat, BGR格式)。
 * @param angular_resolution 角度分辨率，对于九宫格就是 3。
 */
void MyTensorRT::preprocessImage_LightField(const cv::Mat& nine_grid_image, int angular)
{
    // --- 参数检查 ---
    if (nine_grid_image.empty()) {
        throw std::runtime_error("Input image is empty.");
    }
    if (angular != 3) {
        throw std::runtime_error("This function is designed for a 3x3 angular grid.");
    }

    // ====================================================================
    // 阶段一：在 CPU 上分割、裁剪并进行预处理
    // ====================================================================

    // 1. 定义模型期望的输入尺寸
    const int targetHeight = 142;
    const int targetWidth = 170;
    const int grid_rows = angular;
    const int grid_cols = angular;

    // 2. 计算每个子图的理论尺寸
    const int cell_width = nine_grid_image.cols / grid_cols;
    const int cell_height = nine_grid_image.rows / grid_rows;

    // 3. 准备一个临时的浮点图像容器
    // 这个容器将按行主序 (row-major) 存放9个预处理好的patch
    std::vector<cv::Mat> processed_patches;
    processed_patches.reserve(angular * angular);

    for (int u = 0; u < grid_rows; ++u) {
        for (int v = 0; v < grid_cols; ++v) {
            cv::Rect cell_rect(v * cell_width, u * cell_height, cell_width, cell_height);
            cv::Mat original_patch = nine_grid_image(cell_rect);

            int startRow = (original_patch.rows - targetHeight) / 2;
            int startCol = (original_patch.cols - targetWidth) / 2;
            if (startRow < 0 || startCol < 0) {
                throw std::runtime_error("Target crop size is larger than the grid cell size.");
            }
            cv::Rect center_crop_rect(startCol, startRow, targetWidth, targetHeight);

            // 直接将裁剪出的灰度图存入（假设输入已是灰度图）
            // .clone() 确保我们得到的是一个独立的Mat副本
            processed_patches.push_back(original_patch(center_crop_rect).clone());
        }
    }
    // ====================================================================
    // 阶段二：将9个小图拼接成一张大图 (426x510)
    // ====================================================================

    // 1. 先将每一行的3个小图水平拼接起来
    std::vector<cv::Mat> rows_stitched;
    for (int u = 0; u < grid_rows; ++u) {
        std::vector<cv::Mat> current_row_patches;
        for (int v = 0; v < grid_cols; ++v) {
            current_row_patches.push_back(processed_patches[u * grid_cols + v]);
        }
        cv::Mat stitched_row;
        cv::hconcat(current_row_patches, stitched_row); // 水平拼接
        rows_stitched.push_back(stitched_row);
    }

    // 2. 再将拼接好的3行垂直拼接起来，形成最终的大图
    cv::Mat final_stitched_image;
    cv::vconcat(rows_stitched, final_stitched_image);
    // 可选：检查拼接后图像的尺寸是否正确
    if (final_stitched_image.rows != (targetHeight * angular) || final_stitched_image.cols != (targetWidth * angular)) {
        throw std::runtime_error("Final stitched image size is incorrect.");
    }

    // ====================================================================
    // 阶段三：对最终的大图进行归一化并准备输入Buffer
    // ====================================================================

    // 1. 类型转换并归一化到 [0, 1]
    cv::Mat float_image;
    // 假设输入已经是灰度图，如果不是，需要先转灰度
    if (final_stitched_image.channels() == 3) {
        cv::cvtColor(final_stitched_image, final_stitched_image, cv::COLOR_BGR2GRAY);
    }
    final_stitched_image.convertTo(float_image, CV_32F, 1.0 / 255.0);

    // 2. 获取模型输入维度并检查
    nvinfer1::Dims dims = getInputDims();
    if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 1 ||
        dims.d[2] != (targetHeight * angular) || dims.d[3] != (targetWidth * angular)) {
        throw std::runtime_error("Warning: Model input dimensions might not match the expected [1, 1, 426, 510] format.");
    }
    cv::Mat continuous_float_image = float_image.clone();
    // 优化：直接从连续的Mat数据区将数据拷贝到GPU，无需中间的CPU vector
    cudaMemcpyAsync(m_inputBuffer, continuous_float_image.data,
        continuous_float_image.total() * continuous_float_image.elemSize(),
        cudaMemcpyHostToDevice, m_stream);
    //// 3. 将数据拷贝到输入Buffer
    //std::vector<float> cpu_input_buffer(volume(dims));
    //// 因为 float_image 是一个完整的、连续的Mat，可以直接拷贝
    //memcpy(cpu_input_buffer.data(), float_image.data, cpu_input_buffer.size() * sizeof(float));

    //// ====================================================================
    //// 阶段四：将准备好的数据上传到 GPU
    //// ====================================================================
    //
    //cudaMemcpyAsync(m_inputBuffer, cpu_input_buffer.data(), cpu_input_buffer.size() * sizeof(float), cudaMemcpyHostToDevice, m_stream);
    cudaStreamSynchronize(m_stream);
}
/**
 * @brief 专为深度估计模型设计的预处理函数。
 *        它将输入图像直接缩放、转换颜色空间、归一化，并转换为CHW格式。
 * @param inputImage 输入的原始图像 (OpenCV Mat, BGR格式)。
 */


void MyTensorRT::preprocessImage_Depth(const cv::Mat& inputImage)
{
    cv::Mat processedImage;
    if (inputImage.channels() == 1) {
        // 如果输入是单通道灰度图，自动转换为3通道BGR图。
        cv::cvtColor(inputImage, processedImage, cv::COLOR_GRAY2BGR);
    }
    else if (inputImage.channels() == 3) {
        // 如果输入已经是3通道图，直接使用。
        // 用 clone() 确保我们有一个独立的副本，避免后续操作意外修改原始数据。
        processedImage = inputImage.clone();
    }
    // 1. 记录原始图像尺寸，用于后处理的放大
    m_lastOriginalImageSize = processedImage.size();
    // 1. 获取模型期望的输入维度 (例如 1x3x518x518)
    nvinfer1::Dims dims = getInputDims();
    if (dims.nbDims != 4) {
        throw std::runtime_error("Depth model requires a 4D input tensor (NCHW).");
    }
    const int modelHeight = dims.d[2];
    const int modelWidth = dims.d[3];

    // 上传图像到 GPU
    cv::cuda::GpuMat gpu_input;
    if (gpu_input.empty()) {
        gpu_input = cv::cuda::GpuMat(processedImage);
    }
    else {
        gpu_input.upload(processedImage);
    }

    int original_w = gpu_input.cols;
    int original_h = gpu_input.rows;

    // 3. 【关键】计算保持宽高比的缩放尺寸
    float scale = (std::min)(static_cast<float>(modelWidth) / original_w, static_cast<float>(modelHeight) / original_h);
    int new_w = static_cast<int>(original_w * scale);
    int new_h = static_cast<int>(original_h * scale);

    // 4. 按计算出的新尺寸进行【等比例】缩放
    cv::cuda::GpuMat gpu_resized;
    cv::cuda::resize(gpu_input, gpu_resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    // 5. 创建一个最终尺寸的黑色“画布” (padding)
    cv::cuda::GpuMat gpu_padded(modelHeight, modelWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    // 6. 将等比例缩放后的图像“粘贴”到画布的左上角
    cv::cuda::GpuMat roi = gpu_padded(cv::Rect(0, 0, new_w, new_h));
    gpu_resized.copyTo(roi);

    // 7. 对【填充后】的完整图像进行后续处理
    //    (BGR->RGB)
    cv::cuda::GpuMat gpu_rgb;
    cv::cuda::cvtColor(gpu_padded, gpu_rgb, cv::COLOR_BGR2RGB);

    // 8. 归一化并转换为浮点类型
    cv::cuda::GpuMat gpu_float;
    gpu_rgb.convertTo(gpu_float, CV_32FC3, 1.0 / 255.0);

    // 9. 标准化 - 使用核函数直接处理
    const float mean[3] = { 0.485f, 0.456f, 0.406f };
    const float std[3] = { 0.229f, 0.224f, 0.225f };
    // 这个函数将替换掉你原来那整个 for 循环
    convert_HWC_to_NCHW_and_normalize(
        gpu_float,         // 输入的 GpuMat (CV_32FC3, HWC)
        m_inputBuffer,     // 最终的输出 buffer (float*, NCHW)
        modelWidth,
        modelHeight,
        mean,
        std,
        m_stream
    );
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
        //if (confidence < 0.25) continue;

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
    if (m_outputDataFP32.empty()) {
        throw std::runtime_error("Output data is empty. Did you run inference() first?");
    }

    // 1. 获取输出维度 (例如 1x3xH_hrxW_hr)
    nvinfer1::Dims outDims = getOutputDims(0);
    if (outDims.nbDims != 4) {
        throw std::runtime_error("Expected a 4D output format [N, C, H, W] for the SR map.");
    }
    const int outputHeight = outDims.d[2];
    const int outputWidth = outDims.d[3];
    // 检查通道数是否为1 (灰度图)
    const int outputChannels = outDims.d[1];
    if (outputChannels != 1) {
        throw std::runtime_error("Expected a single channel (grayscale) output, but got " + std::to_string(outputChannels) + " channels.");
    }
    cv::Mat output_image_float(outputHeight, outputWidth, CV_32FC1, m_outputDataFP32.data());

    // 3. 【核心修正】使用 normalize 将网络输出的实际动态范围拉伸到 [0, 255]
    // 这会自动找到最小值(比如0.332)和最大值(比如0.876)，
    // 然后将最小值映射为0，最大值映射为255，进行线性拉伸。
    cv::Mat final_image_uint8;
    cv::normalize(output_image_float, final_image_uint8, 0, 255, cv::NORM_MINMAX, CV_8U);

    // 因为我们返回的是 final_image_uint8，它有自己的数据拷贝，
    // 所以不用担心 output_image_float 指向的 m_outputDataFP32 数据在后续被覆盖的问题。
    return final_image_uint8;
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
    cv::dnn::NMSBoxes(boxes, scores, 0.30f, 0.70f, indices);  

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
        name = "people";
    }
    else if (classId == 1)
    {
        name = "tank";
    }
    else if (classId == 2)
    {
        name = "airplane";
    }
    else if (classId == 3)
    {
        name = "vehicle";
    }
    return name;
}
// 新增方法的实现
void MyTensorRT::setModelType(ModelType type)
{
    m_modelType = type;
}
void MyTensorRT::warmup(int numIterations)
{
    try {
        std::cout << "开始GPU预热..." << std::endl;
        // 检查模型类型是否已设置
        if (m_modelType == ModelType::Unknown) {
            throw std::runtime_error("预热失败: 未设置模型类型。请在使用warmup前调用setModelType()。");
        }
        // 获取模型输入尺寸
        nvinfer1::Dims dims = getInputDims();
        const int modelHeight = dims.d[2];
        const int modelWidth = dims.d[3];

        // 创建一个空白图像用于预热
        cv::Mat warmupImage = cv::Mat::zeros(modelHeight, modelWidth, CV_8UC3);
        //注意：深度估计和目标检测的输入都是三通道的，而光场超分的输入是单通道的
        for (int i = 0; i < numIterations; ++i) {
            // 使用 switch 根据模型类型调用正确的流程
            switch (m_modelType) {
            case ModelType::ObjectDetection_YOLOv8:
                preprocessImage_Detect(warmupImage);
                inference(1);
                postprocessOutputYOLOV8(1);
                break;

            case ModelType::DepthEstimation_DepthAnything:
                preprocessImage_Depth(warmupImage);
                inference(1);
                postprocessOutput_Depth(1);
                break;

            case ModelType::SuperResolution_IINet:
                // 注意：光场超分的输入尺寸可能与dims不符，我们使用专门的预热图
                preprocessImage_LightField(warmupImage, 3);
                inference(1);
                postprocessOutput_Super(1);
                break;

                // default case 已在函数开头检查，此处可以省略
            }

            // 确保所有CUDA操作完成
            cudaDeviceSynchronize();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "GPU预热失败: " << e.what() << std::endl;
        throw;
    }
}
