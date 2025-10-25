// GrabDemoDlg.h : header file
//
#include <stdio.h>  
#include "SapClassBasic.h"
#include "SapClassGui.h"

// GPU 头文件
#include <stdio.h>  
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include "afxwin.h"
#include "DistortionCailbration.h"
#include "MyTensorRT.h"
#include "tracker_utils.hpp"
// GPU 函数声明
extern "C"
cudaError_t Image_Solution(unsigned short *Image, int Length, int Width, float *pTP_Gain, float *pTP_Bias, int TP_On,int Blind_On, unsigned short *pBlind_Ram, int Histogram_On);  //  pTP_Gain,pTP_Bias指向两点矫正参数----TP_On表示是否开启两点矫正
//直方图增强
extern "C" 
cudaError_t GPU_Histogram_Enhancement(unsigned short *Image, unsigned int *Histogram,float * Histogram_Float,unsigned short *dev_img, unsigned int *dev_Histogram,float* dev_Histogram_float, int Length, int Width);
// 两点矫正函数
extern "C"
cudaError_t GPU_TwoPoint_Correction(unsigned short *Image, unsigned short *dev_img, float* dev_pTP_Gain, float* dev_pTP_Bias, int Length, int Width);
//盲元矫正函数
extern "C"
cudaError_t GPU_Blind_Correction(unsigned short *Image, unsigned short *dev_img, unsigned short *dev_pBlind_Ram, int Length, int Width);
#if !defined(AFX_GRABDEMODLG_H__82BFE149_F01E_11D1_AF74_00A0C91AC0FB__INCLUDED_)
#define AFX_GRABDEMODLG_H__82BFE149_F01E_11D1_AF74_00A0C91AC0FB__INCLUDED_

#if _MSC_VER >= 1000
#pragma once
#endif // _MSC_VER >= 1000


/////////////////////////////////////////////////////////////////////////////
// CGrabDemoDlg dialog

class CGrabDemoDlg : public CDialog, public CImageExWndEventHandler // 继承自CDialog和CImageExWndEventHandler
{
	// Construction
public:
	DistortionCailbration distortionCailbration; // 畸变校准对象
	CGrabDemoDlg(CWnd* pParent = NULL);	// 标准构造函数

	BOOL CreateObjects(); // 创建对象
	BOOL DestroyObjects(); // 销毁对象
	void UpdateMenu(); // 更新菜单
	static void XferCallback(SapXferCallbackInfo* pInfo); // 数据传输回调函数
	static void SignalCallback(SapAcqCallbackInfo* pInfo); // 信号回调函数
	void GetSignalStatus(); // 获取信号状态
	void GetSignalStatus(SapAcquisition::SignalStatus signalStatus); // 获取指定信号状态
	void PixelChanged(int x, int y); // 像素改变事件处理
	void FPGA_Send(); // 发送数据到FPGA
	void FPGA_Receive(); // 从FPGA接收数据
 // Dialog Data
	 //{{AFX_DATA(CGrabDemoDlg)
	enum { IDD = IDD_GRABDEMO_DIALOG }; // 对话框资源ID
	float       m_BufferFrameRate; // 缓冲帧率
	CStatic	m_statusWnd; // 状态窗口控件
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGrabDemoDlg)
protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV支持
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON		m_hIcon; // 图标句柄
	CString  m_appTitle; // 应用程序标题

	CString m_cstrWorkPath;    // 工作路径，存放采集数据和计算中间数据
	int m_imageHeight;             // 图像高度
	int m_imageWidth;              // 图像宽度

	CImageExWnd		m_ImageWnd; // 显示图像的窗口
	SapAcquisition* m_Acq; // 采集对象指针
	SapBuffer* m_Buffers; // 缓冲区对象指针
	SapTransfer* m_Xfer; // 传输对象指针
	SapView* m_View; // 视图对象指针
	//SapView        *m_ViewProcessed;

	static CGrabDemoDlg* m_DlgPointer;  // 回调函数访问非静态成员
	static MyTensorRT* m_super_tensorRT; // TensorRT超分辨率模型指针
	static MyTensorRT* m_detect_tensorRT; // TensorRT检测模型指针
	static MyTensorRT* m_depth_tensorRT; // TensorRT深度估计模型指针
	static MyTensorRT* m_track_tensorRT; // TensorRT深度估计模型指针
	BOOL m_IsSignalDetected;   // TRUE if camera signal is detected 相机信号是否被检测到

	 // Generated message map functions
	 //{{AFX_MSG(CGrabDemoDlg)
	virtual BOOL OnInitDialog(); // 初始化对话框
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam); // 系统命令消息
	afx_msg void OnPaint(); // 绘制消息
	afx_msg HCURSOR OnQueryDragIcon(); // 查询拖动图标
	afx_msg void OnDestroy(); // 销毁消息
	afx_msg void OnSize(UINT nType, int cx, int cy); // 大小变化消息
	afx_msg void OnSnap(); // 快照按钮点击消息
	afx_msg void OnGrab(); // 抓取按钮点击消息
	afx_msg void OnFreeze(); // 冻结按钮点击消息
	afx_msg void OnGeneralOptions(); // 通用选项按钮点击消息
	afx_msg void OnAreaScanOptions(); // 区域扫描选项按钮点击消息
	afx_msg void OnLineScanOptions(); // 线扫描选项按钮点击消息
	afx_msg void OnCompositeOptions(); // 复合选项按钮点击消息
	afx_msg void OnLoadAcqConfig(); // 加载采集配置按钮点击消息
	afx_msg void OnImageFilterOptions(); // 图像滤波选项按钮点击消息
	afx_msg void OnBufferOptions(); // 缓冲区选项按钮点击消息
	afx_msg void OnViewOptions(); // 视图选项按钮点击消息
	afx_msg void OnFileLoad(); // 文件加载按钮点击消息
	afx_msg void OnFileNew(); // 新建文件按钮点击消息
	afx_msg void OnFileSave(); // 文件保存按钮点击消息
	afx_msg void OnExit(); // 退出按钮点击消息
	afx_msg void OnEndSession(BOOL bEnding); // 结束会话消息
	afx_msg BOOL OnQueryEndSession(); // 查询结束会话消息
	afx_msg void OnKillfocusBufferFrameRate(void); // BufferFrameRate失去焦点时的消息

	afx_msg void OnTimer(UINT_PTR nIDEvent); // 定时器消息

	 //}}AFX_MSG
	DECLARE_MESSAGE_MAP()

public:
	afx_msg void OnBnClickedSavemulti(); // "savemulti"按钮点击消息

	afx_msg void OnBnClickedSaveTiming(); // "保存计时"按钮点击消息
	afx_msg void OnBnClickedTimingStop(); // "停止计时"按钮点击消息
	CButton GPU_State; // GPU状态按钮
	CButton TP_Correction; // 温漂校正按钮
	afx_msg void OnBnClickedTpCold(); // 温漂冷端校正按钮点击消息
	afx_msg void OnBnClickedTpHot(); // 温漂热端校正按钮点击消息
	CButton Blind_Correction; // 盲元校正按钮
	CButton H_Enhance; // 高增强按钮
	afx_msg void OnBnClickedHEnhance(); // 高增强按钮点击消息
	afx_msg void OnBnClickedGpu(); // GPU按钮点击消息

	afx_msg void OnBnClickedGetlow(); // 获取低值按钮点击消息
	afx_msg void OnBnClickedNetd(); // NETD按钮点击消息

	void CalculateNETD(int& flag, int Height, int Width); // 计算NETD

	afx_msg void OnBnClickedOk(); // OK按钮点击消息
	CComboBox m_Combo; // 组合框控件
	afx_msg void OnBnClickedOpen(); // 打开按钮点击消息
	afx_msg void OnBnClickedClose(); // 关闭按钮点击消息
	afx_msg void OnBnClickedHe(); // HE按钮点击消息
	// 下位机串口接收到数据的中断函数
	afx_msg void localEnlarge(int Height, int Widt); // 局部放大函数
	LRESULT CGrabDemoDlg::SerialRead(WPARAM, LPARAM); // 串口读取消息
	afx_msg void OnBnClickedGethigh(); // 获取高值按钮点击消息
	afx_msg void OnBnClickedBlindCorrection(); // 盲元校正按钮点击消息
	afx_msg void OnBnClickedTpCorrection(); // 温漂校正按钮点击消息
	afx_msg void OnBnClickedMedianFilter(); // 中值滤波按钮点击消息
	afx_msg void OnBnClickedtest(); // 测试按钮点击消息
	afx_msg void OnBnClickedTest(); // 另一个测试按钮点击消息
	afx_msg void OnBnClickedBpmap(); // BP映射按钮点击消息
	afx_msg void OnBnClickedTpmap(); // 温漂映射按钮点击消息
	afx_msg void OnBnClickedIntegral(); // 积分按钮点击消息
	afx_msg void OnBnClickedBitrate(); // 比特率按钮点击消息
	int mHeight; // 图像高度
	int mWidth; // 图像宽度
	int imageBits; // 图像比特数

	afx_msg void OnMouseMove(UINT nFlags, CPoint point); // 鼠标移动消息
	CComboBox Comb_Rate; // 组合框控件用于选择速率
	CComboBox Combe_I2CMode; // I2C模式组合框
	CComboBox Combe_I2CBitSet; // I2C比特设置组合框
	CComboBox Combe_I2C_TimeSet; // I2C时间设置组合框
	afx_msg void OnBnClickedChange1(); // 改变1按钮点击消息
	afx_msg void OnBnClickedChange2(); // 改变2按钮点击消息
	afx_msg void OnBnClickedChange3(); // 改变3按钮点击消息
	CButton PC_BilateralFilter; // 双边滤波按钮
	afx_msg void OnBnClickedNu(); // NU按钮点击消息
	afx_msg void OnBnClickedUpdate(); // 更新按钮点击消息
	afx_msg void CUDA_Algorithm(); // CUDA算法
	afx_msg void OnBnClickedWinchange(); // 窗口改变按钮点击消息
	afx_msg void OnBnClickedLocalEnlarge(); // 畸变矫正
	//afx_msg void OnStnClickedViewWnd(); // ViewWnd控件单击消息
	afx_msg void OnStnClickedCancel(); // Cancel控件单击消息
//	CImageExWnd m_ImageWnd2;
	afx_msg void OnStnClickedViewWnd2(); // ViewWnd2控件单击消息
private:
	cv::Mat extractAndResizeCenterView(const cv::Mat& sourceImage); // 提取并调整中心视图大小
	TrackerUtils* m_tracker = nullptr;   // 跟踪模块指针
	TrackingResult m_trackResult;        // 存储每帧跟踪结果
	bool m_enableTracking = false;       // 是否开启跟踪
public:
	afx_msg void OnBnClickedCheck1(); // 目标检测
	afx_msg void OnBnClickedDepthCheck(); // 深度检
	afx_msg void OnBnClickedSuperCheck(); // 视角合成
	afx_msg void OnCbnSelchangeCombo1(); // Combo1选中项改变消息
	cv::Mat m_saiHost;   // 用于保存超分辨率结果					 
	CComboBox m_comboBox;  // 云，雨，雾，正常四选项选择
	// CGrabDemoDlg.h 中声明
private:
	void UpdateSavePath(); // 更新 m_cstrWorkPath
public:
	afx_msg void OnBnClickedTrackCheck();
	afx_msg void OnEnChangeframes();
};
//{{AFX_INSERT_LOCATION}}
// Microsoft Developer Studio will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GRABDEMODLG_H__82BFE149_F01E_11D1_AF74_00A0C91AC0FB__INCLUDED_)
