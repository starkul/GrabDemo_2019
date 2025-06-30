// GrabDemoDlg.cpp : implementation file
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>  
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <stdio.h>  
#include "GrabDemo.h"
#include "GrabDemoDlg.h"

#include "math.h"

//#include "cuda_runtime.h"  
//#include "device_launch_parameters.h"  
//#include <opencv2/>

// GPU 头文件
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  

// 与下位机通信
#include "IRCMD_COM/SerialThread.h"
#include "IRCMD_COM/ComFrame.h"


// I2C通信exe服务头文件
#include "E:\INLF\QZB\串口通信\USB转I2C通信\ATL_test\ATLCOMProject\ATLCOMProject\ATLCOMProject_i.h"
#include "E:\INLF\QZB\串口通信\USB转I2C通信\ATL_test\ATLCOMProject\ATLCOMProject\ATLCOMProject_i.c"

//  Float32 转 float16
#include "Float32_16.h"

using namespace cv;
using namespace cv::cuda;
using namespace std;
//#pragma comment(lib,"cublas.lib");

#pragma comment(lib, "opencv_world455.lib")
/*
#pragma comment(lib, "opencv_aruco450d.lib")
#pragma comment(lib, "opencv_bgsegm450d.lib")
#pragma comment(lib, "opencv_bioinspired450d.lib")
#pragma comment(lib, "opencv_calib3d450d.lib")
#pragma comment(lib, "opencv_ccalib450d.lib")
#pragma comment(lib, "opencv_core450d.lib")
#pragma comment(lib, "opencv_cudaarithm450d.lib")
#pragma comment(lib, "opencv_cudabgsegm450d.lib")
#pragma comment(lib, "opencv_cudacodec450d.lib")
#pragma comment(lib, "opencv_cudafeatures2d450d.lib")
#pragma comment(lib, "opencv_cudafilters450d.lib")
#pragma comment(lib, "opencv_cudaimgproc450d.lib")
#pragma comment(lib, "opencv_cudalegacy450d.lib")
#pragma comment(lib, "opencv_cudaobjdetect450d.lib")
#pragma comment(lib, "opencv_cudaoptflow450d.lib")
#pragma comment(lib, "opencv_cudastereo450d.lib")
#pragma comment(lib, "opencv_cudawarping450d.lib")
#pragma comment(lib, "opencv_cudev450d.lib")
#pragma comment(lib, "opencv_datasets450d.lib")
#pragma comment(lib, "opencv_dnn450d.lib")
#pragma comment(lib, "opencv_dnn_objdetect450d.lib")
#pragma comment(lib, "opencv_dnn_superres450d.lib")
#pragma comment(lib, "opencv_dpm450d.lib")
#pragma comment(lib, "opencv_features2d450d.lib")
#pragma comment(lib, "opencv_flann450d.lib")
#pragma comment(lib, "opencv_fuzzy450d.lib")
#pragma comment(lib, "opencv_gapi450d.lib")
#pragma comment(lib, "opencv_hfs450d.lib")
#pragma comment(lib, "opencv_highgui450d.lib")
#pragma comment(lib, "opencv_imgcodecs450d.lib")
#pragma comment(lib, "opencv_imgproc450d.lib")
#pragma comment(lib, "opencv_img_hash450d.lib")
#pragma comment(lib, "opencv_intensity_transform450d.lib")
#pragma comment(lib, "opencv_line_descriptor450d.lib")
#pragma comment(lib, "opencv_mcc450d.lib")
#pragma comment(lib, "opencv_ml450d.lib")
#pragma comment(lib, "opencv_objdetect450d.lib")
#pragma comment(lib, "opencv_optflow450d.lib")
#pragma comment(lib, "opencv_phase_unwrapping450d.lib")
#pragma comment(lib, "opencv_photo450d.lib")
#pragma comment(lib, "opencv_plot450d.lib")
#pragma comment(lib, "opencv_quality450d.lib")
#pragma comment(lib, "opencv_rapid450d.lib")
#pragma comment(lib, "opencv_reg450d.lib")
#pragma comment(lib, "opencv_rgbd450d.lib")
#pragma comment(lib, "opencv_saliency450d.lib")
#pragma comment(lib, "opencv_shape450d.lib")
#pragma comment(lib, "opencv_stereo450d.lib")
#pragma comment(lib, "opencv_structured_light450d.lib")r
#pragma comment(lib, "opencv_superres450d.lib")
#pragma comment(lib, "opencv_surface_matching450d.lib")
#pragma comment(lib, "opencv_text450d.lib")
#pragma comment(lib, "opencv_tracking450d.lib")
#pragma comment(lib, "opencv_video450d.lib")
#pragma comment(lib, "opencv_videoio450d.lib")
#pragma comment(lib, "opencv_videostab450d.lib")
#pragma comment(lib, "opencv_ximgproc450d.lib")
#pragma comment(lib, "opencv_xobjdetect450d.lib")
#pragma comment(lib, "opencv_xphoto450d.lib")
*/

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

//// GPU 函数声明
//extern "C"
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

CGrabDemoDlg *CGrabDemoDlg::m_DlgPointer = NULL; //静态对象指针,需要提前初始化
MyTensorRT* CGrabDemoDlg::m_tensorRT = nullptr;
unsigned short rImage[1280*1024] = { 0 };   //GPU处理的图像数据
unsigned short *Image = rImage;

unsigned char rImage_uchar[1280 * 1024] = { 0 };   //GPU处理的图像数据
unsigned char *Image_uchar = rImage_uchar;

unsigned char mImage[640 * 512] = { 0 };   //局部放大的图像数据
unsigned char *mimage = mImage;

unsigned int rHistogram[65536] = {0};  //用于CPU处理直方图
unsigned int *Histogram = rHistogram;
float rHistogram_Float[65536] = { 0 };  //用于CPU处理直方图
float *Histogram_Float = rHistogram_Float;

unsigned short rCold_Ram[1280 * 1024] = { 0 };   //低温本底的图像数据
unsigned short *Cold_Ram = rCold_Ram;

unsigned short rHot_Ram[1280 * 1024] = { 20000 };   //高温本底的图像数据
unsigned short *Hot_Ram = rHot_Ram;

float rTP_Gain[1280 * 1024] = { 0 };		//两点矫正的各个像素增益
float *TP_Gain = rTP_Gain;

float rTP_Bias[1280 * 1024] = { 0 };   //两点矫正的各个像素偏移
float *TP_Bias = rTP_Bias; 

unsigned short rBlind_Ram[1280 * 1024] = { 0 };   //盲元矫正表
unsigned short *pBlind_Ram = rBlind_Ram;

//--------初始化GPU 指针 -----------
unsigned short *dev_img = 0;   //GPU 图像指针
float *dev_pTP_Gain = 0, *dev_pTP_Bias = 0; //GPU 两点矫正增益和偏移
unsigned short *dev_pBlind_Ram = 0;  //GPU 盲元矫正表
unsigned int *dev_Histogram = 0;   //GPU 直方图均衡
float *dev_Histogram_float = 0; //GPU 直方图均衡

//-------------NETD测量变量-------------
long Death_num = 0; long Hot_num = 0;
int NETD_frames = 50;   //配置参数
int NETD_K = 1; 
float T0 = 20; float T = 35;  // NETD: 第一次采集的温度  第二次采集的温度

int Flag_NETD = 0;   //是否执行NETD测量
float NETD_Vt[1280 * 1024] = { 0 };   //NETD
float NETD_Vt0[1280 * 1024] = { 0 };   //NETD
float NETD_Vn[1280 * 1024][50] = { 0 };   //NETD
float NETD_VnA[1280 * 1024] = { 0 };   //NETD 平均噪声值
int current = 0;  //表示当前读取的图像帧数

int frame_count;  // 用于记录帧频：每秒计数 

//下位机通信用的数据类型转换缓存器
unsigned char parameters[1024 * 1280 * 4 ];
unsigned char transfer_data[6]; //窗口数据
//int NETD_Vt = 0; int NETD_Vt0 = 0; int NETD_Vn = 0;
//-------------------------------------------------

//-----------串口超时重传时间间隔--------------
unsigned int Serial_Delay_Time = 10000;  //  xxx ms
cudaError_t cudaStatus;

//float rHistogram_Enhancement[65536] = { 0 };   //直方图增强表
//float *pHistogram_Enhancement = rHistogram_Enhancement;

int N_Saved_Frames = 0;
int Current_Saved = 0;
//------------------下位机通信部分初始化---------
// 串口状态 ： 打开：1
bool m_ComPortFlag = 0;
SerialThread st;

/////////////////////////////////////////////////////////////////////////////
// CAboutDlg dialog used for App About

class CAboutDlg : public CDialog
{
public:
   CAboutDlg();

   // Dialog Data
   //{{AFX_DATA(CAboutDlg)
   enum { IDD = IDD_ABOUTBOX };
   //}}AFX_DATA

   // ClassWizard generated virtual function overrides
   //{{AFX_VIRTUAL(CAboutDlg)
protected:
   virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
   //}}AFX_VIRTUAL

   // Implementation
protected:
   //{{AFX_MSG(CAboutDlg)
   //}}AFX_MSG
   DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
   //{{AFX_DATA_INIT(CAboutDlg)
   //}}AFX_DATA_INIT
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
   CDialog::DoDataExchange(pDX);
   //{{AFX_DATA_MAP(CAboutDlg)
   //}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
   //{{AFX_MSG_MAP(CAboutDlg)
   // No message handlers
   //}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGrabDemoDlg dialog

CGrabDemoDlg::CGrabDemoDlg(CWnd* pParent /*=NULL*/)
   : CDialog(CGrabDemoDlg::IDD, pParent)
	, mHeight(0)
	, mWidth(0)
{
   //{{AFX_DATA_INIT(CGrabDemoDlg)
   // NOTE: the ClassWizard will add member initialization here
   //}}AFX_DATA_INIT
   // Note that LoadIcon does not require a subsequent DestroyIcon in Win32
   m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

   m_Acq					= NULL;
   m_Buffers			= NULL;
   m_Xfer				= NULL;
   m_View            = NULL;
   //m_ViewProcessed = NULL;
   m_DlgPointer = this; //对象指针，构造函数式指向this
   //m_cstrWorkPath = "H:\\QZB\\Sapera\\Demos\\Classes\\Vc\\GrabDemo\\Data"; //定义数据存储位置
   //m_cstrWorkPath = "H:\\QZB\\Sapera\\Demos\\Classes\\Vc\\subData"; //定义数据存储位置
   m_cstrWorkPath = "E:\\INLF\\QZB\\Sapera\\Demos\\Classes\\Vc\\Raw20241112";//
   m_IsSignalDetected = TRUE;

}

void CGrabDemoDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGrabDemoDlg)
	DDX_Control(pDX, IDC_STATUS, m_statusWnd);
	DDX_Control(pDX, IDC_VIEW_WND, m_ImageWnd);
	DDX_Text(pDX, IDC_BUFFER_FRAME_RATE, m_BufferFrameRate);  // 获取帧频数量
															  //}}AFX_DATA_MAP
	DDV_MinMaxFloat(pDX, m_BufferFrameRate, 0, 9999);
	DDX_Control(pDX, IDC_GPU, GPU_State);
	DDX_Control(pDX, IDC_TP_Correction, TP_Correction);
	DDX_Control(pDX, IDC_Blind_Correction, Blind_Correction);
	DDX_Control(pDX, IDC_H_Enhance, H_Enhance);
	DDX_Control(pDX, FPGA_COMBO, m_Combo);
	DDX_Text(pDX, FPGA_Height, mHeight);
	DDV_MinMaxInt(pDX, mHeight, 0, 5000);
	DDX_Text(pDX, FPGA_Width, mWidth);
	DDV_MinMaxInt(pDX, mWidth, 0, 5000);
	DDX_Control(pDX, FPGA_COmbeRate, Comb_Rate);
	DDX_Control(pDX, I2C_Mode, Combe_I2CMode);
	DDX_Control(pDX, I2C_BitSet, Combe_I2CBitSet);
	DDX_Control(pDX, I2C_TImeSet, Combe_I2C_TimeSet);
	DDX_Control(pDX, PC_bilateralFilter, PC_BilateralFilter);
	//  DDX_Control(pDX, IDC_VIEW_WND2, m_ImageWnd2);
}

BEGIN_MESSAGE_MAP(CGrabDemoDlg, CDialog)
   //{{AFX_MSG_MAP(CGrabDemoDlg)
   ON_WM_SYSCOMMAND()
   ON_WM_PAINT()
   ON_WM_QUERYDRAGICON()
   ON_WM_DESTROY()
   ON_WM_SIZE()
   ON_BN_CLICKED(IDC_SNAP, OnSnap)
   ON_BN_CLICKED(IDC_GRAB, OnGrab)
   ON_BN_CLICKED(IDC_FREEZE, OnFreeze)
   ON_BN_CLICKED(IDC_GENERAL_OPTIONS, OnGeneralOptions)
   ON_BN_CLICKED(IDC_AREA_SCAN_OPTIONS, OnAreaScanOptions)
   ON_BN_CLICKED(IDC_LINE_SCAN_OPTIONS, OnLineScanOptions)
   ON_BN_CLICKED(IDC_COMPOSITE_OPTIONS, OnCompositeOptions)
   ON_BN_CLICKED(IDC_LOAD_ACQ_CONFIG, OnLoadAcqConfig)
   ON_BN_CLICKED(IDC_IMAGE_FILTER_OPTIONS, OnImageFilterOptions)
   ON_BN_CLICKED(IDC_BUFFER_OPTIONS, OnBufferOptions)
   ON_BN_CLICKED(IDC_VIEW_OPTIONS, OnViewOptions)
   ON_BN_CLICKED(IDC_FILE_LOAD, OnFileLoad)
   ON_BN_CLICKED(IDC_FILE_NEW, OnFileNew)
   ON_BN_CLICKED(IDC_FILE_SAVE, OnFileSave)
   ON_BN_CLICKED(IDC_EXIT, OnExit)
   ON_WM_ENDSESSION()
   ON_WM_QUERYENDSESSION()
   ON_BN_KILLFOCUS(IDC_BUFFER_FRAME_RATE, OnKillfocusBufferFrameRate) //
   //}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_SaveMulti, &CGrabDemoDlg::OnBnClickedSavemulti)

	ON_BN_CLICKED(IDC_Save_Timing, &CGrabDemoDlg::OnBnClickedSaveTiming)
	ON_WM_TIMER()  //添加定时器需要
	ON_BN_CLICKED(IDC_Timing_Stop, &CGrabDemoDlg::OnBnClickedTimingStop)

	ON_BN_CLICKED(IDC_TP_Cold, &CGrabDemoDlg::OnBnClickedTpCold)
	ON_BN_CLICKED(IDC_TP_Hot, &CGrabDemoDlg::OnBnClickedTpHot)

	ON_BN_CLICKED(IDC_H_Enhance, &CGrabDemoDlg::OnBnClickedHEnhance)
	ON_BN_CLICKED(IDC_GPU, &CGrabDemoDlg::OnBnClickedGpu)

	ON_BN_CLICKED(FPGA_GetLow, &CGrabDemoDlg::OnBnClickedGetlow)
	ON_BN_CLICKED(IDC_NETD, &CGrabDemoDlg::OnBnClickedNetd)
	ON_BN_CLICKED(I2C_OK, &CGrabDemoDlg::OnBnClickedOk)
	ON_BN_CLICKED(FPGA_Open, &CGrabDemoDlg::OnBnClickedOpen)
	ON_BN_CLICKED(FPGA_Close, &CGrabDemoDlg::OnBnClickedClose)
	ON_BN_CLICKED(FPGA_HE, &CGrabDemoDlg::OnBnClickedHe)

	ON_MESSAGE(ON_COM_RXCHAR, SerialRead)   //串口接收响应映射
	ON_BN_CLICKED(FPGA_GetHigh, &CGrabDemoDlg::OnBnClickedGethigh)
	ON_BN_CLICKED(FPGA_Blind_Correction, &CGrabDemoDlg::OnBnClickedBlindCorrection)
	ON_BN_CLICKED(FPGA_TP_Correction, &CGrabDemoDlg::OnBnClickedTpCorrection)
	ON_BN_CLICKED(FPGA_Median_Filter, &CGrabDemoDlg::OnBnClickedMedianFilter)
	ON_BN_CLICKED(NETD_test, &CGrabDemoDlg::OnBnClickedtest)
	ON_BN_CLICKED(FPGA_TEST, &CGrabDemoDlg::OnBnClickedTest)
	ON_BN_CLICKED(FPGA_BPmap, &CGrabDemoDlg::OnBnClickedBpmap)

	ON_BN_CLICKED(FPGA_TPmap, &CGrabDemoDlg::OnBnClickedTpmap)
	ON_BN_CLICKED(FPGA_Integral, &CGrabDemoDlg::OnBnClickedIntegral)
	ON_BN_CLICKED(I2C_BitRate, &CGrabDemoDlg::OnBnClickedBitrate)
	ON_WM_MOUSEMOVE()
	ON_WM_MOUSEMOVE()
	ON_BN_CLICKED(I2C_Change1, &CGrabDemoDlg::OnBnClickedChange1)
	ON_BN_CLICKED(I2C_Change2, &CGrabDemoDlg::OnBnClickedChange2)
	ON_BN_CLICKED(I2C_Change3, &CGrabDemoDlg::OnBnClickedChange3)
	ON_BN_CLICKED(FPGA_NU, &CGrabDemoDlg::OnBnClickedNu)
	ON_BN_CLICKED(FPGA_Update, &CGrabDemoDlg::OnBnClickedUpdate)
	ON_BN_CLICKED(FPGA_WinChange, &CGrabDemoDlg::OnBnClickedWinchange)
	ON_BN_CLICKED(IDC_Local_Enlarge, &CGrabDemoDlg::OnBnClickedLocalEnlarge)
	ON_STN_CLICKED(IDC_VIEW_WND2, &CGrabDemoDlg::OnStnClickedViewWnd2)
	ON_BN_CLICKED(IDC_DETECTION_CHECK, &CGrabDemoDlg::OnBnClickedCheck1)
	ON_BN_CLICKED(IDC_DEPTH_CHECK, &CGrabDemoDlg::OnBnClickedDepthCheck)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CGrabDemoDlg message handlers

void CGrabDemoDlg::XferCallback(SapXferCallbackInfo *pInfo)
{
   CGrabDemoDlg *pDlg= (CGrabDemoDlg *) pInfo->GetContext();

   //___________________添加处理图像函数______________
   int Height = m_DlgPointer->m_Buffers->GetHeight();//获取当前图像大小  768
   int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
   int Pixel_Depth = m_DlgPointer->m_Buffers->GetPixelDepth();
   CString cs;
   //弹窗输出字符串

   //-------------------显示帧频------------------
   //SapXferFrameRateInfo* pFrames;
   //pFrames = m_DlgPointer->m_Xfer->GetFrameRateStatistics();
   //float current_frames;
   //current_frames = pFrames->GetBufferFrameRate();

   frame_count++;  // 帧频计数加1
   //char   str[10];
			//				// double   x=   atof(jidian)   ;//转换成浮点数
   //int num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
   //wchar_t *wide = new wchar_t[num];
   //MultiByteToWideChar(0, 0, str, -1, wide, num);
   //((CButton*)m_DlgPointer->GetDlgItem(FPGA_frames))->SetWindowText(wide);
   //------------------------------------------------

   //cs.Format(_T("Image Size: %d * %d----%d"), Height, Width,Pixel_Depth);
   //pDlg->MessageBox(cs);
   
 
   //---------读取图像，保存为Image（768*1024）max 
   //BOOL ReadRect(int index, int x, int y, int width, int height, void* pData);
   m_DlgPointer->m_Buffers->ReadRect(0, 0, Width, Height, Image);
 /*  //------------图像处理算法测试代码---------
   // 自定义图像
   for (int i = 0; i < 768; i++)
	   for (int j = 0; j < 1024; j++)
		   Image[j + i * 1024] = 40000;
   //------两点矫正测试------
   for (int i = 0; i < 50; i++)
	   for (int j = 0; j < 50; j++)
		   Image[j + i * 1024] = 20000;

   //-------盲元测试-----
   for (int i = 700; i < 768; i = i + 3)
	   for (int j = 980; j < 1024; j = j + 3)
		   Image[j + i * 1024] = 0;
*/
   // 图像处理功能选择情况
   int TP_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_TP_Correction))->GetCheck();
   int Blind_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_Blind_Correction))->GetCheck();
   int Histogram_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_H_Enhance))->GetCheck();
   

   /*33の33333333333333333333333333
    //------测试两点矫正：固定输入图像-----
   for (int i = 0; i < 768; i++)
	   for (int j = 0; j < 1024; j++)
		   Image[j + i * 1024] = 40000;

   for (int i = 0; i < 50; i++)
	   for (int j = 0; j < 50; j++)
		   Image[j + i * 1024] = 20000;

   //-------------盲元测试---------
   for (int i = 700; i < 768; i = i + 3)
	   for (int j = 980; j < 1024; j = j + 3)
		   Image[j + i * 1024] = 0;  
   
   //---------非均匀矫正测试------
   for (int i = 0; i < 768; i++)
	   for (int j = 0; j < 1024; j++)
		   Image[j + i * 1024] = 20000 * i / 768;
		   */
   for (int i = 0; i < Height; i++)
	   for (int j = 0; j < Width; j++)
		   mImage[i * Width + j] = rImage[i * Width + j] >> 8;
   // src为原图像，gray为处理后图像，hist为直方图
   Mat gray_host = Mat(Height, Width, CV_8UC1, mImage);
   Mat flipped_both;
   // 上下和左右翻转
   cv::flip(gray_host, flipped_both, -1);
   gray_host = flipped_both;


   //Mat gray_host1_2;

   ////------------------------------------------------------------------------------
   //int cropHeight = 512; // 例如，裁剪区域高度为原始高度的一半
   //int cropWidth = 193;   // 裁剪区域宽度为原始宽度的一半						   
   //int startHeight1_2 = 0; // 计算裁剪区域的起始行
   //int startWidth1_2 = 168;     // 计算裁剪区域的起始列
   //float ini;
   //cv::Rect cropRegion1_2(startWidth1_2, startHeight1_2, cropWidth, cropHeight);
   //gray_host1_2 = gray_host(cropRegion1_2);
   ////------------------------------------------------------------------------------
   //gray_host = gray_host;
   // 创建一个大的空白图像来存放拼接结果  
   //gray_host1_2.copyTo(gray_host(cv::Rect(447, 0, 193, 512)));

   //for (int i = 0; i < 512; i++) {
	  // for (int j = 440; j < 460; j++) {
		 //  gray_host.at<uchar>(i, j) = (gray_host.at<uchar>(i, j - 3) + gray_host.at<uchar>(i, j - 2) + gray_host.at<uchar>(i, j - 1) + gray_host.at<uchar>(i, j) + \
			//   gray_host.at<uchar>(i, j + 1) + gray_host.at<uchar>(i, j + 2) + gray_host.at<uchar>(i, j + 3)) / 7.0;
	  // }

   //}
   //这里放拼接（放过了）

   if (BST_CHECKED == ((CButton*)pDlg->GetDlgItem(IDC_DETECTION_CHECK))->GetCheck())
   {
	   // 定义要分割的区域（与MATLAB脚本相同）
	   std::vector<std::vector<int>> regions = {
		   {1, 150, 1, 170},     // 区域1: 1-150行, 1-170列
		   {1, 150, 210, 420},   // 区域2: 1-150行, 210-420列
		   {1, 150, 460, Width}, // 区域3: 1-150行, 460到最后一列
		   {171, 350, 1, 170},   // 区域4: 171-350行, 1-170列
		   {171, 350, 210, 420}, // 区域5: 171-350行, 210-420列 (中心视图)
		   {171, 350, 460, Width}, // 区域6: 171-350行, 460到最后一列
		   {371, Height, 1, 170}, // 区域7: 370到最后一行, 1-170列
		   {371, Height, 210, 420}, // 区域8: 370到最后一行, 210-420列
		   {371, Height, 460, Width} // 区域9: 370到最后一行, 460到最后一列
	   };

	   // 创建一个vector来存储分割后的图像
	   std::vector<Mat> splitImgs(9);

	   // 抠出每个区域的图像
	   for (int i = 0; i < 9; i++) {
		   // 注意：OpenCV的行列索引从0开始，需要减1
		   int startRow = regions[i][0] - 1;
		   int endRow = regions[i][1] - 1;
		   int startCol = regions[i][2] - 1;
		   int endCol = regions[i][3] - 1;

		   // 确保索引在有效范围内
		   startRow = std::max(0, startRow);
		   endRow = std::min(Height - 1, endRow);
		   startCol = std::max(0, startCol);
		   endCol = std::min(Width - 1, endCol);

		   // 使用OpenCV的Rect创建感兴趣区域(ROI)
		   cv::Rect roi(startCol, startRow, endCol - startCol + 1, endRow - startRow + 1);
		   splitImgs[i] = gray_host(roi).clone();  // 克隆以创建独立图像
	   }

	   // 找到最小尺寸的图像
	   int minHeight = INT_MAX;
	   int minWidth = INT_MAX;
	   int minIndex = 0;

	   for (int i = 0; i < 9; i++) {
		   int height = splitImgs[i].rows;
		   int width = splitImgs[i].cols;

		   if (height < minHeight || (height == minHeight && width < minWidth)) {
			   minHeight = height;
			   minWidth = width;
			   minIndex = i;
		   }
	   }

	   // 获取最小尺寸
	   minHeight = splitImgs[minIndex].rows;
	   minWidth = splitImgs[minIndex].cols;

	   // 裁剪其他图像到最小尺寸（居中裁剪）
	   for (int i = 0; i < 9; i++) {
		   if (i != minIndex) {
			   int height = splitImgs[i].rows;
			   int width = splitImgs[i].cols;

			   int startRow = (height - minHeight) / 2;
			   int startCol = (width - minWidth) / 2;

			   cv::Rect roi(startCol, startRow, minWidth, minHeight);
			   splitImgs[i] = splitImgs[i](roi).clone();
		   }
	   }

	   // 使用中心视图（区域5）替换原始图像
	   // 注意：索引4对应区域5（C++从0开始计数）
	   Mat centerView = splitImgs[4];
	   Mat resizedCenterView;
	   cv::resize(centerView, resizedCenterView, cv::Size(Width, Height), 0, 0, cv::INTER_LINEAR);
	   // 创建一个与原始图像相同大小的空白图像
	   Mat resultImage = Mat::zeros(Height, Width, CV_8UC1);

	   // 将中心视图放置在结果图像的中心位置
	   int startRow = (Height - resizedCenterView.rows) / 2;
	   int startCol = (Width - resizedCenterView.cols) / 2;

	   // 确保位置有效
	   startRow = std::max(0, startRow);
	   startCol = std::max(0, startCol);

	   // 计算实际可以放置的尺寸
	   int actualHeight = std::min(resizedCenterView.rows, Height - startRow);
	   int actualWidth = std::min(resizedCenterView.cols, Width - startCol);

	   // 复制中心视图到结果图像
	   cv::Rect roi(startCol, startRow, actualWidth, actualHeight);
	   Mat resultROI = resultImage(roi);
	   resizedCenterView(cv::Rect(0, 0, actualWidth, actualHeight)).copyTo(resultROI);

	   // 更新显示的图像
	   gray_host = resultImage;
	   DWORD start, end;
	   DWORD engineProcessTime;
	   //预处理
	   start = GetTickCount();
	   m_tensorRT->preprocessImage(gray_host);
	   // 执行推理
	   m_tensorRT->inference(1);
	   // 后处理
	   std::vector<Detection> detections = m_tensorRT->postprocessOutputYOLOV8(1);
	   end = GetTickCount();
	   engineProcessTime = end - start;

	   for (const auto& det : detections) {
		   cv::rectangle(gray_host,
			   cv::Point(det.x, det.y),
			   cv::Point(det.x + det.width, det.y + det.height),
			   cv::Scalar(0, 255, 0), 2);
		   cv::putText(gray_host,
			   m_tensorRT->getClassName(det.classId) + " " + std::to_string(det.confidence) 
			   + " delay:" + std::to_string(engineProcessTime),
			   cv::Point(det.x, det.y - 5),
			   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
	   }
   }
   if (BST_CHECKED == ((CButton*)pDlg->GetDlgItem(IDC_DEPTH_CHECK))->GetCheck())
   {
	   // 1. 获取原始图像尺寸
	   int Width = gray_host.cols;
	   int Height = gray_host.rows;

	   // 2.【简化】直接定义并提取高质量的中心视图
	   // 区域5的坐标 (1-based): {171, 350, 210, 420}
	   int startCol = 210 - 1;
	   int startRow = 171 - 1;
	   int roiWidth = (420 - 1) - startCol + 1;
	   int roiHeight = (350 - 1) - startRow + 1;

	   // 边界检查，防止ROI超出图像范围
	   if (startCol + roiWidth > Width || startRow + roiHeight > Height) {
		   // 可以选择报错或调整ROI尺寸
		   return; // 暂时简单返回
	   }

	   cv::Rect centerViewROI(startCol, startRow, roiWidth, roiHeight);
	   // 直接从原始高质量图像中提取ROI，不使用clone()，除非后续会修改gray_host
	   cv::Mat centerView = gray_host(centerViewROI);
	   // 创建一个新的Mat对象来存储RGB图像
	   cv::Mat rgbImage;

	   // 将灰度图像转换为RGB图像
	   cv::cvtColor(centerView, rgbImage, cv::COLOR_GRAY2BGR);
	   // 3.【修正】对高质量的中心视图进行推理
	   DWORD start, end;
	   DWORD engineProcessTime;
	   start = GetTickCount();

	   // 将原始、清晰的 centerView 送入，你的 TensorRT 类会正确处理缩放
	   m_tensorRT->preprocessImage_Depth(rgbImage);
	   m_tensorRT->inference(1);
	   // 后处理得到的结果 depth_result 是对应 centerView 尺寸的深度图
	   cv::Mat depth_result = m_tensorRT->postprocessOutput_Depth(1);

	   end = GetTickCount();
	   engineProcessTime = end - start;

	   // 4.【修正】正确地显示深度图结果
	   // 创建一个与原始大图相同尺寸的黑色背景
	   // 注意：深度图是3通道彩色的(CV_8UC3)，所以背景也应该是3通道
	   //cv::Mat resultImage = cv::Mat::zeros(Height, Width, CV_8UC3);

	   //// 将计算出的深度图（它和centerView尺寸相同）拷贝回它在原图中的位置
	   //depth_result.copyTo(resultImage(centerViewROI));

	   // 更新显示的图像。注意 gray_host 现在会变成彩色图
	   cv::Mat final_display_img;
	   cv::cvtColor(depth_result, final_display_img, cv::COLOR_BGR2GRAY);
	   // 调整final_display_img的大小为原始图像的大小
	   cv::Mat resizedFinalDisplayImg;
	   cv::resize(final_display_img, resizedFinalDisplayImg, cv::Size(Width, Height), 0, 0, cv::INTER_LINEAR);
	   gray_host = resizedFinalDisplayImg;
   }
   unsigned char* rdata;
   rdata = gray_host.ptr<unsigned char>(0);
   for (int i = 0; i < Height; i++)
	   for (int j = 0; j < Width; j++)
		   rImage[i * Width + j] = rdata[i * Width + j] << 8;
   //----------------   GPU执行程序   -------------------
   if (BST_CHECKED == ((CButton*)m_DlgPointer->GetDlgItem(IDC_GPU))->GetCheck())
   {
	   //----------GPU- 测试代码------------

	   // GPU 处理函数 ： 图像指针，图像宽，图像长，使用GPU线程数
	   //Image[666] = 60000;
	   //Image[0] = 2;
	   if(Blind_On > 0)
		   cudaStatus=GPU_Blind_Correction(Image,dev_img, dev_pBlind_Ram, Height,Width);
	   if(TP_On > 0)
		   cudaStatus=GPU_TwoPoint_Correction(Image,dev_img, dev_pTP_Gain,dev_pTP_Bias, Height, Width);
	   if (Histogram_On > 0)
		cudaStatus = GPU_Histogram_Enhancement(Image, Histogram, Histogram_Float,dev_img, dev_Histogram, dev_Histogram_float, Height, Width);
	   //int k = pHistogram_Enhancement[65535];

	   if (cudaStatus != cudaSuccess) {
		   fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		   //goto Error;
	   }

   }
   m_DlgPointer->m_Buffers->WriteRect(0, 0, Width, Height, Image);
   // 调用cuda算法程序：直接将结果保存到全局变量rImage中。
 /*  m_DlgPointer->CUDA_Algorithm();*/
   if (((CButton*)m_DlgPointer->GetDlgItem(IDC_Local_Enlarge))->GetCheck()) {
	   m_DlgPointer->localEnlarge(Height, Width);
	   m_DlgPointer->m_Buffers->WriteRect(0, 0, Width, Height, Image);
   }
   pDlg->m_View->Show();

	   
   //---------向缓存区中写图像-----
   // BOOL WriteRect(int x, int y, int width, int height, const void* pData);
   



   // If grabbing in trash buffer, do not display the image, update the
   // appropriate number of frames on the status bar instead
 
   //_________判断是否为垃圾内容暂时没作用——删掉_____
/*   if (pInfo->IsTrash())  //判断是否为垃圾内容
   {
      CString str;
      str.Format(_T("Frames acquired in trash buffer: %d"), pInfo->GetEventCount());
      pDlg->m_statusWnd.SetWindowText(str);
   }
   
   // Refresh view 需要显示的图像进行保存
   else
   {
	   //-------------------多幅采集图像----------------
	   //---------保存图片的格式--------
	   if (Current_Saved < N_Saved_Frames)
	   {
		   Current_Saved++;
		   CString fileName = m_DlgPointer->m_cstrWorkPath;
		   CString tmp = "";
		   CTime time = CTime::GetCurrentTime();
		   fileName += "\\MultiSaved_";
		   tmp.Format(_T("%d"), Current_Saved);
		   fileName += tmp;
		   fileName += time.Format(_T("-%b-%d-%H-%M-%S.jpg"));
		   char szStr[256] = {};
		   wcstombs(szStr, fileName, fileName.GetLength());
		   const char* pBuf = szStr;
		   m_DlgPointer->m_Buffers->Save(pBuf, "-format jpg");
	   }
	   else
	   {
		   N_Saved_Frames = 0;
		   Current_Saved = 0;
	   }
*/
	   //m_Buffers->Save(pBuf, "-format avi",-1, numSave);

   //-----------NETD 计算-------------
   m_DlgPointer->CalculateNETD(Flag_NETD, Height, Width);
   
	   //-------------------多幅采集图像----------------
	   //---------保存图片的格式--------
   if (Current_Saved < N_Saved_Frames)
   {
	   Current_Saved++;
	   CString fileName = m_DlgPointer->m_cstrWorkPath;
	   CString tmp = "";
	   //CTime time = CTime::GetCurrentTime();
	   //fileName += "\\MultiSaved_";
	   tmp.Format(_T("\\%d.raw"), Current_Saved);
	   fileName += tmp;
	   //fileName += time.Format(_T("-%b-%d-%H-%M-%S.raw"));

	   CString pngFileName = fileName.Left(fileName.GetLength() - 4) + _T(".png");
	   for (int i = 0; i < Height; i++)
		   for (int j = 0; j < Width; j++)
			   mImage[i * Width + j] = rImage[i * Width + j] >> 8;
	   Mat src_png = Mat(Height, Width, CV_8UC1, mImage);
	   cv::imwrite(std::string(CT2CA(pngFileName)), src_png);

	   char szStr[256] = {};
	   wcstombs(szStr, fileName, fileName.GetLength());
	   const char* pBuf = szStr;
	   m_DlgPointer->m_Buffers->Save(pBuf, "-format raw");
   }
   else
   {
	   N_Saved_Frames = 0;
	   Current_Saved = 0;
   }
	  //显示图像
   
   
}
//畸变矫正
void CGrabDemoDlg::localEnlarge(int Height, int Width)
{
	for (int i = 0; i < Height; i++)
		for (int j = 0; j < Width; j++)
			mImage[i * Width + j] = rImage[i * Width + j] >> 8;
	// src为原图像，gray为处理后图像，hist为直方图
	Mat src_host = Mat(Height, Width, CV_8UC1, mImage);

	Mat gray_host;
	gray_host = distortionCailbration.process(src_host);
	////-------------------------之前--------------------------
	//int cropHeight = Height / 2; // 例如，裁剪区域高度为原始高度的一半
	//int cropWidth = Width / 2;   // 裁剪区域宽度为原始宽度的一半
	//int startHeight = (Height - cropHeight) / 2; // 计算裁剪区域的起始行
	//int startWidth = (Width - cropWidth) / 2;     // 计算裁剪区域的起始列

	//											  // 裁剪图像
	//cv::Rect cropRegion(startWidth, startHeight, cropWidth, cropHeight);
	//gray_host = src_host(cropRegion);
	//cv::resize(gray_host, gray_host, cv::Size(Width, Height));
	////---------------------------之前----------------------------
 
	

	//------ 输出部分： 将数据格式 8 bits 变为 16 bits ------------
	unsigned char* rdata;
	rdata = gray_host.ptr<unsigned char>(0);
	for (int i = 0; i < Height; i++)
		for (int j = 0; j < Width; j++)
			rImage[i * Width + j] = rdata[i * Width + j] << 8;
}

void CGrabDemoDlg::SignalCallback(SapAcqCallbackInfo *pInfo)
{
   CGrabDemoDlg *pDlg = (CGrabDemoDlg *) pInfo->GetContext();
   pDlg->GetSignalStatus(pInfo->GetSignalStatus());
}

void CGrabDemoDlg::PixelChanged(int x, int y)
{
   CString str = m_appTitle;
   str += "  " + m_ImageWnd.GetPixelString(CPoint(x, y));
   SetWindowText(str);
}

//***********************************************************************************
// Initialize Demo Dialog based application
//***********************************************************************************
BOOL CGrabDemoDlg::OnInitDialog()
{
   CRect rect;

   CDialog::OnInitDialog();
   
   // Add "About..." menu item to system menu.

   // IDM_ABOUTBOX must be in the system command range.
   ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
   ASSERT(IDM_ABOUTBOX < 0xF000);

   CMenu* pSysMenu = GetSystemMenu(FALSE);
   if (pSysMenu != NULL)
   {
      CString strAboutMenu;
      strAboutMenu.LoadString(IDS_ABOUTBOX);
      if (!strAboutMenu.IsEmpty())
      {
         pSysMenu->AppendMenu(MF_SEPARATOR);
         pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
      }

      pSysMenu->EnableMenuItem(SC_MAXIMIZE, MF_BYCOMMAND | MF_DISABLED | MF_GRAYED);
      pSysMenu->EnableMenuItem(SC_SIZE, MF_BYCOMMAND | MF_DISABLED | MF_GRAYED);
   }

   // Set the icon for this dialog.  The framework does this automatically
   //  when the application's main window is not a dialog
   SetIcon(m_hIcon, FALSE);	// Set small icon
   SetIcon(m_hIcon, TRUE);		// Set big icon

   // Initialize variables
   GetWindowText(m_appTitle);

   // Are we operating on-line?
   CAcqConfigDlg dlg(this, NULL);
   if (dlg.DoModal() == IDOK)
   {
      // Define on-line objects
      m_Acq			= new SapAcquisition(dlg.GetAcquisition());
      m_Buffers	= new SapBufferWithTrash(2, m_Acq);
      m_Xfer		= new SapAcqToBuf(m_Acq, m_Buffers, XferCallback, this);  //执行捕获到buffer，进入回调函数
//	  m_imageHeight = m_Buffers->GetHeight();
//	  m_imageWidth = m_Buffers->GetWidth();
   }
   else
   {
      // Define off-line objects
      m_Buffers	= new SapBuffer();
   }

   // Define other objects
   m_View = new SapView( m_Buffers);
   //m_ViewProcessed = new SapView(m_Buffers);
   // Attach sapview to image viewer
   m_ImageWnd.AttachSapView(m_View);
   //m_ImageWnd2.AttachSapView(m_ViewProcessed);
   // Create all objects
   if (!CreateObjects()) { EndDialog(TRUE); return FALSE; }

   m_ImageWnd.AttachEventHandler(this);
   m_ImageWnd.CenterImage(true);
   m_ImageWnd.Reset(); 
   //m_ImageWnd2.AttachEventHandler(this);
   //m_ImageWnd2.CenterImage(true);
   //m_ImageWnd2.Reset();
   UpdateMenu();

   // Get current input signal connection status
   GetSignalStatus();

   //--------串口菜单栏初始化------------
   m_Combo.AddString(_T("Xcelera-CL_PX4_1_Serial_0"));
   m_Combo.AddString(_T("Xcelera-CL_PX4_1_Serial_1"));
   m_Combo.AddString(_T("COM2"));
   m_Combo.SetCurSel(0);//初始时下拉列表为COM2

   //-------串口传输速率--------------
   Comb_Rate.InsertString(0,_T("9600"));
   Comb_Rate.InsertString(1, _T("115200"));
   Comb_Rate.SetCurSel(0);//初始时下拉列表为9600

   //----------I2C 配置----------
   Combe_I2CMode.InsertString(0,_T("2K模式"));
   Combe_I2CMode.InsertString(1,_T("8K模式"));
   Combe_I2CMode.InsertString(2,_T("QPSK"));
   Combe_I2CMode.InsertString(3,_T("16-QAM"));
   Combe_I2CMode.InsertString(4,_T("64-QAM"));
   Combe_I2CMode.SetCurSel(0);//初始时下拉列表为9600

   Combe_I2CBitSet.InsertString(0,_T("内纠错码率：1/2"));
   Combe_I2CBitSet.InsertString(1,_T("内纠错码率：2/3"));
   Combe_I2CBitSet.InsertString(2,_T("内纠错码率：3/4"));
   Combe_I2CBitSet.InsertString(3,_T("内纠错码率：5/6"));
   Combe_I2CBitSet.InsertString(4,_T("内纠错码率：7/8"));
   Combe_I2CBitSet.SetCurSel(0);//初始时下拉列表为9600

   Combe_I2C_TimeSet.InsertString(0,_T("保护间隔：1/4"));
   Combe_I2C_TimeSet.InsertString(1,_T("保护间隔：1/8"));
   Combe_I2C_TimeSet.InsertString(2,_T("保护间隔：1/16"));
   Combe_I2C_TimeSet.InsertString(3,_T("保护间隔：1/32"));
   Combe_I2C_TimeSet.SetCurSel(0);//初始时下拉列表为9600

   st.pWnd = this->GetSafeHwnd();  // 获得当前窗口句柄

   //--------初始化时更新图像分辨率----------
   int Height = m_DlgPointer->m_Buffers->GetHeight();//获取当前图像大小  768
   int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024

   mHeight = Height;
   mWidth = Width;
   imageBits = m_DlgPointer->m_Buffers->GetPixelDepth();

   char str[10];
   int num;
   wchar_t *wide;
   sprintf(str, "%d", imageBits);//转换成字符串   
   num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
   wide = new wchar_t[num];
   MultiByteToWideChar(0, 0, str, -1, wide, num);
   ((CButton*)m_DlgPointer->GetDlgItem(FPGA_Bits))->SetWindowText(wide);

   UpdateData(FALSE);

   SetTimer(2, 1000, NULL);  //  单位：ms   //记录帧频时间

   try {
	   TCHAR exePath[MAX_PATH];
	   GetModuleFileName(NULL, exePath, MAX_PATH);
	   std::wstring wstrExePath(exePath);
	   std::string strExePath(wstrExePath.begin(), wstrExePath.end());

	   size_t pos = strExePath.find_last_of("\\/");
	   std::string exeDir = (pos != std::string::npos) ? strExePath.substr(0, pos) : "";

	   std::string enginePath = exeDir + "\\depth_anything_v2_vits_518x616.engine";
	   m_tensorRT = new MyTensorRT(enginePath, true);

	   //gpu 预热
	   //gpu 预热
	   m_tensorRT->warmup(5);
	   ////TODO本地文件测试，正式使用以下屏蔽
	   //cv::Mat testMat = cv::imread(exeDir + "\\001_7.png");
	   //DWORD start, end;
	   //DWORD processTime;

	   //// --- 深度估计的推理流程 ---
	   //start = GetTickCount();

	   //// 1. 预处理 (调用新增的深度专用函数)
	   //m_tensorRT->preprocessImage_Depth(testMat);

	   //// 2. 执行推理 (复用通用的推理函数)
	   //m_tensorRT->inference(1);

	   //// 3. 后处理 (调用新增的深度专用函数)
	   //cv::Mat depth_result = m_tensorRT->postprocessOutput_Depth(1);

	   //end = GetTickCount();
	   //processTime = end - start;

	   //// 显示结果
	   //CString msg;
	   //msg.Format(_T("深度估计完成, 花费时间 %d 毫秒"), processTime);
	   //MessageBox(msg, _T("处理结果"), MB_OK | MB_ICONINFORMATION);
	   //cv::Mat final_display_img;
	   //cv::cvtColor(depth_result, final_display_img, cv::COLOR_BGR2GRAY);
	   //// 在窗口中显示原始图像和深度图
	   //cv::imshow("Original Image", testMat);
	   //cv::imshow("Depth Result", final_display_img);
	   //cv::waitKey(0); // 等待按键后关闭窗口
	   //TODO本地文件测试，正式使用以下屏蔽
	   //cv::Mat testMat = cv::imread(exeDir + "\\001_6.png");
	   //DWORD start, end;
	   //DWORD engineProcessTime;
	   ////预处理
	   //start = GetTickCount();
	   //m_tensorRT->preprocessImage(testMat);
	   //// 执行推理
	   //m_tensorRT->inference(1);
	   //// 后处理
	   //std::vector<Detection> detections = m_tensorRT->postprocessOutputYOLOV8(1);
	   //end = GetTickCount();
	   //engineProcessTime = end - start;
	   //CString msg;
	   //msg.Format(_T("检测到 %d 个目标, 花费时间 %d 毫秒"), detections.size(), engineProcessTime);

	   //MessageBox(msg, _T("检测结果"), MB_OK | MB_ICONINFORMATION);
	   //for (const auto& det : detections) {
		  // cv::rectangle(testMat,
			 //  cv::Point(det.x, det.y),
			 //  cv::Point(det.x + det.width, det.y + det.height),
			 //  cv::Scalar(0, 255, 0), 2);
		  // cv::putText(testMat,
			 //  m_tensorRT->getClassName(det.classId) + " " + std::to_string(det.confidence),
			 //  cv::Point(det.x, det.y - 5),
			 //  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
	   //}
	   //cv::imshow("detectResult",testMat);
	   //cv::waitKey(0);
	   //TODO本地文件测试，正式使用以上屏蔽
   }
   catch (const std::exception& e) {
	   CString errorMsg;  
	   errorMsg.Format(_T("初始化 TensorRT 失败:\n%hs"), e.what());
	   MessageBox(errorMsg, _T("错误"), MB_OK | MB_ICONERROR);
	   return FALSE;
   }
   return TRUE;  // return TRUE  unless you set the focus to a control
}

BOOL CGrabDemoDlg::CreateObjects()
{
   CWaitCursor wait;

   // Create acquisition object
   if (m_Acq && !*m_Acq && !m_Acq->Create())
   {
      DestroyObjects();
      return FALSE;
   }

   // Create buffer object
   if (m_Buffers && !*m_Buffers)
   {
      if( !m_Buffers->Create())
      {
         DestroyObjects();
         return FALSE;
      }
      // Clear all buffers
      m_Buffers->Clear();
   }

   // Create view object
   if (m_View && !*m_View && !m_View->Create())
   {
      DestroyObjects();
      return FALSE;
   }
   //if (m_ViewProcessed && !*m_ViewProcessed && !m_ViewProcessed->Create())
   //{
	  // DestroyObjects();
	  // return FALSE;
   //}

   // Create transfer object
   if (m_Xfer && !*m_Xfer && !m_Xfer->Create())
   {
      DestroyObjects();
      return FALSE;
   }

   //---------------初始化数组-------------------
   memset(rHot_Ram, 78, 1280 * 1024 * sizeof(unsigned short));   //0~255 八位二进制数进行填充 所以用78 对应 20000左右
   memset(rBlind_Ram, 0, 1280 * 1024 * sizeof(unsigned short));

   return TRUE;
}

BOOL CGrabDemoDlg::DestroyObjects()
{
   // Destroy transfer object
   if (m_Xfer && *m_Xfer) m_Xfer->Destroy();

   // Destroy view object
   if (m_View && *m_View) m_View->Destroy();
   //if (m_ViewProcessed && *m_ViewProcessed) m_ViewProcessed->Destroy();
   // Destroy buffer object
   if (m_Buffers && *m_Buffers) m_Buffers->Destroy();

   // Destroy acquisition object
   if (m_Acq && *m_Acq) m_Acq->Destroy();

   return TRUE;
}

//**********************************************************************************
//
//				Window related functions
//
//**********************************************************************************
void CGrabDemoDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
   if(( nID & 0xFFF0) == IDM_ABOUTBOX)
   {
      CAboutDlg dlgAbout;
      dlgAbout.DoModal();
   }
   else
   {
      CDialog::OnSysCommand(nID, lParam);
   }
}


// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.
void CGrabDemoDlg::OnPaint() 
{
   if( IsIconic())
   {
      CPaintDC dc(this); // device context for painting

      SendMessage(WM_ICONERASEBKGND, (WPARAM) dc.GetSafeHdc(), 0);

      // Center icon in client rectangle
      INT32 cxIcon = GetSystemMetrics(SM_CXICON);
      INT32 cyIcon = GetSystemMetrics(SM_CYICON);
      CRect rect;
      GetClientRect(&rect);
      INT32 x = (rect.Width() - cxIcon + 1) / 2;
      INT32 y = (rect.Height() - cyIcon + 1) / 2;

      // Draw the icon
      dc.DrawIcon(x, y, m_hIcon);
   }
   else
   {
      CDialog::OnPaint();
   }
}

void CGrabDemoDlg::OnDestroy() 
{
   CDialog::OnDestroy();

   // Destroy all objects
   DestroyObjects();

   // Delete all objects
   if (m_Xfer)			delete m_Xfer; 
   if (m_View)			delete m_View;
   //if (m_ViewProcessed)	delete m_ViewProcessed;
   if (m_Buffers)		delete m_Buffers; 
   if (m_Acq)			delete m_Acq; 
}

void CGrabDemoDlg::OnSize(UINT nType, int cx, int cy) 
{
   CDialog::OnSize(nType, cx, cy);

   CRect rClient;
   GetClientRect(rClient);

   // resize image viewer
   if (m_ImageWnd.GetSafeHwnd())
   {
      CRect rWnd;
      m_ImageWnd.GetWindowRect(rWnd);
      ScreenToClient(rWnd);
      rWnd.right = rClient.right - 5;
      rWnd.bottom = rClient.bottom - 5;
      m_ImageWnd.MoveWindow(rWnd);
   }
   //if (m_ImageWnd2.GetSafeHwnd())
   //{
	  // CRect rWnd;
	  // m_ImageWnd2.GetWindowRect(rWnd);
	  // ScreenToClient(rWnd);
	  // rWnd.right = rClient.right - 5;
	  // rWnd.bottom = rClient.bottom - 5;
	  // m_ImageWnd2.MoveWindow(rWnd);
   //}
}


// The system calls this to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CGrabDemoDlg::OnQueryDragIcon()
{
   return (HCURSOR) m_hIcon;
}


void CGrabDemoDlg::OnExit() 
{
   EndDialog(TRUE);
}

void CGrabDemoDlg::OnEndSession(BOOL bEnding)
{
   CDialog::OnEndSession(bEnding);

   if( bEnding)
   {
      // If ending the session, free the resources.
      OnDestroy(); 
   }
}

BOOL CGrabDemoDlg::OnQueryEndSession()
{
   if (!CDialog::OnQueryEndSession())
      return FALSE;

   return TRUE;
}

//==============================================================================
// Name      : CSeqGrabDemoDlg::OnKillfocusBufferFrameRate
// Purpose   : Handle the
// Parameters: None
//==============================================================================
void CGrabDemoDlg::OnKillfocusBufferFrameRate(void)
{
	UpdateData(TRUE);
	m_Buffers->SetFrameRate(m_BufferFrameRate);
} // End of the CSeqGrabDemoDlg::OnKillfocusFrameRate method.

//**************************************************************************************
// Updates the menu items enabling/disabling the proper items depending on the state
//  of the application
//**************************************************************************************
void CGrabDemoDlg::UpdateMenu( void)
{
   BOOL bAcqNoGrab	= m_Xfer && *m_Xfer && !m_Xfer->IsGrabbing();
   BOOL bAcqGrab		= m_Xfer && *m_Xfer && m_Xfer->IsGrabbing();
   BOOL bNoGrab		= !m_Xfer || !m_Xfer->IsGrabbing();
   INT32	 scan = 0;
   BOOL bLineScan    = m_Acq && m_Acq->GetParameter(CORACQ_PRM_SCAN, &scan) && (scan == CORACQ_VAL_SCAN_LINE);
   INT32 iInterface = CORACQ_VAL_INTERFACE_DIGITAL;
   if (m_Acq)
      m_Acq->GetCapability(CORACQ_CAP_INTERFACE, (void *) &iInterface);

   // Acquisition Control
   GetDlgItem(IDC_GRAB)->EnableWindow(bAcqNoGrab);
   GetDlgItem(IDC_SNAP)->EnableWindow(bAcqNoGrab);
   GetDlgItem(IDC_FREEZE)->EnableWindow(bAcqGrab);

   // Acquisition Options
   GetDlgItem(IDC_GENERAL_OPTIONS)->EnableWindow(bAcqNoGrab);
   GetDlgItem(IDC_AREA_SCAN_OPTIONS)->EnableWindow(bAcqNoGrab && !bLineScan);
   GetDlgItem(IDC_LINE_SCAN_OPTIONS)->EnableWindow(bAcqNoGrab && bLineScan);
   GetDlgItem(IDC_COMPOSITE_OPTIONS)->EnableWindow(bAcqNoGrab && (iInterface == CORACQ_VAL_INTERFACE_ANALOG) );
   GetDlgItem(IDC_LOAD_ACQ_CONFIG)->EnableWindow(m_Xfer && !m_Xfer->IsGrabbing());

   // File Options
   GetDlgItem(IDC_FILE_NEW)->EnableWindow(bNoGrab);
   GetDlgItem(IDC_FILE_LOAD)->EnableWindow(bNoGrab);
   GetDlgItem(IDC_FILE_SAVE)->EnableWindow(bNoGrab);

   // Image filter Options
   GetDlgItem(IDC_IMAGE_FILTER_OPTIONS)->EnableWindow(bAcqNoGrab && m_Acq && *m_Acq && m_Acq->IsImageFilterAvailable());

   // General Options
   GetDlgItem(IDC_BUFFER_OPTIONS)->EnableWindow(bNoGrab);

   // If last control was disabled, set default focus
   if (!GetFocus())
      GetDlgItem(IDC_EXIT)->SetFocus();

   //---------初始化串口开关------------
   //GetDlgItem(FPGA_HE)->SetWindowText(_T("增强_打开"));
}


//*****************************************************************************************
//
//					Acquisition Control
//
//*****************************************************************************************

void CGrabDemoDlg::OnFreeze( ) 
{
   if( m_Xfer->Freeze())
   {
      if (CAbortDlg(this, m_Xfer).DoModal() != IDOK) 
         m_Xfer->Abort();

      UpdateMenu();
   }
}

void CGrabDemoDlg::OnGrab() 
{
   m_statusWnd.SetWindowText(_T(""));

   if( m_Xfer->Grab())  //开始连续传输
   {
      UpdateMenu();	
   }
}

void CGrabDemoDlg::OnSnap() 
{
   m_statusWnd.SetWindowText(_T(""));

   if( m_Xfer->Snap())
   {
      if (CAbortDlg(this, m_Xfer).DoModal() != IDOK) 
         m_Xfer->Abort();

      UpdateMenu();	
   }
}


//*****************************************************************************************
//
//					Acquisition Options
//
//*****************************************************************************************

void CGrabDemoDlg::OnGeneralOptions() 
{
   CAcqDlg dlg(this, m_Acq);
   dlg.DoModal();
}

void CGrabDemoDlg::OnAreaScanOptions() 
{
   CAScanDlg dlg(this, m_Acq);
   dlg.DoModal();
}

void CGrabDemoDlg::OnLineScanOptions() 
{
   CLScanDlg dlg(this, m_Acq);
   dlg.DoModal();
}

void CGrabDemoDlg::OnCompositeOptions() 
{
   if( m_Xfer->Snap())
   {
      CCompDlg dlg(this, m_Acq, m_Xfer);
      dlg.DoModal();

      UpdateMenu();
   }
}

void CGrabDemoDlg::OnLoadAcqConfig() 
{
   // Set acquisition parameters
   CAcqConfigDlg dlg(this, m_Acq);
   if (dlg.DoModal() == IDOK)
   {
      // Destroy objects
      DestroyObjects();

      // Update acquisition object
      SapAcquisition acq = *m_Acq;
      *m_Acq = dlg.GetAcquisition();

      // Recreate objects
      if (!CreateObjects())
      {
         *m_Acq = acq;
         CreateObjects();
      }

      GetSignalStatus();

      m_ImageWnd.Reset();
	  //m_ImageWnd2.Reset();
      InvalidateRect(NULL);
      UpdateWindow();
      UpdateMenu();
   }
}

void CGrabDemoDlg::OnImageFilterOptions()
{
   CImageFilterEditorDlg dlg(m_Acq);
   dlg.DoModal();
   
}

//*****************************************************************************************
//
//					General Options
//
//*****************************************************************************************

void CGrabDemoDlg::OnBufferOptions() 
{
   CBufDlg dlg(this, m_Buffers, m_View->GetDisplay());
   if (dlg.DoModal() == IDOK)
   {
      // Destroy objects
      DestroyObjects();

      // Update buffer object
      SapBuffer buf = *m_Buffers;
      *m_Buffers = dlg.GetBuffer();

      // Recreate objects
      if (!CreateObjects())
      {
         *m_Buffers = buf;
         CreateObjects();
      }
	  //改变buffer设置，更新图像大小
//	  m_imageHeight = m_Buffers->GetHeight();
//	  m_imageWidth = m_Buffers->GetWidth();

      m_ImageWnd.Reset();
	  //m_ImageWnd2.Reset();
      InvalidateRect(NULL);
      UpdateWindow();
      UpdateMenu();
   }
}

void CGrabDemoDlg::OnViewOptions() 
{
   CViewDlg dlg(this, m_View);
   if( dlg.DoModal() == IDOK)
      m_ImageWnd.Refresh();
      //m_ImageWnd2.Refresh();
}

//*****************************************************************************************
//
//					File Options
//
//*****************************************************************************************

void CGrabDemoDlg::OnFileNew() 
{
   m_Buffers->Clear();
   InvalidateRect( NULL, FALSE);
}

void CGrabDemoDlg::OnFileLoad() 
{
   CLoadSaveDlg dlg(this, m_Buffers, TRUE);
   if (dlg.DoModal() == IDOK)
   {
      InvalidateRect(NULL);
      UpdateWindow();
   }
}

//  Save按键  保存单帧图像
void CGrabDemoDlg::OnFileSave() 
{
	int Height = m_DlgPointer->m_Buffers->GetHeight();//获取当前图像大小  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
	CString fileName = m_cstrWorkPath;
	CTime time = CTime::GetCurrentTime();
	fileName += time.Format(_T("\\single-%b-%d-%H-%M-%S.raw"));
	char szStr[256] = {};
	wcstombs(szStr, fileName, fileName.GetLength());
	const char* pBuf = szStr;
	m_Buffers->Save(pBuf,"-format raw");
  // SaveSingleFrame();
	CString pngFileName = fileName.Left(fileName.GetLength() - 4) + _T(".png");
	for (int i = 0; i < Height; i++)
		for (int j = 0; j < Width; j++)
			mImage[i * Width + j] = rImage[i * Width + j] >> 8;
	Mat src_png = Mat(Height, Width, CV_8UC1, mImage);
	cv::imwrite(std::string(CT2CA(pngFileName)), src_png);

	//手动选择保存路径
	// dlg.DoModal();
	// CLoadSaveDlg dlg(this, m_Buffers, FALSE);
}

void CGrabDemoDlg::GetSignalStatus()
{
   SapAcquisition::SignalStatus signalStatus;

   if (m_Acq && m_Acq->IsSignalStatusAvailable())
   {
      if (m_Acq->GetSignalStatus(&signalStatus, SignalCallback, this))
         GetSignalStatus(signalStatus);
   }
}

void CGrabDemoDlg::GetSignalStatus(SapAcquisition::SignalStatus signalStatus)
{
   m_IsSignalDetected = (signalStatus != SapAcquisition::SignalNone);

   if (m_IsSignalDetected)
      SetWindowText(m_appTitle);
   else
   {
      CString newTitle = m_appTitle;
      newTitle += " (No camera signal detected)";
      SetWindowText(newTitle);
   }
}


//  采集多帧图像
void CGrabDemoDlg::OnBnClickedSavemulti()
{
	// TODO: 在此添加控件通知处理程序代码
	// 获取连续采集的帧数
	CString nFrames;
	int numSave;		//采集的帧频数
	//GetDlgItem(IDC_Frame_Count)->GetWindowTextW(nFrames);
	GetDlgItemText(IDC_Frame_Count, nFrames);
	numSave = _ttoi(nFrames);
	if (numSave <= 0 || numSave >= 1000)
	{
		MessageBox(_T("The Input of Frame Counts Warning !"));
		numSave = 0;
	}
		
/*	//---------保存图片的格式--------
	CString fileName = m_cstrWorkPath;
	CTime time = CTime::GetCurrentTime();
	fileName += time.Format(_T("\\MultiSaved-%b-%d-%H-%M-%S.bmp"));
	char szStr[256] = {};
	wcstombs(szStr, fileName, fileName.GetLength());
	const char* pBuf = szStr;
	//m_Buffers->Save(pBuf, "-format avi",-1, numSave);
*/
	N_Saved_Frames = numSave;
	
	// demo例程 保存avi文件
/*	if ((m_Buffers->GetFormat() == SapFormatMono16) ||
		(m_Buffers->GetFormat() == SapFormatRGB101010) ||
		(m_Buffers->GetFormat() == SapFormatRGB161616) ||
		(m_Buffers->GetFormat() == SapFormatRGB16161616))
	{
		MessageBox(_T("Saving images in AVI format requires downsampling them to 8-bit pixel depth.\nYou will not be able to reload this sequence in this application unless you change the buffer format."));
	}

	if ((m_Buffers->GetFormat() == SapFormatRGBR888))
	{
		MessageBox(_T("Saving images in AVI format requires conversion to RGB888 format (blue first).\nYou will not be able to reload this sequence in this application unless you change the buffer format."));
	}

	CLoadSaveDlg dlg(this, m_Buffers, FALSE, TRUE);
	dlg.DoModal();
*/
}

//实现定时器 
void CGrabDemoDlg::OnTimer(UINT_PTR nIDEvent) {
	char   str[20];
	int num;
	wchar_t *wide;
	HWND hWnd;
	switch (nIDEvent)
	{
	case 0:  //定时器处理程序
		CGrabDemoDlg::OnFileSave();
		//MessageBox(_T("Time is on !"));
		break;
	case 1 :
		if (st.m_listFrames.size() > 0)
			st.OnReceive();
			//st.sendFrame(*(*st.m_listFrames.begin()));
		else
			KillTimer(1);
		break;
	case 2:
		sprintf(str, "%d", frame_count);//转换成字符串   
		num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
		wide = new wchar_t[num];
		//delete[] wide;
		MultiByteToWideChar(0, 0, str, -1, wide, num);
		((CButton*)m_DlgPointer->GetDlgItem(FPGA_frames))->SetWindowText(wide);
		//---------计算传输速率----------------
		sprintf(str, "%.4f", float(frame_count)*mHeight*mWidth*imageBits/1000000);//转换成字符串   
		num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
		wide = new wchar_t[num];
		MultiByteToWideChar(0, 0, str, -1, wide, num);
		((CButton*)m_DlgPointer->GetDlgItem(FPGA_BitsRates))->SetWindowText(wide);

		//GetDlgItem(FPGA_frames)->SetWindowText((LPCTSTR)str);
		frame_count = 0;
		//Frame_Count = 0;
		//-------------发送鼠标移动消息--------
		//hWnd = AfxGetMainWnd()->m_hWnd;
		//PostMessage( WM_MOUSEMOVE, 1,NULL);
		SendMessage(WM_MOUSEMOVE, MK_SHIFT, 0x12345678);
		break;
	default:
		break;
	}
}


//	定时采集模式，打开定时器，读取时长，开始采集
//	需要考虑何时关闭的问题：按了其他保存模式，或退出程序。
void CGrabDemoDlg::OnBnClickedSaveTiming()
{
	// TODO: 在此添加控件通知处理程序代码
	//MessageBox(_T("Button is on !"));
	// 获取连续采集的帧数
	CString strTime;
	int nMS;		//采集的帧频数
						//GetDlgItem(IDC_Frame_Count)->GetWindowTextW(nFrames);
	GetDlgItemText(IDC_nMS, strTime);
	nMS = _ttoi(strTime);

	SetTimer(0, nMS,NULL);   // 第二个参数为x ms
}


void CGrabDemoDlg::OnBnClickedTimingStop()
{
	// TODO: 在此添加控件通知处理程序代码
	KillTimer(0);
}


//int GPU_test()
//{
//	const int arraySize = 5;
//	const int a[arraySize] = { 1, 2, 3, 4, 5 };
//	const int b[arraySize] = { 10, 20, 30, 40, 50 };
//	int c[arraySize] = { 0 };
//
//	// Add vectors in parallel.  
//	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "addWithCuda failed!");
//		return 1;
//	}
//
//	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//		c[0], c[1], c[2], c[3], c[4]);
//	printf("cuda工程中调用cpp成功！\n");
//
//	// cudaDeviceReset must be called before exiting in order for profiling and  
//	// tracing tools such as Nsight and Visual Profiler to show complete traces.  
//	cudaStatus = cudaDeviceReset();
//	if (cudaStatus != cudaSuccess) {
//		fprintf(stderr, "cudaDeviceReset failed!");
//		return 1;
//	}
//	getchar(); //here we want the console to hold for a while  
//	return 0;
//}


//--------------------低温本底按键-----------------
void CGrabDemoDlg::OnBnClickedTpCold()
{
	// 采集低温本底图像信息，保存如 Cold_Ram
	int Height = m_DlgPointer->m_Buffers->GetHeight();//获取当前图像大小  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
	m_DlgPointer->m_Buffers->ReadRect(0, 0, Width, Height, Cold_Ram);
}

//--------------------高温本底按键-----------------
void CGrabDemoDlg::OnBnClickedTpHot()
{
	// 采集高温本底图像信息，保存如 Hot_Ram
	int Height = m_DlgPointer->m_Buffers->GetHeight();//获取当前图像大小  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
	m_DlgPointer->m_Buffers->ReadRect(0, 0, Width, Height, Hot_Ram);

/*	//------------------测试效果------------------
	//开始时，先点击GPU，开辟内存，然后高温本底采集，然后显示
	//------两点矫正测试------
	for (int i = 0; i < 50; i++)
	for (int j = 0; j < 50; j++)
	Hot_Ram[j + i * 1024] = 10000;

	//-------盲元测试-----
	for (int i = 700; i < 768; i=i+3)
	for (int j = 980; j < 1024; j=j+3)
	Hot_Ram[j + i * 1024] = 50;
*/	

	//---------------------------------------------


	//-------------------------------------------------------
	//					两点矫正实现
	//-------------------------------------------------------
	//计算两点矫正参数
	double Mean_Cold = 0; double Mean_Hot = 0;

	//-------------------- 两点矫正 两个本底的均值计算 -------------
	int n = 0;
	for (int i = 0; i < Height; i++)
		for (int j = 0; j < Width; j++)
		{			
			n = j + i * Width + 1; //表示第n个进入求解的数字
			//--------简化平均运算------
			Mean_Cold = ((n - 1)*Mean_Cold + Cold_Ram[n-1]) / n;
			Mean_Hot = ((n - 1)*Mean_Hot + Hot_Ram[n-1]) / n;
		}


	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
		{
			{
				if ((Hot_Ram[j + i * Width] - Cold_Ram[j + i * Width]) > 0)
					TP_Gain[j + i * Width] = (Mean_Hot - Mean_Cold) / (Hot_Ram[j + i * Width] - Cold_Ram[j + i * Width]);
				//if (TP_Gain[j + i * 1024]<1 )
				//	TP_Gain[j + i * 1024] = 1;
			}

		}
	}
	//int test = TP_Gain[10];
	//test = (Hot_Ram[10000] - Cold_Ram[10000]);

	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
			TP_Bias[j + i * Width] = Mean_Cold - TP_Gain[j + i * Width] * Cold_Ram[j + i * Width];
	}

	//------------------将两点矫正表拷贝到GPU，以供调用------------------
	

	//如果GPU可以使用，将本底写入GPU内存
	cudaStatus = cudaMemcpy(dev_pTP_Gain, TP_Gain, Width * Height * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_pTP_Bias, TP_Bias, Width * Height * sizeof(float), cudaMemcpyHostToDevice);


	//-------------------------------------------------------
	//					盲元矫正实现
	//  实现：需要更改时，CPU计算盲元表，导入GPU进行及时运算
	//-------------------------------------------------------
	// 盲元矫正 ----- 在高温本底中，灰度值为0的设置为盲元，Ram设置为0
	Death_num = 0;
	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
			// 选取盲元的规则：
			if (Hot_Ram[j + i * Width] <= Cold_Ram[j + i * Width] || Hot_Ram[j + i * Width] - Cold_Ram[j + i * Width] < 0.8*(Mean_Hot - Mean_Cold) || Hot_Ram[j + i * Width] - Cold_Ram[j + i * Width] > 1.3*(Mean_Hot - Mean_Cold))
				//if (Hot_Ram[j + i * Width] < Cold_Ram[j + i * Width] || Hot_Ram[j + i * Width] - Cold_Ram[j + i * Width] < 100)
			{
				pBlind_Ram[j + i * Width] = 1;
				Death_num++;   //统计盲元个数
			}
			else
			{
				pBlind_Ram[j + i * Width] = 0;
			}

	}
	//------------------将盲元表拷贝到GPU，以供调用------------------	
	cudaStatus = cudaMemcpy(dev_pBlind_Ram, pBlind_Ram, Width * Height * sizeof(unsigned short), cudaMemcpyHostToDevice);
	//test = pBlind_Ram[700 * 1024 + 980];
	//test = pBlind_Ram[700 * 1024 + 981];
	//test = pBlind_Ram[700 * 1024 + 982];

}



//--------------直方图增强按键------------------
void CGrabDemoDlg::OnBnClickedHEnhance()
{
/*	// TODO: 在此添加控件通知处理程序代码
	int Histogram_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_H_Enhance))->GetCheck();

	////---------非均匀矫正测试------
	//for (int i = 0; i < 768; i++)
	//	for (int j = 0; j < 1024; j++)
	//		Image[j + i * 1024] = 20000 * i / 768;
	//_______________________________________

	//-------点击增强，开始重新建立增强表
	if (Histogram_On > 0)
	{
		// ----------统计各个像素的数量------------
		unsigned short Histogram_Count[65536] = { 0 };   //直方图增强表
		for (int i = 0; i < 768; i++)
			for (int j = 0; j < 1024; j++)
			{
				Histogram_Count[Image[j + i * 1024]] = Histogram_Count[Image[j + i * 1024]] + 1;
				int k = Histogram_Count[Image[j + i * 1024]];
				k = 1;
			}

		//------------
		float sum = 0;

		//-----------生成直方图表--------------
		for (int i = 0; i < 65536; i++)
		{
			sum = sum + Histogram_Count[i];
			pHistogram_Enhancement[i] = sum / 768 / 1024;
		}

		//float k = pHistogram_Enhancement[100];
		//k = 22;
		//k = pHistogram_Enhancement[65535];
		//k = 0;
	}
	//--------------------------------------------
	//int x1 = 0;
	*/
}

//------------------CUDA响应程序：ON---则重新配置CUDA---------
void CGrabDemoDlg::OnBnClickedGpu()
{

	// ON---则重新配置CUDA    OFF---释放内存，重置GPU
	int GPU_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_GPU))->GetCheck();
	int Height = m_DlgPointer->m_Buffers->GetHeight();//获取当前图像大小  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
	cudaError_t cudaStatus;

	////---------非均匀矫正测试------
	//for (int i = 0; i < 768; i++)
	//	for (int j = 0; j < 1024; j++)
	//		Image[j + i * 1024] = 20000 * i / 768;
	//_______________________________________

	//-------点击增强，开始重新建立增强表
	if (GPU_On > 0)
	{
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}

		//---------------------------图像内存开辟---------------------------------------------------
		// 开辟存放图像的内存    .  
		cudaStatus = cudaMalloc((void**)&dev_img, Height * Width * sizeof(unsigned short));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}


		//----------------------------两点矫正中矫正表内存开辟---------------------------------------
		cudaStatus = cudaMalloc((void**)&dev_pTP_Gain, Height * Width * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Gain cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)&dev_pTP_Bias, Height * Width * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pTP_Bias cudaMalloc failed!");
			goto Error;
		}

		//--------------------------------盲元矫正实现--------------------------------
		cudaStatus = cudaMalloc((void**)&dev_pBlind_Ram, Height * Width * sizeof(unsigned short));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pBlind_Ram cudaMalloc failed!");
			goto Error;
		}

		//--------------------------------直方图均衡表内存开辟--------------------------------
		cudaStatus = cudaMalloc((void**)&dev_Histogram, 65536 * sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_Histogram cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMalloc((void**)&dev_Histogram_float, 65536 * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_Histogram cudaMalloc failed!");
			goto Error;
		}
		
	}
	else
	{
	Error:
		cudaFree(dev_img);
		cudaFree(dev_pTP_Gain); cudaFree(dev_pTP_Bias);
		cudaFree(dev_pBlind_Ram);
		cudaFree(dev_Histogram);
		cudaFree(dev_Histogram_float);
		return;
	}

	//应该每次都需要清空内存吧
	//cudaFree(dev_img);

}


// NETD计算：需要先采集低温，高温，按GPU，生成盲元表，然后计算NETD值
// 按一下采集低温下的数值，按第二下采集高温下数值
//int NETD_Vt = 0; int NETD_Vt0 = 0; int NETD_Vn = 0;
void CGrabDemoDlg::OnBnClickedNetd()
{
	CString str;
	str.Format(_T("%d"), NETD_frames);//固定格式
	GetDlgItem(IDC_Frame_Count)->SetWindowText(str);

	// 在采集低温本底的时候按一下，采集50帧低温数据。
	// 然后在采集高温本底的时候按一下，采集50帧高温数据，计算出NETD值
	Flag_NETD++;
	if (Flag_NETD == 1)
		CGrabDemoDlg::OnBnClickedTpCold();
	else
		CGrabDemoDlg::OnBnClickedTpHot();
	current = 0;  //初始化采集的帧数

	//--------保存图像-----------
	OnBnClickedSavemulti();
}

//计算各点的响应电压和噪声电压

void CGrabDemoDlg::CalculateNETD(int & flag,int Height, int Width)
{
	//第一个温度值采样
	if (flag == 1 && current < NETD_frames)
	{
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				//低温本底：响应电压平均值计算
				double tmp = (double(Image[j + i * Width]) - double(NETD_Vt[j + i*Width])) / (current + 1) + double(NETD_Vt[j + i*Width]);
				NETD_Vt[j + i*Width] = tmp;
			}
		}
		//-------------第一次采集完毕，给采用完成提示--------------
		if (current == NETD_frames - 1)
			GetDlgItem(IDC_NETDshow)->SetWindowText(_T("Ready"));
	}
	// 第二个温度值采样
	else if (flag == 2 && current < NETD_frames)
	{
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				//高温本底：响应电压平均值计算
				double tmp = (double(Image[j + i * Width]) - double(NETD_Vt0[j + i*Width])) / (current + 1) + double(NETD_Vt0[j + i*Width]);
				NETD_Vt0[j + i*Width] = tmp;
				//存储50帧高温电压数据，用于计算噪声电压
				NETD_Vn[j + i*Width][current] = Image[j + i * Width];
				//计算50帧噪声电压平均值
				//NETD_VnA[j + i*Width] = (Image[j + i * Width] - NETD_VnA[j + i*Width]) / (current + 1) + NETD_VnA[j + i*Width];
			}

		}
		//采集完毕后计算NETD值
		if (current == NETD_frames - 1)
		{
			// 计算各像元的噪声电压
			for (int i = 0; i < Height; i++)
			{
				for (int j = 0; j < Width; j++)
				{
					//计算噪声平均值
					double sum = 0;
					for (int k = 0; k < NETD_frames; k++)
					{
						sum = sum + pow(double(NETD_Vn[j + i*Width][k]) - double(NETD_Vt0[j + i*Width]), 2);  //计算方差：得到噪声电压
					}
					NETD_VnA[j + i*Width] = (1 / double(NETD_K)) * sqrt((sum / (NETD_frames - 1)));  //计算各像素点噪声电压
				}
			}
			//为排除过热像元，计算噪声电压均值，>2倍平均噪声电压的为过热像元，将盲元表值设为2，CPU中只进行盲元的替换，判断条件==1
			double Average_Noise = 0;
			for (int i = 0; i < Height; i++)
			{
				for (int j = 0; j < Width; j++)
				{
					Average_Noise = (double(NETD_VnA[j + i*Width]) - Average_Noise) / (j + i*Width + 1) + Average_Noise;
				}
			}
			Hot_num = 0;
			for (int i = 0; i < Height; i++)
			{
				for (int j = 0; j < Width; j++)
				{
					//计算过热像元
					if (NETD_VnA[j + i*Width] > 2 * Average_Noise && pBlind_Ram[j + i*Width] == 0)
					{
						pBlind_Ram[j + i*Width] = 2;
						Hot_num++;
					}

				}
			}

			//计算单个像元的NETD
			for (int i = 0; i < Height; i++)
			{
				for (int j = 0; j < Width; j++)
				{
					if (pBlind_Ram[j + i*Width] == 0)
					{
						NETD_Vt[j + i*Width] = (1 / double(NETD_K)) * double(NETD_Vt0[j + i*Width] - NETD_Vt[j + i*Width]);  //各像素响应电压
						NETD_Vt0[j + i*Width] = double(T - T0) / double(NETD_Vt[j + i*Width]) * double(NETD_VnA[j + i*Width]);  // 各点NETD 计算完毕
					}

				}
			}
			//----------NETD计算完成，求平均，输出到控件---------
			int count = 0;	double Aver = 0;
			for (int i = 0; i < Height; i++)
			{
				for (int j = 0; j < Width; j++)
				{
					if (pBlind_Ram[j + i*Width] == 0)
					{
						Aver = (double(NETD_Vt0[j + i*Width]) - Aver) / (count + 1) + Aver;  //计算平均NETD
						count++;
					}
				}
			}
			//NETDshow->SetWindowText(str);
			//((CButton*)m_DlgPointer->GetDlgItem(IDC_NETDshow));
			char   str[20];
			sprintf(str, "%.4f", 1000*Aver);//转换成字符串   
			// double   x=   atof(jidian)   ;//转换成浮点数
			int num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
			wchar_t *wide = new wchar_t[num];
			MultiByteToWideChar(0, 0, str, -1, wide, num);
			GetDlgItem(IDC_NETDshow)->SetWindowText(wide);

			flag = 0;
		}
	}
	else
	{
		return;
	}

	current++;//表示帧数增加
}

void CGrabDemoDlg::OnBnClickedOk()
{
	// --------------------调用I2C服务-------------------
	IMyCalc *pMyCalc = NULL;
	IUnknown *pUnknown = NULL;
	// 1.初始化COM库
	HRESULT hr = CoInitialize(NULL);
	// 2.根据已知ProgID找对应CLSID
	CLSID CalcID;
	hr = ::CLSIDFromProgID(L"MyCalc.math", &CalcID);   // 此处是开始设置的ID
	if (S_OK != hr)
		AfxMessageBox(_T("查找CalcID失败"));
	// 3.创建对应的接口实例
	hr = CoCreateInstance(CalcID, NULL, CLSCTX_LOCAL_SERVER, IID_IUnknown, (void **)&pUnknown);
	if (S_OK != hr)
		AfxMessageBox(_T("创建实例失败"));
	// 4.查询接口
	hr = pUnknown->QueryInterface(IID_IMyCalc, (void **)&pMyCalc);
	if (S_OK != hr)
		AfxMessageBox(_T("创建IMyCalc失败"));
	// 5.调用内部接口


	// TODO: 在此添加控件通知处理程序代码
	CString tmpString;
	char n_in = 2; char n_out = 3;  //输入多少读取多少？

	GetDlgItemText(I2C_nW, tmpString);
	n_in = _ttoi(tmpString);
	GetDlgItemText(I2C_nR, tmpString);
	n_out = _ttoi(tmpString);


	char InData[4];
	char OutData[3];
	// 第一字段默认已经输入了 0xA0，直接从第二字段开始写
	//InData[-1] = 0xA0;
	//-----------------读取输入数据--------------------
	GetDlgItemText(I2C_i0, tmpString);
	int num = wcstoll(tmpString, 0, 16);
	InData[0] = char(num);

	GetDlgItemText(I2C_i1, tmpString);
	num = _ttoi(tmpString);
	num = wcstoll(tmpString, 0, 16);
	InData[1] = char(num);

	GetDlgItemText(I2C_i2, tmpString);
	num = wcstoll(tmpString, 0, 16);
	InData[2] = char(num);

	GetDlgItemText(I2C_i3, tmpString);
	num = wcstoll(tmpString, 0, 16);
	InData[3] = char(num);
	
	//-------------------调用I2C 32位程序接口------------------------
	pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);

	//--------------------输出到窗口----------------
	if (n_out > 0)
	{
		num = OutData[0];
		tmpString.Format(_T("%X"), num);
		SetDlgItemText(I2C_o0, tmpString);
	}
	else
		SetDlgItemText(I2C_o0, L" ");
	if (n_out > 1)
	{
		num = OutData[1];
		tmpString.Format(_T("%X"), num);
		SetDlgItemText(I2C_o1, tmpString);
	}
	else
		SetDlgItemText(I2C_o1, L" ");
	if (n_out > 2)
	{
		num = OutData[2];
		tmpString.Format(_T("%X"), num);
		SetDlgItemText(I2C_o2, tmpString);
	}
	else
		SetDlgItemText(I2C_o2, L" ");

	//清理资源及反初始化COM组件
	pMyCalc->Release();
	pUnknown->Release();
	CoUninitialize();

}

// ------------   码率计算 ----------------
void CGrabDemoDlg::OnBnClickedBitrate()
{
	// --------------------调用I2C服务-------------------
	IMyCalc *pMyCalc = NULL;
	IUnknown *pUnknown = NULL;
	// 1.初始化COM库
	HRESULT hr = CoInitialize(NULL);
	// 2.根据已知ProgID找对应CLSID
	CLSID CalcID;
	hr = ::CLSIDFromProgID(L"MyCalc.math", &CalcID);   // 此处是开始设置的ID
	if (S_OK != hr)
		AfxMessageBox(_T("查找CalcID失败"));
	// 3.创建对应的接口实例
	hr = CoCreateInstance(CalcID, NULL, CLSCTX_LOCAL_SERVER, IID_IUnknown, (void **)&pUnknown);
	if (S_OK != hr)
		AfxMessageBox(_T("创建实例失败"));
	// 4.查询接口
	hr = pUnknown->QueryInterface(IID_IMyCalc, (void **)&pMyCalc);
	if (S_OK != hr)
		AfxMessageBox(_T("创建IMyCalc失败"));
	// 5.调用内部接口

	//--------------------------码率计算---------------------------
	//  码率：Ru = Rs * b * CRv * CRrs * (Tu/Ts)
	double Rs = double(6048) / 896; double B = 2; double CRv = double(1 / 2); double CRrs = double(188) / 204; double TuTs = double(4) / 5;
	char InData[4];
	char OutData[3];
	char n_in = 2; char n_out = 3;  //输入多少读取多少？

	//-----读取 Rs ------ 读取数据 0 : 2K---->1512/224uS  _________  1 : 8K  ----> 6048/896uS
	InData[0] = 0x1D;
	n_in = 2; n_out = 1;
	pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
	if (OutData[0] == 0)
		Rs = double(1512) / 224;
	else if (OutData[0] == 1)
		Rs = double(6048) / 896;
	else 
		MessageBox(_T("I2C Read Data Error !"));

	//-----读取 b ------
	InData[0] = 0x1E;
	n_in = 2; n_out = 1;
	pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
	if (OutData[0] == 0)
		B = 2;
	else if (OutData[0] == 1)
		B = 4;
	else if (OutData[0] == 2)
		B = 6;
	else
		MessageBox(_T("I2C Read Data Error !"));

	//-----读取 CRv ------
	InData[0] = 0x1F;
	n_in = 2; n_out = 1;
	pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
	if (OutData[0] == 0)
		CRv = double(1) / 2;
	else if (OutData[0] == 1)
		CRv = double(2) / 3;
	else if (OutData[0] == 2)
		CRv = double(3) / 4;
	else if (OutData[0] == 3)
		CRv = double(5) / 6;
	else if (OutData[0] == 4)
		CRv = double(7) / 8;
	else
		MessageBox(_T("I2C Read Data Error !"));

	//-----读取 Ts ------
	InData[0] = 0x20;
	n_in = 2; n_out = 1;
	pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
	if (OutData[0] == 0)
		TuTs = double(4) / 5;
	else if (OutData[0] == 1)
		TuTs = double(8) / 9;
	else if (OutData[0] == 2)
		TuTs = double(16) / 17;
	else if (OutData[0] == 3)
		TuTs = double(32) / 33;
	else
		MessageBox(_T("I2C Read Data Error !"));

	CString tmpString;
	double Ru = Rs * B * CRv * CRrs * TuTs;
	tmpString.Format(_T("%f"), Ru);
	SetDlgItemText(I2C_Rates, tmpString);

	//清理资源及反初始化COM组件
	pMyCalc->Release();
	pUnknown->Release();
	CoUninitialize();
}

//------------------------与下位机通信模块-----------------------
void CGrabDemoDlg::OnBnClickedOpen()
{
	// 获取串口选择下拉条选项
	int nIndex = m_Combo.GetCurSel();
	CString strCom; CString strRate;
	m_Combo.GetLBText(nIndex, strCom);

	// 获取串口传输速度选择下拉条选项
	nIndex = Comb_Rate.GetCurSel();
	Comb_Rate.GetLBText(nIndex, strRate);
	int Rate = _ttoi(strRate);

	//strtmp.Format(_T("COM:%d\n"), com_data);// 消息显示数字
	//MessageBox(strtmp);
	m_ComPortFlag = st.ThreadInit(strCom,Rate);
	if (m_ComPortFlag == true)
	{
		::MessageBox(NULL, _T("串口打开成功"), _T("提示"), MB_OK);
	}
	else
	{
		::MessageBox(NULL, _T("串口打开失败"), _T("提示"), MB_OK);
	}

	st.Com.SetWnd(this->GetSafeHwnd());   //串口消息传到当前窗口
}

// 下位机串口接收到数据的中断函数
LRESULT CGrabDemoDlg::SerialRead(WPARAM, LPARAM)
{

	//AfxMessageBox(_T("数据来啦"));
	SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
	int result = st.OnReceive();

	if (result == 0)
		FPGA_Receive();
	else
	{
		CString temp_value = _T("");   //temp_value用来处理int值
		temp_value.Format(_T("数据错误，代码：%d"), result);//固定格式
		AfxMessageBox(temp_value);
	}
	//st.OnReceive(readData);
	//if (readData == "123")
	//KillTimer(TIMER_ALARM);
	//st.Com.Purge(PURGE_RXCLEAR);
	//st.Com.Purge(PURGE_RXABORT);
	return 0;
}

void CGrabDemoDlg::OnBnClickedClose()
{
	// TODO: 在此添加控件通知处理程序代码
	st.CloseSerialPort();
	//::MessageBox(NULL, _T("串口已关闭"), _T("提示"), MB_OK);
	m_ComPortFlag = false;
}


void CGrabDemoDlg::OnBnClickedHe()
{
	FPGA_Send();
	if (m_ComPortFlag == true)
	{
		// 计时器1表示与下位机通信
		SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
		bool status = 0;
		CString HE;
		GetDlgItemText(FPGA_HE, HE); //取按钮标题
		if (HE == _T("增强_打开"))
		{
			status = st.SendCommand(COMMAND_PE_MODULE_CONTROL, 0xff);
			if (status)
				GetDlgItem(FPGA_HE)->SetWindowText(_T("增强_关闭"));
			return;
		}
		else
		{
			status = st.SendCommand(COMMAND_PE_MODULE_CONTROL, 0xf0);
			if (status)
				GetDlgItem(FPGA_HE)->SetWindowText(_T("增强_打开"));
			return;
		}
	}
	::MessageBox(NULL, _T("串口未打开，无法发送"), _T("提示"), MB_OK);
}

//下位机 采集低温本底
void CGrabDemoDlg::OnBnClickedGetlow()
{
	FPGA_Send();
	SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
	if (m_ComPortFlag == true)
	{
		bool status = st.SendCommand(COMMAND_NUC_CONTROL, 0x00);
		if (!status)		
			::MessageBox(NULL, _T("发送命令失败"), _T("提示"), MB_OK);
		return;
	}
	::MessageBox(NULL, _T("串口未打开，无法发送"), _T("提示"), MB_OK);
}
void CGrabDemoDlg::OnBnClickedGethigh()
{
	FPGA_Send();
	SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
	if (m_ComPortFlag == true)
	{
		bool status = st.SendCommand(COMMAND_NUC_CONTROL, 0x01);
		if (!status)
			::MessageBox(NULL, _T("发送命令失败"), _T("提示"), MB_OK);
		return;
	}
	::MessageBox(NULL, _T("串口未打开，无法发送"), _T("提示"), MB_OK);
}

void CGrabDemoDlg::OnBnClickedBlindCorrection()
{
	FPGA_Send();
	// 计时器1表示与下位机通信
	SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
	if (m_ComPortFlag == true)
	{
		bool status = 0;
		CString HE;
		GetDlgItemText(FPGA_Blind_Correction, HE); //取按钮标题
		if (HE == _T("盲元矫正_打开"))
		{
			status = st.SendCommand(COMMAND_DY_BADPOINT_MODULE_CONTROL, 0xff);
			if (status)
				GetDlgItem(FPGA_Blind_Correction)->SetWindowText(_T("盲元矫正_关闭"));
			return;
		}
		else
		{
			status = st.SendCommand(COMMAND_DY_BADPOINT_MODULE_CONTROL, 0xf0);
			if (status)
				GetDlgItem(FPGA_Blind_Correction)->SetWindowText(_T("盲元矫正_打开"));
			return;
		}
	}
	::MessageBox(NULL, _T("串口未打开，无法发送"), _T("提示"), MB_OK);
}
//下位机 两点矫正
void CGrabDemoDlg::OnBnClickedTpCorrection()
{
	FPGA_Send();
	// 计时器1表示与下位机通信
	SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
	if (m_ComPortFlag == true)
	{
		bool status = 0;
		CString HE;
		GetDlgItemText(FPGA_TP_Correction, HE); //取按钮标题
		if (HE == _T("两点矫正_打开"))
		{
			status = st.SendCommand(COMMAND_TWOPOINT_MODULE_CONTROL, 0xff);
			if (status)
				GetDlgItem(FPGA_TP_Correction)->SetWindowText(_T("两点矫正_关闭"));
			return;
		}
		else
		{
			status = st.SendCommand(COMMAND_TWOPOINT_MODULE_CONTROL, 0xf0);
			if (status)
				GetDlgItem(FPGA_TP_Correction)->SetWindowText(_T("两点矫正_打开"));
			return;
		}
	}
	::MessageBox(NULL, _T("串口未打开，无法发送"), _T("提示"), MB_OK);
}


void CGrabDemoDlg::OnBnClickedMedianFilter()
{
	FPGA_Send();
	// 计时器1表示与下位机通信
	SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
	if (m_ComPortFlag == true)
	{
		bool status = 0;
		CString HE;
		GetDlgItemText(FPGA_Median_Filter, HE); //取按钮标题
		if (HE == _T("中值滤波_打开"))
		{
			status = st.SendCommand(COMMAND_MID_MODULE_CONTROL, 0xff);
			if (status)
				GetDlgItem(FPGA_Median_Filter)->SetWindowText(_T("中值滤波_关闭"));
			return;
		}
		else
		{
			status = st.SendCommand(COMMAND_MID_MODULE_CONTROL, 0xf0);
			if (status)
				GetDlgItem(FPGA_Median_Filter)->SetWindowText(_T("中值滤波_打开"));
			return;
		}
	}
	::MessageBox(NULL, _T("串口未打开，无法发送"), _T("提示"), MB_OK);

}


void CGrabDemoDlg::OnBnClickedtest()
{
	// NETD测试，控制输入的图像

	//计算均值

	Flag_NETD = 1;
	for (int i = 0; i < 768; i++)
		for (int j = 0; j < 1024; j++)
			Image[j + i * 1024] = 80;
	m_DlgPointer->CalculateNETD(Flag_NETD, 768, 1024);

	

	//高温图像采集
	//计算均值
	current = 0;
	Flag_NETD = 2;
	for (int k = 0; k < 50; k++)
	{
		//低温图像采集
		if (k<16)
			for (int i = 0; i < 768; i++)
				for (int j = 0; j < 1024; j++)
					Image[j + i * 1024] = 99;
		else if (k<32)
			for (int i = 0; i < 768; i++)
				for (int j = 0; j < 1024; j++)
					Image[j + i * 1024] = 101;
		else
			for (int i = 0; i < 768; i++)
				for (int j = 0; j < 1024; j++)
					Image[j + i * 1024] = 100;
		m_DlgPointer->CalculateNETD(Flag_NETD, 768, 1024);
	}


}


void CGrabDemoDlg::OnBnClickedTest()
{
	// 计时器1表示与下位机通信
	SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
	for (int i = 0; i < 50000; i++)
		parameters[i] = i;
	bool status = st.SendCommand(COMMAND_UPDATE_BADPOINT_TABLE, parameters,50000);

}


void CGrabDemoDlg::OnBnClickedBpmap()
{
	FPGA_Send();
	// 向下位机传输上位机计算的盲元表
	SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
	int Height = m_DlgPointer->m_Buffers->GetHeight();//获取当前图像大小  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
	long k = 0;

	// 测试使用 ： 用于给盲元表自定义赋值
	//for (int i = 0; i < Height; i++)
	//{
	//	for (int j = 0; j < Width; j++)
	//		if (j % 2 == 0)
	//			pBlind_Ram[j + i * Width] = 1;
	//		else
	//			pBlind_Ram[j + i * Width] = 0;
	//}

	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
				pBlind_Ram[j + i * Width] = 0;
	}
	for (int i = 102; i <= 120; i++)
	{
		for (int j = 102; j <= 120; j++)
			pBlind_Ram[j + i * Width] = 1;
	}

	//根据盲元表转换为下位机能接受的数据格式
	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
		{
			if (pBlind_Ram[j + i * Width] > 0)
				parameters[k] = 1;
			else
				parameters[k] = 0;
			k++;
		}

	}

	//for (int i = 0; i < 50000; i++)
	//	parameters[i] = i;
	bool status = st.SendCommand(COMMAND_UPDATE_BADPOINT_TABLE, parameters, Height*Width);
}


void CGrabDemoDlg::OnBnClickedTpmap()
{
	FPGA_Send();
	// 向下位机传输上位机计算的两点校正表
	SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
	int Height = m_DlgPointer->m_Buffers->GetHeight();//获取当前图像大小  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
	
	//由于存储下位机可接收的数据类型
	//unsigned char parameters[1024 * 768];
	long k = 0;
	unsigned char* ptmp;
	//TP_Gain[0] = 0xa00b;
	//ptmp = (unsigned char*)&TP_Gain[0];

	//测试代码：
	for (int i = 0; i < 20; i++)
	{
		for (int j = 0; j < 20; j++)
		{
			TP_Gain[j + i * Width] = 2;
		}
	}
	for (int i = 50; i < 80; i++)
	{
		for (int j = 50; j < 80; j++)
		{
			TP_Gain[j + i * Width] = 0.5;
		}
	}
	for (int i = 100; i < 120; i++)
	{
		for (int j = 100; j < 120; j++)
		{
			TP_Bias[j + i * Width] = -5000;
		}
	}
	//根据矫正表转换为下位机能接受的数据格式 unsigned char 
	uint16_t tmpFloat;
	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
		{
			tmpFloat = Float16(TP_Gain[j + i * Width]);
			//tmpFloat = Float16(float(3.5));
			ptmp = (unsigned char*)&tmpFloat;
			parameters[k++] = *ptmp;
			parameters[k++] = *(ptmp + 1);
		}
	}
	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
		{
			tmpFloat = Float16(TP_Bias[j + i * Width]);
			ptmp = (unsigned char*)&tmpFloat;
			parameters[k++] = *ptmp;
			parameters[k++] = *(ptmp + 1);
		}
	}




	bool status = st.SendCommand(COMMAND_UPDATE_TWOPOINT_TABLE, parameters, Height*Width*4);
}


void CGrabDemoDlg::OnBnClickedIntegral()
{
	//-----将按键变灰色------
	FPGA_Send();

	// 获取连续采集的帧数
	CString nFrames;
	int numIntergral;		//采集的帧频数s
						//GetDlgItem(IDC_Frame_Count)->GetWindowTextW(nFrames);
	GetDlgItemText(FPGA_InterNum, nFrames);
	numIntergral = _ttoi(nFrames);
	if (numIntergral >= 100)
	{
		MessageBox(_T("The Input of Intergral Time Warning !"));
		return;
	}

	//----------发送数据--------------
	if (m_ComPortFlag == true)
	{
		// 计时器1表示与下位机通信
		SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
		bool status = 0;
		status = st.SendCommand(COMMAND_INTEGRATION_TIME_CONTROL, unsigned char(numIntergral));
		if (status)
			GetDlgItem(FPGA_HE)->SetWindowText(_T("增强_关闭"));
		return;
	}
	::MessageBox(NULL, _T("串口未打开，无法发送"), _T("提示"), MB_OK);
}




void CGrabDemoDlg::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值
	//int x=0, y=0;
	//POINT PP = m_DlgPointer->m_View->GetScrollPos();
	//m_DlgPointer->m_View->OnHScroll(x);
	//m_DlgPointer->m_View->OnVScroll(y);

	//CRect  rect;
	//GetClientRect(&rect);//获取客户区的大小
	////CPoint point;
	//GetCursorPos(&point);//获取当前指针的坐标（注意，这是屏幕的）
	//GetWindowRect(&rect);//获取客户区（客户区的左上角）相对于屏幕的位置
	//int x = (point.x - rect.left);//通过变换的到客户区的坐标  
	//int y = (point.y - rect.top);
/*	SIZE Rsize = m_DlgPointer->m_View->GetScrollRange();

	CString str;
	point.x = point.x - Rsize.cx;
	point.y = point.y - Rsize.cy;
	str = m_ImageWnd.GetPixelString(point);
	SetDlgItemText(PointValue, str);

	//str.Format(_T("x=%d,y=%d"), Rsize.cx, Rsize.cy);
	////str.Format("鼠标处于x=%d,y=%d的位置", point.x, point.y);
	//SetDlgItemText(PointValue, str);
*/
	__super::OnMouseMove(nFlags, point);
}


void CGrabDemoDlg::OnBnClickedChange1()
{
	// TODO: 在此添加控件通知处理程序代码
	// --------------------调用I2C服务-------------------
	IMyCalc *pMyCalc = NULL;
	IUnknown *pUnknown = NULL;
	// 1.初始化COM库
	HRESULT hr = CoInitialize(NULL);
	// 2.根据已知ProgID找对应CLSID
	CLSID CalcID;
	hr = ::CLSIDFromProgID(L"MyCalc.math", &CalcID);   // 此处是开始设置的ID
	if (S_OK != hr)
		AfxMessageBox(_T("查找CalcID失败"));
	// 3.创建对应的接口实例
	hr = CoCreateInstance(CalcID, NULL, CLSCTX_LOCAL_SERVER, IID_IUnknown, (void **)&pUnknown);
	if (S_OK != hr)
		AfxMessageBox(_T("创建实例失败"));
	// 4.查询接口
	hr = pUnknown->QueryInterface(IID_IMyCalc, (void **)&pMyCalc);
	if (S_OK != hr)
		AfxMessageBox(_T("创建IMyCalc失败"));
	// 5.调用内部接口

	//--------------------------码率计算---------------------------
	//  码率：Ru = Rs * b * CRv * CRrs * (Tu/Ts)
	double Rs = double(6048) / 896; double B = 2; double CRv = double(1 / 2); double CRrs = double(188) / 204; double TuTs = double(4) / 5;
	char InData[4];
	char OutData[3];
	char n_in = 2; char n_out = 3;  //输入多少读取多少？

	int nIndex = Combe_I2CMode.GetCurSel();
	CString strCom; 
	Combe_I2CMode.GetLBText(nIndex, strCom);

	switch (nIndex)
	{
	case 0:
	{
		//-----写 Rs ------ 2K模式
		InData[0] = 0x1D; InData[1] = 0x00;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 1:
	{
		//-----写 Rs ------ 8K模式
		InData[0] = 0x1D; InData[1] = 0x01;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 2:
	{
		//-----写 b ------ QPSK
		InData[0] = 0x1E; InData[1] = 0x00;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 3:
	{
		//-----写 b ------ 16-QAM
		InData[0] = 0x1E; InData[1] = 0x01;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 4:
	{
		//-----写 b ------ 64-QAM
		InData[0] = 0x1E; InData[1] = 0x02;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	default:
		AfxMessageBox(_T("未匹配到选项"));
		break;
	}

	//清理资源及反初始化COM组件
	pMyCalc->Release();
	pUnknown->Release();
	CoUninitialize();
}


void CGrabDemoDlg::OnBnClickedChange2()
{
	// TODO: 在此添加控件通知处理程序代码
	// --------------------调用I2C服务-------------------
	IMyCalc *pMyCalc = NULL;
	IUnknown *pUnknown = NULL;
	// 1.初始化COM库
	HRESULT hr = CoInitialize(NULL);
	// 2.根据已知ProgID找对应CLSID
	CLSID CalcID;
	hr = ::CLSIDFromProgID(L"MyCalc.math", &CalcID);   // 此处是开始设置的ID
	if (S_OK != hr)
		AfxMessageBox(_T("查找CalcID失败"));
	// 3.创建对应的接口实例
	hr = CoCreateInstance(CalcID, NULL, CLSCTX_LOCAL_SERVER, IID_IUnknown, (void **)&pUnknown);
	if (S_OK != hr)
		AfxMessageBox(_T("创建实例失败"));
	// 4.查询接口
	hr = pUnknown->QueryInterface(IID_IMyCalc, (void **)&pMyCalc);
	if (S_OK != hr)
		AfxMessageBox(_T("创建IMyCalc失败"));
	// 5.调用内部接口

	//--------------------------码率计算---------------------------
	//  码率：Ru = Rs * b * CRv * CRrs * (Tu/Ts)
	double Rs = double(6048) / 896; double B = 2; double CRv = double(1 / 2); double CRrs = double(188) / 204; double TuTs = double(4) / 5;
	char InData[4];
	char OutData[3];
	char n_in = 3; char n_out = 0;  //写3读0

	int nIndex = Combe_I2CBitSet.GetCurSel();
	CString strCom;
	Combe_I2CBitSet.GetLBText(nIndex, strCom);

	switch (nIndex)
	{
	case 0:
	{
		//-----写 内纠错（卷积）码率 CRv ------ 1/2
		InData[0] = 0x1F; InData[1] = 0x00;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 1:
	{
		//-----写 内纠错（卷积）码率 CRv ------ 2/3
		InData[0] = 0x1F; InData[1] = 0x01;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 2:
	{
		//-----写 内纠错（卷积）码率 CRv ------ 3/4
		InData[0] = 0x1F; InData[1] = 0x02;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 3:
	{
		//-----写 内纠错（卷积）码率 CRv ------ 5/6
		InData[0] = 0x1F; InData[1] = 0x03;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 4:
	{
		//-----写 内纠错（卷积）码率 CRv ------ 7/8
		InData[0] = 0x1F; InData[1] = 0x04;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	default:
		AfxMessageBox(_T("未匹配到选项"));
		break;
	}

	//清理资源及反初始化COM组件
	pMyCalc->Release();
	pUnknown->Release();
	CoUninitialize();
}


void CGrabDemoDlg::OnBnClickedChange3()
{
	// TODO: 在此添加控件通知处理程序代码
	// --------------------调用I2C服务-------------------
	IMyCalc *pMyCalc = NULL;
	IUnknown *pUnknown = NULL;
	// 1.初始化COM库
	HRESULT hr = CoInitialize(NULL);
	// 2.根据已知ProgID找对应CLSID
	CLSID CalcID;
	hr = ::CLSIDFromProgID(L"MyCalc.math", &CalcID);   // 此处是开始设置的ID
	if (S_OK != hr)
		AfxMessageBox(_T("查找CalcID失败"));
	// 3.创建对应的接口实例
	hr = CoCreateInstance(CalcID, NULL, CLSCTX_LOCAL_SERVER, IID_IUnknown, (void **)&pUnknown);
	if (S_OK != hr)
		AfxMessageBox(_T("创建实例失败"));
	// 4.查询接口
	hr = pUnknown->QueryInterface(IID_IMyCalc, (void **)&pMyCalc);
	if (S_OK != hr)
		AfxMessageBox(_T("创建IMyCalc失败"));
	// 5.调用内部接口

	//--------------------------码率计算---------------------------
	//  码率：Ru = Rs * b * CRv * CRrs * (Tu/Ts)
	char InData[4];
	char OutData[3];
	char n_in = 2; char n_out = 3;  //输入多少读取多少？

	int nIndex = Combe_I2C_TimeSet.GetCurSel();
	CString strCom;
	Combe_I2C_TimeSet.GetLBText(nIndex, strCom);

	switch (nIndex)
	{
	case 0:
	{
		// 保护间隔1/4, 1/8, 1/16, 1/32
		InData[0] = 0x20; InData[1] = 0x00;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 1:
	{
		//保护间隔1/8
		InData[0] = 0x20; InData[1] = 0x01;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 2:
	{
		//保护间隔1/16
		InData[0] = 0x20; InData[1] = 0x02;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 3:
	{
		//-----保护间隔1/32
		InData[0] = 0x20; InData[1] = 0x03;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	default:
		AfxMessageBox(_T("未匹配到选项"));
		break;
	}

	//清理资源及反初始化COM组件
	pMyCalc->Release();
	pUnknown->Release();
	CoUninitialize();
}

// 接收下位机消息批量处理函数：批量将按键变灰色
void CGrabDemoDlg::FPGA_Send()
{
	GetDlgItem(FPGA_GetLow)->EnableWindow(FALSE);
	GetDlgItem(FPGA_GetHigh)->EnableWindow(FALSE);
	GetDlgItem(FPGA_Blind_Correction)->EnableWindow(FALSE);
	GetDlgItem(FPGA_HE)->EnableWindow(FALSE);
	GetDlgItem(FPGA_TP_Correction)->EnableWindow(FALSE);
	GetDlgItem(FPGA_Median_Filter)->EnableWindow(FALSE);
	GetDlgItem(FPGA_BPmap)->EnableWindow(FALSE);
	GetDlgItem(FPGA_TPmap)->EnableWindow(FALSE);
	GetDlgItem(FPGA_Integral)->EnableWindow(FALSE);
	GetDlgItem(FPGA_NU)->EnableWindow(FALSE);
}

void CGrabDemoDlg::FPGA_Receive()
{
	GetDlgItem(FPGA_GetLow)->EnableWindow(TRUE);
	GetDlgItem(FPGA_GetHigh)->EnableWindow(TRUE);
	GetDlgItem(FPGA_Blind_Correction)->EnableWindow(TRUE);
	GetDlgItem(FPGA_HE)->EnableWindow(TRUE);
	GetDlgItem(FPGA_TP_Correction)->EnableWindow(TRUE);
	GetDlgItem(FPGA_Median_Filter)->EnableWindow(TRUE);
	GetDlgItem(FPGA_BPmap)->EnableWindow(TRUE);
	GetDlgItem(FPGA_TPmap)->EnableWindow(TRUE);
	GetDlgItem(FPGA_Integral)->EnableWindow(TRUE);
	GetDlgItem(FPGA_NU)->EnableWindow(TRUE);
}



void CGrabDemoDlg::OnBnClickedNu()
{
	// TODO: 在此添加控件通知处理程序代码
	FPGA_Send();
	SetTimer(1, 50000, NULL);  //  单位：ms
	if (m_ComPortFlag == true)
	{
		bool status = st.SendCommand(COMMAND_NUC_CONTROL, 0x02);
		if (!status)
			::MessageBox(NULL, _T("发送命令失败"), _T("提示"), MB_OK);
		return;
	}
	::MessageBox(NULL, _T("串口未打开，无法发送"), _T("提示"), MB_OK);
}


void CGrabDemoDlg::OnBnClickedUpdate()
{
	// TODO: 在此添加控件通知处理程序代码
	CGrabDemoDlg::FPGA_Receive();
}

// 用于添加CUDA算法的接口
void CGrabDemoDlg::CUDA_Algorithm()
{
	//判断是否有运行算法，没有则返回
	//--------算法开关-------
	int Bilateral_On = ((CButton*)m_DlgPointer->GetDlgItem(PC_bilateralFilter))->GetCheck();
	int Enhance_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_H_Enhance))->GetCheck();

	if (!Bilateral_On && !Enhance_On)
		return;
	//------------使用opencv+CUDA的写法---------
	int Height = m_DlgPointer->m_Buffers->GetHeight();//获取当前图像大小  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024


	// 将16 bits图像转为 8bits进行 opencv + cuda 处理
	for (int i = 0; i < Height; i++)
		for (int j = 0; j < Width; j++)
			rImage_uchar[i * Width + j] = rImage[i * Width + j] >> 8;
	// src为原图像，gray为处理后图像，hist为直方图
	Mat src_host = Mat(Height, Width, CV_8UC1, rImage_uchar);
	Mat gray_host;
	Mat hist_host;
	GpuMat src, gray, hist;

	// 双边滤波算法实现：
	if (Bilateral_On > 0)
	{		
		src.upload(src_host);
		cv::cuda::bilateralFilter(src, gray, 8, 15, 15);
		gray.download(gray_host);
	}
	//直方图增强算法实现
	if (Enhance_On)
	{
		//没进行双边滤波，直接调用原图 || 如果已经执行双边滤波，则直接上传原图像到gray
		if (Bilateral_On <= 0)
		{
			gray.upload(src_host);
			gray.download(gray_host);
		}		
		cv::cuda::calcHist(gray, hist);
		hist.download(hist_host);
		//-------双平台矫正 针对0-255灰度级 -------
		int L = 0; //用于存储直方图的非零项数目
		double Sum = 0;
		int count = 0;
		//float* data;
		//data = hist_host.ptr<float>(0);
		int tmp_hist[256];
		for (int i = 0; i < 256; i++)
			tmp_hist[i] = hist_host.at<int>(0,i);
		if (tmp_hist[0] > 0)
			L++;
		if (tmp_hist[1] > 0)
			L++;

		for (int i = 2; i < 254; i++)
		{
			if (tmp_hist[i-1] >= tmp_hist[i + 2] && tmp_hist[i + 1] >= tmp_hist[i-2])
			{
				Sum += tmp_hist[i - 1] + tmp_hist[i] + tmp_hist[i + 1];
				count++;
			}

			if (tmp_hist[i] > 0)
				L++;
		}
		int Tup = 3000;// Sum / count;
		int Tdown = 10;// min(Height*Width, Tup*L) / 256;
		//------- 用双平台门限矫正 矫正直方图 ------------

		Sum = 0; //重新统计总数，用于后续计算概率直方图
		for (int i = 0; i < 256; i++)
		{
			if (tmp_hist[i] < Tdown)
				tmp_hist[i] = Tdown;
			else if (tmp_hist[i] > Tup)
				tmp_hist[i] = Tup;
			Sum += tmp_hist[i];
		}
		// 绘制概率密度图
		for (int i = 1; i < 256; i++)
		{
			tmp_hist[i] = tmp_hist[i] + tmp_hist[i - 1];
		}
		//-------- 矫正图像 ----------
		unsigned char* inData = src_host.ptr<unsigned char>(0); //原图
		unsigned char* outData = gray_host.ptr<unsigned char>(0); //处理后图像
		int tmp;
		for (int i = 0; i < Height; i++)
			for (int j = 0; j < Width; j++)
			{
				tmp = tmp_hist[inData[i * Width + j]];
				tmp = tmp_hist[inData[i * Width + j]] * 255 / Sum;
				outData[i * Width + j] = unsigned char(tmp_hist[inData[i * Width + j]] * 255 / Sum);
				//outData[i * Width + j] = unsigned char(tmp_hist[inData[i * Width + j]] * 255 / Sum);
				tmp = outData[i * Width + j];
			}

		//gray.download(gray_host);
	}


	//------将数据格式 8 bits 变为 16 bits ------------
	unsigned char* rdata;
	rdata = gray_host.ptr<unsigned char>(0);
	for (int i = 0; i < Height; i++)
		for (int j = 0; j < Width; j++)
			rImage[i * Width + j] = rdata[i * Width + j] << 8;
	//m_DlgPointer->m_Buffers->WriteRect(0, 0, Width, Height, rdata);
	src.release(); gray.release(); src_host.release(); gray_host.release();
}


void CGrabDemoDlg::OnBnClickedWinchange()
{
	// 控制下位机进行开窗处理
	//-----将按键变灰色------
	FPGA_Send();

	// 获取连续采集的帧数
	CString nFrames;
	int Win_H, Win_W;		//采集的帧频数s
							//GetDlgItem(IDC_Frame_Count)->GetWindowTextW(nFrames);
	GetDlgItemText(FPGA_WinHeight, nFrames);
	Win_H = _ttoi(nFrames);
	GetDlgItemText(FPGA_WinWidth, nFrames);
	Win_W = _ttoi(nFrames);

	// 窗口大小限制：0 <= H（Y坐标）<=511  ||   128 <= W (X坐标) <= 639
	if (Win_H < 0 || Win_H>511 || Win_W < 128 || Win_W > 639)
	{
		MessageBox(_T("The Window Size is Out of Range !"));
		return;
	}
	int index = 0;

	//------传输坐标数据：data[0],[1]为Ymin，[2],[3]为Ymax，[4]为Xmin，[5]为Xmax 都是八位数据传输
	transfer_data[index++] = 0x00;
	transfer_data[index++] = 0x00;
	unsigned short tmpValue = unsigned short(Win_H);
	transfer_data[index++] = tmpValue;
	transfer_data[index++] = tmpValue >> 8;
	transfer_data[index++] = 0x00;
	transfer_data[index++] = unsigned short(Win_W / 4); 

	//----------发送数据--------------
	if (m_ComPortFlag == true)
	{
		// 计时器1表示与下位机通信
		SetTimer(1, Serial_Delay_Time, NULL);  //  单位：ms
		bool status = 0;
		status = st.SendCommand(COMMAND_WINDOW_MODE_SWITCH, transfer_data, 6);
		if (status)
			GetDlgItem(FPGA_HE)->SetWindowText(_T("增强_关闭"));
		return;
	}
	::MessageBox(NULL, _T("串口未打开，无法发送"), _T("提示"), MB_OK);
}


void CGrabDemoDlg::OnBnClickedLocalEnlarge()
{
	// TODO: 在此添加控件通知处理程序代码
}


void CGrabDemoDlg::OnStnClickedViewWnd2()
{
	// TODO: 在此添加控件通知处理程序代码
}


void CGrabDemoDlg::OnBnClickedCheck1()
{
	// TODO: 在此添加控件通知处理程序代码
}


void CGrabDemoDlg::OnBnClickedDepthCheck()
{
	// TODO: 在此添加控件通知处理程序代码
}
