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

// GPU ͷ�ļ�
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  

// ����λ��ͨ��
#include "IRCMD_COM/SerialThread.h"
#include "IRCMD_COM/ComFrame.h"


// I2Cͨ��exe����ͷ�ļ�
#include "E:\INLF\QZB\����ͨ��\USBתI2Cͨ��\ATL_test\ATLCOMProject\ATLCOMProject\ATLCOMProject_i.h"
#include "E:\INLF\QZB\����ͨ��\USBתI2Cͨ��\ATL_test\ATLCOMProject\ATLCOMProject\ATLCOMProject_i.c"

//  Float32 ת float16
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

//// GPU ��������
//extern "C"
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

CGrabDemoDlg *CGrabDemoDlg::m_DlgPointer = NULL; //��̬����ָ��,��Ҫ��ǰ��ʼ��
MyTensorRT* CGrabDemoDlg::m_tensorRT = nullptr;
unsigned short rImage[1280*1024] = { 0 };   //GPU�����ͼ������
unsigned short *Image = rImage;

unsigned char rImage_uchar[1280 * 1024] = { 0 };   //GPU�����ͼ������
unsigned char *Image_uchar = rImage_uchar;

unsigned char mImage[640 * 512] = { 0 };   //�ֲ��Ŵ��ͼ������
unsigned char *mimage = mImage;

unsigned int rHistogram[65536] = {0};  //����CPU����ֱ��ͼ
unsigned int *Histogram = rHistogram;
float rHistogram_Float[65536] = { 0 };  //����CPU����ֱ��ͼ
float *Histogram_Float = rHistogram_Float;

unsigned short rCold_Ram[1280 * 1024] = { 0 };   //���±��׵�ͼ������
unsigned short *Cold_Ram = rCold_Ram;

unsigned short rHot_Ram[1280 * 1024] = { 20000 };   //���±��׵�ͼ������
unsigned short *Hot_Ram = rHot_Ram;

float rTP_Gain[1280 * 1024] = { 0 };		//��������ĸ�����������
float *TP_Gain = rTP_Gain;

float rTP_Bias[1280 * 1024] = { 0 };   //��������ĸ�������ƫ��
float *TP_Bias = rTP_Bias; 

unsigned short rBlind_Ram[1280 * 1024] = { 0 };   //äԪ������
unsigned short *pBlind_Ram = rBlind_Ram;

//--------��ʼ��GPU ָ�� -----------
unsigned short *dev_img = 0;   //GPU ͼ��ָ��
float *dev_pTP_Gain = 0, *dev_pTP_Bias = 0; //GPU ������������ƫ��
unsigned short *dev_pBlind_Ram = 0;  //GPU äԪ������
unsigned int *dev_Histogram = 0;   //GPU ֱ��ͼ����
float *dev_Histogram_float = 0; //GPU ֱ��ͼ����

//-------------NETD��������-------------
long Death_num = 0; long Hot_num = 0;
int NETD_frames = 50;   //���ò���
int NETD_K = 1; 
float T0 = 20; float T = 35;  // NETD: ��һ�βɼ����¶�  �ڶ��βɼ����¶�

int Flag_NETD = 0;   //�Ƿ�ִ��NETD����
float NETD_Vt[1280 * 1024] = { 0 };   //NETD
float NETD_Vt0[1280 * 1024] = { 0 };   //NETD
float NETD_Vn[1280 * 1024][50] = { 0 };   //NETD
float NETD_VnA[1280 * 1024] = { 0 };   //NETD ƽ������ֵ
int current = 0;  //��ʾ��ǰ��ȡ��ͼ��֡��

int frame_count;  // ���ڼ�¼֡Ƶ��ÿ����� 

//��λ��ͨ���õ���������ת��������
unsigned char parameters[1024 * 1280 * 4 ];
unsigned char transfer_data[6]; //��������
//int NETD_Vt = 0; int NETD_Vt0 = 0; int NETD_Vn = 0;
//-------------------------------------------------

//-----------���ڳ�ʱ�ش�ʱ����--------------
unsigned int Serial_Delay_Time = 10000;  //  xxx ms
cudaError_t cudaStatus;

//float rHistogram_Enhancement[65536] = { 0 };   //ֱ��ͼ��ǿ��
//float *pHistogram_Enhancement = rHistogram_Enhancement;

int N_Saved_Frames = 0;
int Current_Saved = 0;
//------------------��λ��ͨ�Ų��ֳ�ʼ��---------
// ����״̬ �� �򿪣�1
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
   m_DlgPointer = this; //����ָ�룬���캯��ʽָ��this
   //m_cstrWorkPath = "H:\\QZB\\Sapera\\Demos\\Classes\\Vc\\GrabDemo\\Data"; //�������ݴ洢λ��
   //m_cstrWorkPath = "H:\\QZB\\Sapera\\Demos\\Classes\\Vc\\subData"; //�������ݴ洢λ��
   m_cstrWorkPath = "E:\\INLF\\QZB\\Sapera\\Demos\\Classes\\Vc\\Raw20241112";//
   m_IsSignalDetected = TRUE;

}

void CGrabDemoDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CGrabDemoDlg)
	DDX_Control(pDX, IDC_STATUS, m_statusWnd);
	DDX_Control(pDX, IDC_VIEW_WND, m_ImageWnd);
	DDX_Text(pDX, IDC_BUFFER_FRAME_RATE, m_BufferFrameRate);  // ��ȡ֡Ƶ����
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
	ON_WM_TIMER()  //��Ӷ�ʱ����Ҫ
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

	ON_MESSAGE(ON_COM_RXCHAR, SerialRead)   //���ڽ�����Ӧӳ��
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

   //___________________��Ӵ���ͼ����______________
   int Height = m_DlgPointer->m_Buffers->GetHeight();//��ȡ��ǰͼ���С  768
   int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
   int Pixel_Depth = m_DlgPointer->m_Buffers->GetPixelDepth();
   CString cs;
   //��������ַ���

   //-------------------��ʾ֡Ƶ------------------
   //SapXferFrameRateInfo* pFrames;
   //pFrames = m_DlgPointer->m_Xfer->GetFrameRateStatistics();
   //float current_frames;
   //current_frames = pFrames->GetBufferFrameRate();

   frame_count++;  // ֡Ƶ������1
   //char   str[10];
			//				// double   x=   atof(jidian)   ;//ת���ɸ�����
   //int num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
   //wchar_t *wide = new wchar_t[num];
   //MultiByteToWideChar(0, 0, str, -1, wide, num);
   //((CButton*)m_DlgPointer->GetDlgItem(FPGA_frames))->SetWindowText(wide);
   //------------------------------------------------

   //cs.Format(_T("Image Size: %d * %d----%d"), Height, Width,Pixel_Depth);
   //pDlg->MessageBox(cs);
   
 
   //---------��ȡͼ�񣬱���ΪImage��768*1024��max 
   //BOOL ReadRect(int index, int x, int y, int width, int height, void* pData);
   m_DlgPointer->m_Buffers->ReadRect(0, 0, Width, Height, Image);
 /*  //------------ͼ�����㷨���Դ���---------
   // �Զ���ͼ��
   for (int i = 0; i < 768; i++)
	   for (int j = 0; j < 1024; j++)
		   Image[j + i * 1024] = 40000;
   //------�����������------
   for (int i = 0; i < 50; i++)
	   for (int j = 0; j < 50; j++)
		   Image[j + i * 1024] = 20000;

   //-------äԪ����-----
   for (int i = 700; i < 768; i = i + 3)
	   for (int j = 980; j < 1024; j = j + 3)
		   Image[j + i * 1024] = 0;
*/
   // ͼ������ѡ�����
   int TP_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_TP_Correction))->GetCheck();
   int Blind_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_Blind_Correction))->GetCheck();
   int Histogram_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_H_Enhance))->GetCheck();
   

   /*33��33333333333333333333333333
    //------��������������̶�����ͼ��-----
   for (int i = 0; i < 768; i++)
	   for (int j = 0; j < 1024; j++)
		   Image[j + i * 1024] = 40000;

   for (int i = 0; i < 50; i++)
	   for (int j = 0; j < 50; j++)
		   Image[j + i * 1024] = 20000;

   //-------------äԪ����---------
   for (int i = 700; i < 768; i = i + 3)
	   for (int j = 980; j < 1024; j = j + 3)
		   Image[j + i * 1024] = 0;  
   
   //---------�Ǿ��Ƚ�������------
   for (int i = 0; i < 768; i++)
	   for (int j = 0; j < 1024; j++)
		   Image[j + i * 1024] = 20000 * i / 768;
		   */
   for (int i = 0; i < Height; i++)
	   for (int j = 0; j < Width; j++)
		   mImage[i * Width + j] = rImage[i * Width + j] >> 8;
   // srcΪԭͼ��grayΪ�����ͼ��histΪֱ��ͼ
   Mat gray_host = Mat(Height, Width, CV_8UC1, mImage);
   Mat flipped_both;
   // ���º����ҷ�ת
   cv::flip(gray_host, flipped_both, -1);
   gray_host = flipped_both;


   //Mat gray_host1_2;

   ////------------------------------------------------------------------------------
   //int cropHeight = 512; // ���磬�ü�����߶�Ϊԭʼ�߶ȵ�һ��
   //int cropWidth = 193;   // �ü�������Ϊԭʼ��ȵ�һ��						   
   //int startHeight1_2 = 0; // ����ü��������ʼ��
   //int startWidth1_2 = 168;     // ����ü��������ʼ��
   //float ini;
   //cv::Rect cropRegion1_2(startWidth1_2, startHeight1_2, cropWidth, cropHeight);
   //gray_host1_2 = gray_host(cropRegion1_2);
   ////------------------------------------------------------------------------------
   //gray_host = gray_host;
   // ����һ����Ŀհ�ͼ�������ƴ�ӽ��  
   //gray_host1_2.copyTo(gray_host(cv::Rect(447, 0, 193, 512)));

   //for (int i = 0; i < 512; i++) {
	  // for (int j = 440; j < 460; j++) {
		 //  gray_host.at<uchar>(i, j) = (gray_host.at<uchar>(i, j - 3) + gray_host.at<uchar>(i, j - 2) + gray_host.at<uchar>(i, j - 1) + gray_host.at<uchar>(i, j) + \
			//   gray_host.at<uchar>(i, j + 1) + gray_host.at<uchar>(i, j + 2) + gray_host.at<uchar>(i, j + 3)) / 7.0;
	  // }

   //}
   //�����ƴ�ӣ��Ź��ˣ�

   if (BST_CHECKED == ((CButton*)pDlg->GetDlgItem(IDC_DETECTION_CHECK))->GetCheck())
   {
	   // ����Ҫ�ָ��������MATLAB�ű���ͬ��
	   std::vector<std::vector<int>> regions = {
		   {1, 150, 1, 170},     // ����1: 1-150��, 1-170��
		   {1, 150, 210, 420},   // ����2: 1-150��, 210-420��
		   {1, 150, 460, Width}, // ����3: 1-150��, 460�����һ��
		   {171, 350, 1, 170},   // ����4: 171-350��, 1-170��
		   {171, 350, 210, 420}, // ����5: 171-350��, 210-420�� (������ͼ)
		   {171, 350, 460, Width}, // ����6: 171-350��, 460�����һ��
		   {371, Height, 1, 170}, // ����7: 370�����һ��, 1-170��
		   {371, Height, 210, 420}, // ����8: 370�����һ��, 210-420��
		   {371, Height, 460, Width} // ����9: 370�����һ��, 460�����һ��
	   };

	   // ����һ��vector���洢�ָ���ͼ��
	   std::vector<Mat> splitImgs(9);

	   // �ٳ�ÿ�������ͼ��
	   for (int i = 0; i < 9; i++) {
		   // ע�⣺OpenCV������������0��ʼ����Ҫ��1
		   int startRow = regions[i][0] - 1;
		   int endRow = regions[i][1] - 1;
		   int startCol = regions[i][2] - 1;
		   int endCol = regions[i][3] - 1;

		   // ȷ����������Ч��Χ��
		   startRow = std::max(0, startRow);
		   endRow = std::min(Height - 1, endRow);
		   startCol = std::max(0, startCol);
		   endCol = std::min(Width - 1, endCol);

		   // ʹ��OpenCV��Rect��������Ȥ����(ROI)
		   cv::Rect roi(startCol, startRow, endCol - startCol + 1, endRow - startRow + 1);
		   splitImgs[i] = gray_host(roi).clone();  // ��¡�Դ�������ͼ��
	   }

	   // �ҵ���С�ߴ��ͼ��
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

	   // ��ȡ��С�ߴ�
	   minHeight = splitImgs[minIndex].rows;
	   minWidth = splitImgs[minIndex].cols;

	   // �ü�����ͼ����С�ߴ磨���вü���
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

	   // ʹ��������ͼ������5���滻ԭʼͼ��
	   // ע�⣺����4��Ӧ����5��C++��0��ʼ������
	   Mat centerView = splitImgs[4];
	   Mat resizedCenterView;
	   cv::resize(centerView, resizedCenterView, cv::Size(Width, Height), 0, 0, cv::INTER_LINEAR);
	   // ����һ����ԭʼͼ����ͬ��С�Ŀհ�ͼ��
	   Mat resultImage = Mat::zeros(Height, Width, CV_8UC1);

	   // ��������ͼ�����ڽ��ͼ�������λ��
	   int startRow = (Height - resizedCenterView.rows) / 2;
	   int startCol = (Width - resizedCenterView.cols) / 2;

	   // ȷ��λ����Ч
	   startRow = std::max(0, startRow);
	   startCol = std::max(0, startCol);

	   // ����ʵ�ʿ��Է��õĳߴ�
	   int actualHeight = std::min(resizedCenterView.rows, Height - startRow);
	   int actualWidth = std::min(resizedCenterView.cols, Width - startCol);

	   // ����������ͼ�����ͼ��
	   cv::Rect roi(startCol, startRow, actualWidth, actualHeight);
	   Mat resultROI = resultImage(roi);
	   resizedCenterView(cv::Rect(0, 0, actualWidth, actualHeight)).copyTo(resultROI);

	   // ������ʾ��ͼ��
	   gray_host = resultImage;
	   DWORD start, end;
	   DWORD engineProcessTime;
	   //Ԥ����
	   start = GetTickCount();
	   m_tensorRT->preprocessImage(gray_host);
	   // ִ������
	   m_tensorRT->inference(1);
	   // ����
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
	   // 1. ��ȡԭʼͼ��ߴ�
	   int Width = gray_host.cols;
	   int Height = gray_host.rows;

	   // 2.���򻯡�ֱ�Ӷ��岢��ȡ��������������ͼ
	   // ����5������ (1-based): {171, 350, 210, 420}
	   int startCol = 210 - 1;
	   int startRow = 171 - 1;
	   int roiWidth = (420 - 1) - startCol + 1;
	   int roiHeight = (350 - 1) - startRow + 1;

	   // �߽��飬��ֹROI����ͼ��Χ
	   if (startCol + roiWidth > Width || startRow + roiHeight > Height) {
		   // ����ѡ�񱨴�����ROI�ߴ�
		   return; // ��ʱ�򵥷���
	   }

	   cv::Rect centerViewROI(startCol, startRow, roiWidth, roiHeight);
	   // ֱ�Ӵ�ԭʼ������ͼ������ȡROI����ʹ��clone()�����Ǻ������޸�gray_host
	   cv::Mat centerView = gray_host(centerViewROI);
	   // ����һ���µ�Mat�������洢RGBͼ��
	   cv::Mat rgbImage;

	   // ���Ҷ�ͼ��ת��ΪRGBͼ��
	   cv::cvtColor(centerView, rgbImage, cv::COLOR_GRAY2BGR);
	   // 3.���������Ը�������������ͼ��������
	   DWORD start, end;
	   DWORD engineProcessTime;
	   start = GetTickCount();

	   // ��ԭʼ�������� centerView ���룬��� TensorRT �����ȷ��������
	   m_tensorRT->preprocessImage_Depth(rgbImage);
	   m_tensorRT->inference(1);
	   // ����õ��Ľ�� depth_result �Ƕ�Ӧ centerView �ߴ�����ͼ
	   cv::Mat depth_result = m_tensorRT->postprocessOutput_Depth(1);

	   end = GetTickCount();
	   engineProcessTime = end - start;

	   // 4.����������ȷ����ʾ���ͼ���
	   // ����һ����ԭʼ��ͼ��ͬ�ߴ�ĺ�ɫ����
	   // ע�⣺���ͼ��3ͨ����ɫ��(CV_8UC3)�����Ա���ҲӦ����3ͨ��
	   //cv::Mat resultImage = cv::Mat::zeros(Height, Width, CV_8UC3);

	   //// ������������ͼ������centerView�ߴ���ͬ������������ԭͼ�е�λ��
	   //depth_result.copyTo(resultImage(centerViewROI));

	   // ������ʾ��ͼ��ע�� gray_host ���ڻ��ɲ�ɫͼ
	   cv::Mat final_display_img;
	   cv::cvtColor(depth_result, final_display_img, cv::COLOR_BGR2GRAY);
	   // ����final_display_img�Ĵ�СΪԭʼͼ��Ĵ�С
	   cv::Mat resizedFinalDisplayImg;
	   cv::resize(final_display_img, resizedFinalDisplayImg, cv::Size(Width, Height), 0, 0, cv::INTER_LINEAR);
	   gray_host = resizedFinalDisplayImg;
   }
   unsigned char* rdata;
   rdata = gray_host.ptr<unsigned char>(0);
   for (int i = 0; i < Height; i++)
	   for (int j = 0; j < Width; j++)
		   rImage[i * Width + j] = rdata[i * Width + j] << 8;
   //----------------   GPUִ�г���   -------------------
   if (BST_CHECKED == ((CButton*)m_DlgPointer->GetDlgItem(IDC_GPU))->GetCheck())
   {
	   //----------GPU- ���Դ���------------

	   // GPU ������ �� ͼ��ָ�룬ͼ���ͼ�񳤣�ʹ��GPU�߳���
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
   // ����cuda�㷨����ֱ�ӽ�������浽ȫ�ֱ���rImage�С�
 /*  m_DlgPointer->CUDA_Algorithm();*/
   if (((CButton*)m_DlgPointer->GetDlgItem(IDC_Local_Enlarge))->GetCheck()) {
	   m_DlgPointer->localEnlarge(Height, Width);
	   m_DlgPointer->m_Buffers->WriteRect(0, 0, Width, Height, Image);
   }
   pDlg->m_View->Show();

	   
   //---------�򻺴�����дͼ��-----
   // BOOL WriteRect(int x, int y, int width, int height, const void* pData);
   



   // If grabbing in trash buffer, do not display the image, update the
   // appropriate number of frames on the status bar instead
 
   //_________�ж��Ƿ�Ϊ����������ʱû���á���ɾ��_____
/*   if (pInfo->IsTrash())  //�ж��Ƿ�Ϊ��������
   {
      CString str;
      str.Format(_T("Frames acquired in trash buffer: %d"), pInfo->GetEventCount());
      pDlg->m_statusWnd.SetWindowText(str);
   }
   
   // Refresh view ��Ҫ��ʾ��ͼ����б���
   else
   {
	   //-------------------����ɼ�ͼ��----------------
	   //---------����ͼƬ�ĸ�ʽ--------
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

   //-----------NETD ����-------------
   m_DlgPointer->CalculateNETD(Flag_NETD, Height, Width);
   
	   //-------------------����ɼ�ͼ��----------------
	   //---------����ͼƬ�ĸ�ʽ--------
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
	  //��ʾͼ��
   
   
}
//�������
void CGrabDemoDlg::localEnlarge(int Height, int Width)
{
	for (int i = 0; i < Height; i++)
		for (int j = 0; j < Width; j++)
			mImage[i * Width + j] = rImage[i * Width + j] >> 8;
	// srcΪԭͼ��grayΪ�����ͼ��histΪֱ��ͼ
	Mat src_host = Mat(Height, Width, CV_8UC1, mImage);

	Mat gray_host;
	gray_host = distortionCailbration.process(src_host);
	////-------------------------֮ǰ--------------------------
	//int cropHeight = Height / 2; // ���磬�ü�����߶�Ϊԭʼ�߶ȵ�һ��
	//int cropWidth = Width / 2;   // �ü�������Ϊԭʼ��ȵ�һ��
	//int startHeight = (Height - cropHeight) / 2; // ����ü��������ʼ��
	//int startWidth = (Width - cropWidth) / 2;     // ����ü��������ʼ��

	//											  // �ü�ͼ��
	//cv::Rect cropRegion(startWidth, startHeight, cropWidth, cropHeight);
	//gray_host = src_host(cropRegion);
	//cv::resize(gray_host, gray_host, cv::Size(Width, Height));
	////---------------------------֮ǰ----------------------------
 
	

	//------ ������֣� �����ݸ�ʽ 8 bits ��Ϊ 16 bits ------------
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
      m_Xfer		= new SapAcqToBuf(m_Acq, m_Buffers, XferCallback, this);  //ִ�в���buffer������ص�����
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

   //--------���ڲ˵�����ʼ��------------
   m_Combo.AddString(_T("Xcelera-CL_PX4_1_Serial_0"));
   m_Combo.AddString(_T("Xcelera-CL_PX4_1_Serial_1"));
   m_Combo.AddString(_T("COM2"));
   m_Combo.SetCurSel(0);//��ʼʱ�����б�ΪCOM2

   //-------���ڴ�������--------------
   Comb_Rate.InsertString(0,_T("9600"));
   Comb_Rate.InsertString(1, _T("115200"));
   Comb_Rate.SetCurSel(0);//��ʼʱ�����б�Ϊ9600

   //----------I2C ����----------
   Combe_I2CMode.InsertString(0,_T("2Kģʽ"));
   Combe_I2CMode.InsertString(1,_T("8Kģʽ"));
   Combe_I2CMode.InsertString(2,_T("QPSK"));
   Combe_I2CMode.InsertString(3,_T("16-QAM"));
   Combe_I2CMode.InsertString(4,_T("64-QAM"));
   Combe_I2CMode.SetCurSel(0);//��ʼʱ�����б�Ϊ9600

   Combe_I2CBitSet.InsertString(0,_T("�ھ������ʣ�1/2"));
   Combe_I2CBitSet.InsertString(1,_T("�ھ������ʣ�2/3"));
   Combe_I2CBitSet.InsertString(2,_T("�ھ������ʣ�3/4"));
   Combe_I2CBitSet.InsertString(3,_T("�ھ������ʣ�5/6"));
   Combe_I2CBitSet.InsertString(4,_T("�ھ������ʣ�7/8"));
   Combe_I2CBitSet.SetCurSel(0);//��ʼʱ�����б�Ϊ9600

   Combe_I2C_TimeSet.InsertString(0,_T("���������1/4"));
   Combe_I2C_TimeSet.InsertString(1,_T("���������1/8"));
   Combe_I2C_TimeSet.InsertString(2,_T("���������1/16"));
   Combe_I2C_TimeSet.InsertString(3,_T("���������1/32"));
   Combe_I2C_TimeSet.SetCurSel(0);//��ʼʱ�����б�Ϊ9600

   st.pWnd = this->GetSafeHwnd();  // ��õ�ǰ���ھ��

   //--------��ʼ��ʱ����ͼ��ֱ���----------
   int Height = m_DlgPointer->m_Buffers->GetHeight();//��ȡ��ǰͼ���С  768
   int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024

   mHeight = Height;
   mWidth = Width;
   imageBits = m_DlgPointer->m_Buffers->GetPixelDepth();

   char str[10];
   int num;
   wchar_t *wide;
   sprintf(str, "%d", imageBits);//ת�����ַ���   
   num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
   wide = new wchar_t[num];
   MultiByteToWideChar(0, 0, str, -1, wide, num);
   ((CButton*)m_DlgPointer->GetDlgItem(FPGA_Bits))->SetWindowText(wide);

   UpdateData(FALSE);

   SetTimer(2, 1000, NULL);  //  ��λ��ms   //��¼֡Ƶʱ��

   try {
	   TCHAR exePath[MAX_PATH];
	   GetModuleFileName(NULL, exePath, MAX_PATH);
	   std::wstring wstrExePath(exePath);
	   std::string strExePath(wstrExePath.begin(), wstrExePath.end());

	   size_t pos = strExePath.find_last_of("\\/");
	   std::string exeDir = (pos != std::string::npos) ? strExePath.substr(0, pos) : "";

	   std::string enginePath = exeDir + "\\depth_anything_v2_vits_518x616.engine";
	   m_tensorRT = new MyTensorRT(enginePath, true);

	   //gpu Ԥ��
	   //gpu Ԥ��
	   m_tensorRT->warmup(5);
	   ////TODO�����ļ����ԣ���ʽʹ����������
	   //cv::Mat testMat = cv::imread(exeDir + "\\001_7.png");
	   //DWORD start, end;
	   //DWORD processTime;

	   //// --- ��ȹ��Ƶ��������� ---
	   //start = GetTickCount();

	   //// 1. Ԥ���� (�������������ר�ú���)
	   //m_tensorRT->preprocessImage_Depth(testMat);

	   //// 2. ִ������ (����ͨ�õ�������)
	   //m_tensorRT->inference(1);

	   //// 3. ���� (�������������ר�ú���)
	   //cv::Mat depth_result = m_tensorRT->postprocessOutput_Depth(1);

	   //end = GetTickCount();
	   //processTime = end - start;

	   //// ��ʾ���
	   //CString msg;
	   //msg.Format(_T("��ȹ������, ����ʱ�� %d ����"), processTime);
	   //MessageBox(msg, _T("������"), MB_OK | MB_ICONINFORMATION);
	   //cv::Mat final_display_img;
	   //cv::cvtColor(depth_result, final_display_img, cv::COLOR_BGR2GRAY);
	   //// �ڴ�������ʾԭʼͼ������ͼ
	   //cv::imshow("Original Image", testMat);
	   //cv::imshow("Depth Result", final_display_img);
	   //cv::waitKey(0); // �ȴ�������رմ���
	   //TODO�����ļ����ԣ���ʽʹ����������
	   //cv::Mat testMat = cv::imread(exeDir + "\\001_6.png");
	   //DWORD start, end;
	   //DWORD engineProcessTime;
	   ////Ԥ����
	   //start = GetTickCount();
	   //m_tensorRT->preprocessImage(testMat);
	   //// ִ������
	   //m_tensorRT->inference(1);
	   //// ����
	   //std::vector<Detection> detections = m_tensorRT->postprocessOutputYOLOV8(1);
	   //end = GetTickCount();
	   //engineProcessTime = end - start;
	   //CString msg;
	   //msg.Format(_T("��⵽ %d ��Ŀ��, ����ʱ�� %d ����"), detections.size(), engineProcessTime);

	   //MessageBox(msg, _T("�����"), MB_OK | MB_ICONINFORMATION);
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
	   //TODO�����ļ����ԣ���ʽʹ����������
   }
   catch (const std::exception& e) {
	   CString errorMsg;  
	   errorMsg.Format(_T("��ʼ�� TensorRT ʧ��:\n%hs"), e.what());
	   MessageBox(errorMsg, _T("����"), MB_OK | MB_ICONERROR);
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

   //---------------��ʼ������-------------------
   memset(rHot_Ram, 78, 1280 * 1024 * sizeof(unsigned short));   //0~255 ��λ��������������� ������78 ��Ӧ 20000����
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

   //---------��ʼ�����ڿ���------------
   //GetDlgItem(FPGA_HE)->SetWindowText(_T("��ǿ_��"));
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

   if( m_Xfer->Grab())  //��ʼ��������
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
	  //�ı�buffer���ã�����ͼ���С
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

//  Save����  ���浥֡ͼ��
void CGrabDemoDlg::OnFileSave() 
{
	int Height = m_DlgPointer->m_Buffers->GetHeight();//��ȡ��ǰͼ���С  768
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

	//�ֶ�ѡ�񱣴�·��
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


//  �ɼ���֡ͼ��
void CGrabDemoDlg::OnBnClickedSavemulti()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	// ��ȡ�����ɼ���֡��
	CString nFrames;
	int numSave;		//�ɼ���֡Ƶ��
	//GetDlgItem(IDC_Frame_Count)->GetWindowTextW(nFrames);
	GetDlgItemText(IDC_Frame_Count, nFrames);
	numSave = _ttoi(nFrames);
	if (numSave <= 0 || numSave >= 1000)
	{
		MessageBox(_T("The Input of Frame Counts Warning !"));
		numSave = 0;
	}
		
/*	//---------����ͼƬ�ĸ�ʽ--------
	CString fileName = m_cstrWorkPath;
	CTime time = CTime::GetCurrentTime();
	fileName += time.Format(_T("\\MultiSaved-%b-%d-%H-%M-%S.bmp"));
	char szStr[256] = {};
	wcstombs(szStr, fileName, fileName.GetLength());
	const char* pBuf = szStr;
	//m_Buffers->Save(pBuf, "-format avi",-1, numSave);
*/
	N_Saved_Frames = numSave;
	
	// demo���� ����avi�ļ�
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

//ʵ�ֶ�ʱ�� 
void CGrabDemoDlg::OnTimer(UINT_PTR nIDEvent) {
	char   str[20];
	int num;
	wchar_t *wide;
	HWND hWnd;
	switch (nIDEvent)
	{
	case 0:  //��ʱ���������
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
		sprintf(str, "%d", frame_count);//ת�����ַ���   
		num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
		wide = new wchar_t[num];
		//delete[] wide;
		MultiByteToWideChar(0, 0, str, -1, wide, num);
		((CButton*)m_DlgPointer->GetDlgItem(FPGA_frames))->SetWindowText(wide);
		//---------���㴫������----------------
		sprintf(str, "%.4f", float(frame_count)*mHeight*mWidth*imageBits/1000000);//ת�����ַ���   
		num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
		wide = new wchar_t[num];
		MultiByteToWideChar(0, 0, str, -1, wide, num);
		((CButton*)m_DlgPointer->GetDlgItem(FPGA_BitsRates))->SetWindowText(wide);

		//GetDlgItem(FPGA_frames)->SetWindowText((LPCTSTR)str);
		frame_count = 0;
		//Frame_Count = 0;
		//-------------��������ƶ���Ϣ--------
		//hWnd = AfxGetMainWnd()->m_hWnd;
		//PostMessage( WM_MOUSEMOVE, 1,NULL);
		SendMessage(WM_MOUSEMOVE, MK_SHIFT, 0x12345678);
		break;
	default:
		break;
	}
}


//	��ʱ�ɼ�ģʽ���򿪶�ʱ������ȡʱ������ʼ�ɼ�
//	��Ҫ���Ǻ�ʱ�رյ����⣺������������ģʽ�����˳�����
void CGrabDemoDlg::OnBnClickedSaveTiming()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	//MessageBox(_T("Button is on !"));
	// ��ȡ�����ɼ���֡��
	CString strTime;
	int nMS;		//�ɼ���֡Ƶ��
						//GetDlgItem(IDC_Frame_Count)->GetWindowTextW(nFrames);
	GetDlgItemText(IDC_nMS, strTime);
	nMS = _ttoi(strTime);

	SetTimer(0, nMS,NULL);   // �ڶ�������Ϊx ms
}


void CGrabDemoDlg::OnBnClickedTimingStop()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
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
//	printf("cuda�����е���cpp�ɹ���\n");
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


//--------------------���±��װ���-----------------
void CGrabDemoDlg::OnBnClickedTpCold()
{
	// �ɼ����±���ͼ����Ϣ�������� Cold_Ram
	int Height = m_DlgPointer->m_Buffers->GetHeight();//��ȡ��ǰͼ���С  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
	m_DlgPointer->m_Buffers->ReadRect(0, 0, Width, Height, Cold_Ram);
}

//--------------------���±��װ���-----------------
void CGrabDemoDlg::OnBnClickedTpHot()
{
	// �ɼ����±���ͼ����Ϣ�������� Hot_Ram
	int Height = m_DlgPointer->m_Buffers->GetHeight();//��ȡ��ǰͼ���С  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
	m_DlgPointer->m_Buffers->ReadRect(0, 0, Width, Height, Hot_Ram);

/*	//------------------����Ч��------------------
	//��ʼʱ���ȵ��GPU�������ڴ棬Ȼ����±��ײɼ���Ȼ����ʾ
	//------�����������------
	for (int i = 0; i < 50; i++)
	for (int j = 0; j < 50; j++)
	Hot_Ram[j + i * 1024] = 10000;

	//-------äԪ����-----
	for (int i = 700; i < 768; i=i+3)
	for (int j = 980; j < 1024; j=j+3)
	Hot_Ram[j + i * 1024] = 50;
*/	

	//---------------------------------------------


	//-------------------------------------------------------
	//					�������ʵ��
	//-------------------------------------------------------
	//���������������
	double Mean_Cold = 0; double Mean_Hot = 0;

	//-------------------- ������� �������׵ľ�ֵ���� -------------
	int n = 0;
	for (int i = 0; i < Height; i++)
		for (int j = 0; j < Width; j++)
		{			
			n = j + i * Width + 1; //��ʾ��n��������������
			//--------��ƽ������------
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

	//------------------���������������GPU���Թ�����------------------
	

	//���GPU����ʹ�ã�������д��GPU�ڴ�
	cudaStatus = cudaMemcpy(dev_pTP_Gain, TP_Gain, Width * Height * sizeof(float), cudaMemcpyHostToDevice);
	cudaStatus = cudaMemcpy(dev_pTP_Bias, TP_Bias, Width * Height * sizeof(float), cudaMemcpyHostToDevice);


	//-------------------------------------------------------
	//					äԪ����ʵ��
	//  ʵ�֣���Ҫ����ʱ��CPU����äԪ������GPU���м�ʱ����
	//-------------------------------------------------------
	// äԪ���� ----- �ڸ��±����У��Ҷ�ֵΪ0������ΪäԪ��Ram����Ϊ0
	Death_num = 0;
	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
			// ѡȡäԪ�Ĺ���
			if (Hot_Ram[j + i * Width] <= Cold_Ram[j + i * Width] || Hot_Ram[j + i * Width] - Cold_Ram[j + i * Width] < 0.8*(Mean_Hot - Mean_Cold) || Hot_Ram[j + i * Width] - Cold_Ram[j + i * Width] > 1.3*(Mean_Hot - Mean_Cold))
				//if (Hot_Ram[j + i * Width] < Cold_Ram[j + i * Width] || Hot_Ram[j + i * Width] - Cold_Ram[j + i * Width] < 100)
			{
				pBlind_Ram[j + i * Width] = 1;
				Death_num++;   //ͳ��äԪ����
			}
			else
			{
				pBlind_Ram[j + i * Width] = 0;
			}

	}
	//------------------��äԪ������GPU���Թ�����------------------	
	cudaStatus = cudaMemcpy(dev_pBlind_Ram, pBlind_Ram, Width * Height * sizeof(unsigned short), cudaMemcpyHostToDevice);
	//test = pBlind_Ram[700 * 1024 + 980];
	//test = pBlind_Ram[700 * 1024 + 981];
	//test = pBlind_Ram[700 * 1024 + 982];

}



//--------------ֱ��ͼ��ǿ����------------------
void CGrabDemoDlg::OnBnClickedHEnhance()
{
/*	// TODO: �ڴ���ӿؼ�֪ͨ����������
	int Histogram_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_H_Enhance))->GetCheck();

	////---------�Ǿ��Ƚ�������------
	//for (int i = 0; i < 768; i++)
	//	for (int j = 0; j < 1024; j++)
	//		Image[j + i * 1024] = 20000 * i / 768;
	//_______________________________________

	//-------�����ǿ����ʼ���½�����ǿ��
	if (Histogram_On > 0)
	{
		// ----------ͳ�Ƹ������ص�����------------
		unsigned short Histogram_Count[65536] = { 0 };   //ֱ��ͼ��ǿ��
		for (int i = 0; i < 768; i++)
			for (int j = 0; j < 1024; j++)
			{
				Histogram_Count[Image[j + i * 1024]] = Histogram_Count[Image[j + i * 1024]] + 1;
				int k = Histogram_Count[Image[j + i * 1024]];
				k = 1;
			}

		//------------
		float sum = 0;

		//-----------����ֱ��ͼ��--------------
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

//------------------CUDA��Ӧ����ON---����������CUDA---------
void CGrabDemoDlg::OnBnClickedGpu()
{

	// ON---����������CUDA    OFF---�ͷ��ڴ棬����GPU
	int GPU_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_GPU))->GetCheck();
	int Height = m_DlgPointer->m_Buffers->GetHeight();//��ȡ��ǰͼ���С  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
	cudaError_t cudaStatus;

	////---------�Ǿ��Ƚ�������------
	//for (int i = 0; i < 768; i++)
	//	for (int j = 0; j < 1024; j++)
	//		Image[j + i * 1024] = 20000 * i / 768;
	//_______________________________________

	//-------�����ǿ����ʼ���½�����ǿ��
	if (GPU_On > 0)
	{
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}

		//---------------------------ͼ���ڴ濪��---------------------------------------------------
		// ���ٴ��ͼ����ڴ�    .  
		cudaStatus = cudaMalloc((void**)&dev_img, Height * Width * sizeof(unsigned short));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}


		//----------------------------��������н������ڴ濪��---------------------------------------
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

		//--------------------------------äԪ����ʵ��--------------------------------
		cudaStatus = cudaMalloc((void**)&dev_pBlind_Ram, Height * Width * sizeof(unsigned short));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "dev_pBlind_Ram cudaMalloc failed!");
			goto Error;
		}

		//--------------------------------ֱ��ͼ������ڴ濪��--------------------------------
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

	//Ӧ��ÿ�ζ���Ҫ����ڴ��
	//cudaFree(dev_img);

}


// NETD���㣺��Ҫ�Ȳɼ����£����£���GPU������äԪ��Ȼ�����NETDֵ
// ��һ�²ɼ������µ���ֵ�����ڶ��²ɼ���������ֵ
//int NETD_Vt = 0; int NETD_Vt0 = 0; int NETD_Vn = 0;
void CGrabDemoDlg::OnBnClickedNetd()
{
	CString str;
	str.Format(_T("%d"), NETD_frames);//�̶���ʽ
	GetDlgItem(IDC_Frame_Count)->SetWindowText(str);

	// �ڲɼ����±��׵�ʱ��һ�£��ɼ�50֡�������ݡ�
	// Ȼ���ڲɼ����±��׵�ʱ��һ�£��ɼ�50֡�������ݣ������NETDֵ
	Flag_NETD++;
	if (Flag_NETD == 1)
		CGrabDemoDlg::OnBnClickedTpCold();
	else
		CGrabDemoDlg::OnBnClickedTpHot();
	current = 0;  //��ʼ���ɼ���֡��

	//--------����ͼ��-----------
	OnBnClickedSavemulti();
}

//����������Ӧ��ѹ��������ѹ

void CGrabDemoDlg::CalculateNETD(int & flag,int Height, int Width)
{
	//��һ���¶�ֵ����
	if (flag == 1 && current < NETD_frames)
	{
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				//���±��ף���Ӧ��ѹƽ��ֵ����
				double tmp = (double(Image[j + i * Width]) - double(NETD_Vt[j + i*Width])) / (current + 1) + double(NETD_Vt[j + i*Width]);
				NETD_Vt[j + i*Width] = tmp;
			}
		}
		//-------------��һ�βɼ���ϣ������������ʾ--------------
		if (current == NETD_frames - 1)
			GetDlgItem(IDC_NETDshow)->SetWindowText(_T("Ready"));
	}
	// �ڶ����¶�ֵ����
	else if (flag == 2 && current < NETD_frames)
	{
		for (int i = 0; i < Height; i++)
		{
			for (int j = 0; j < Width; j++)
			{
				//���±��ף���Ӧ��ѹƽ��ֵ����
				double tmp = (double(Image[j + i * Width]) - double(NETD_Vt0[j + i*Width])) / (current + 1) + double(NETD_Vt0[j + i*Width]);
				NETD_Vt0[j + i*Width] = tmp;
				//�洢50֡���µ�ѹ���ݣ����ڼ���������ѹ
				NETD_Vn[j + i*Width][current] = Image[j + i * Width];
				//����50֡������ѹƽ��ֵ
				//NETD_VnA[j + i*Width] = (Image[j + i * Width] - NETD_VnA[j + i*Width]) / (current + 1) + NETD_VnA[j + i*Width];
			}

		}
		//�ɼ���Ϻ����NETDֵ
		if (current == NETD_frames - 1)
		{
			// �������Ԫ��������ѹ
			for (int i = 0; i < Height; i++)
			{
				for (int j = 0; j < Width; j++)
				{
					//��������ƽ��ֵ
					double sum = 0;
					for (int k = 0; k < NETD_frames; k++)
					{
						sum = sum + pow(double(NETD_Vn[j + i*Width][k]) - double(NETD_Vt0[j + i*Width]), 2);  //���㷽��õ�������ѹ
					}
					NETD_VnA[j + i*Width] = (1 / double(NETD_K)) * sqrt((sum / (NETD_frames - 1)));  //��������ص�������ѹ
				}
			}
			//Ϊ�ų�������Ԫ������������ѹ��ֵ��>2��ƽ��������ѹ��Ϊ������Ԫ����äԪ��ֵ��Ϊ2��CPU��ֻ����äԪ���滻���ж�����==1
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
					//���������Ԫ
					if (NETD_VnA[j + i*Width] > 2 * Average_Noise && pBlind_Ram[j + i*Width] == 0)
					{
						pBlind_Ram[j + i*Width] = 2;
						Hot_num++;
					}

				}
			}

			//���㵥����Ԫ��NETD
			for (int i = 0; i < Height; i++)
			{
				for (int j = 0; j < Width; j++)
				{
					if (pBlind_Ram[j + i*Width] == 0)
					{
						NETD_Vt[j + i*Width] = (1 / double(NETD_K)) * double(NETD_Vt0[j + i*Width] - NETD_Vt[j + i*Width]);  //��������Ӧ��ѹ
						NETD_Vt0[j + i*Width] = double(T - T0) / double(NETD_Vt[j + i*Width]) * double(NETD_VnA[j + i*Width]);  // ����NETD �������
					}

				}
			}
			//----------NETD������ɣ���ƽ����������ؼ�---------
			int count = 0;	double Aver = 0;
			for (int i = 0; i < Height; i++)
			{
				for (int j = 0; j < Width; j++)
				{
					if (pBlind_Ram[j + i*Width] == 0)
					{
						Aver = (double(NETD_Vt0[j + i*Width]) - Aver) / (count + 1) + Aver;  //����ƽ��NETD
						count++;
					}
				}
			}
			//NETDshow->SetWindowText(str);
			//((CButton*)m_DlgPointer->GetDlgItem(IDC_NETDshow));
			char   str[20];
			sprintf(str, "%.4f", 1000*Aver);//ת�����ַ���   
			// double   x=   atof(jidian)   ;//ת���ɸ�����
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

	current++;//��ʾ֡������
}

void CGrabDemoDlg::OnBnClickedOk()
{
	// --------------------����I2C����-------------------
	IMyCalc *pMyCalc = NULL;
	IUnknown *pUnknown = NULL;
	// 1.��ʼ��COM��
	HRESULT hr = CoInitialize(NULL);
	// 2.������֪ProgID�Ҷ�ӦCLSID
	CLSID CalcID;
	hr = ::CLSIDFromProgID(L"MyCalc.math", &CalcID);   // �˴��ǿ�ʼ���õ�ID
	if (S_OK != hr)
		AfxMessageBox(_T("����CalcIDʧ��"));
	// 3.������Ӧ�Ľӿ�ʵ��
	hr = CoCreateInstance(CalcID, NULL, CLSCTX_LOCAL_SERVER, IID_IUnknown, (void **)&pUnknown);
	if (S_OK != hr)
		AfxMessageBox(_T("����ʵ��ʧ��"));
	// 4.��ѯ�ӿ�
	hr = pUnknown->QueryInterface(IID_IMyCalc, (void **)&pMyCalc);
	if (S_OK != hr)
		AfxMessageBox(_T("����IMyCalcʧ��"));
	// 5.�����ڲ��ӿ�


	// TODO: �ڴ���ӿؼ�֪ͨ����������
	CString tmpString;
	char n_in = 2; char n_out = 3;  //������ٶ�ȡ���٣�

	GetDlgItemText(I2C_nW, tmpString);
	n_in = _ttoi(tmpString);
	GetDlgItemText(I2C_nR, tmpString);
	n_out = _ttoi(tmpString);


	char InData[4];
	char OutData[3];
	// ��һ�ֶ�Ĭ���Ѿ������� 0xA0��ֱ�Ӵӵڶ��ֶο�ʼд
	//InData[-1] = 0xA0;
	//-----------------��ȡ��������--------------------
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
	
	//-------------------����I2C 32λ����ӿ�------------------------
	pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);

	//--------------------���������----------------
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

	//������Դ������ʼ��COM���
	pMyCalc->Release();
	pUnknown->Release();
	CoUninitialize();

}

// ------------   ���ʼ��� ----------------
void CGrabDemoDlg::OnBnClickedBitrate()
{
	// --------------------����I2C����-------------------
	IMyCalc *pMyCalc = NULL;
	IUnknown *pUnknown = NULL;
	// 1.��ʼ��COM��
	HRESULT hr = CoInitialize(NULL);
	// 2.������֪ProgID�Ҷ�ӦCLSID
	CLSID CalcID;
	hr = ::CLSIDFromProgID(L"MyCalc.math", &CalcID);   // �˴��ǿ�ʼ���õ�ID
	if (S_OK != hr)
		AfxMessageBox(_T("����CalcIDʧ��"));
	// 3.������Ӧ�Ľӿ�ʵ��
	hr = CoCreateInstance(CalcID, NULL, CLSCTX_LOCAL_SERVER, IID_IUnknown, (void **)&pUnknown);
	if (S_OK != hr)
		AfxMessageBox(_T("����ʵ��ʧ��"));
	// 4.��ѯ�ӿ�
	hr = pUnknown->QueryInterface(IID_IMyCalc, (void **)&pMyCalc);
	if (S_OK != hr)
		AfxMessageBox(_T("����IMyCalcʧ��"));
	// 5.�����ڲ��ӿ�

	//--------------------------���ʼ���---------------------------
	//  ���ʣ�Ru = Rs * b * CRv * CRrs * (Tu/Ts)
	double Rs = double(6048) / 896; double B = 2; double CRv = double(1 / 2); double CRrs = double(188) / 204; double TuTs = double(4) / 5;
	char InData[4];
	char OutData[3];
	char n_in = 2; char n_out = 3;  //������ٶ�ȡ���٣�

	//-----��ȡ Rs ------ ��ȡ���� 0 : 2K---->1512/224uS  _________  1 : 8K  ----> 6048/896uS
	InData[0] = 0x1D;
	n_in = 2; n_out = 1;
	pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
	if (OutData[0] == 0)
		Rs = double(1512) / 224;
	else if (OutData[0] == 1)
		Rs = double(6048) / 896;
	else 
		MessageBox(_T("I2C Read Data Error !"));

	//-----��ȡ b ------
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

	//-----��ȡ CRv ------
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

	//-----��ȡ Ts ------
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

	//������Դ������ʼ��COM���
	pMyCalc->Release();
	pUnknown->Release();
	CoUninitialize();
}

//------------------------����λ��ͨ��ģ��-----------------------
void CGrabDemoDlg::OnBnClickedOpen()
{
	// ��ȡ����ѡ��������ѡ��
	int nIndex = m_Combo.GetCurSel();
	CString strCom; CString strRate;
	m_Combo.GetLBText(nIndex, strCom);

	// ��ȡ���ڴ����ٶ�ѡ��������ѡ��
	nIndex = Comb_Rate.GetCurSel();
	Comb_Rate.GetLBText(nIndex, strRate);
	int Rate = _ttoi(strRate);

	//strtmp.Format(_T("COM:%d\n"), com_data);// ��Ϣ��ʾ����
	//MessageBox(strtmp);
	m_ComPortFlag = st.ThreadInit(strCom,Rate);
	if (m_ComPortFlag == true)
	{
		::MessageBox(NULL, _T("���ڴ򿪳ɹ�"), _T("��ʾ"), MB_OK);
	}
	else
	{
		::MessageBox(NULL, _T("���ڴ�ʧ��"), _T("��ʾ"), MB_OK);
	}

	st.Com.SetWnd(this->GetSafeHwnd());   //������Ϣ������ǰ����
}

// ��λ�����ڽ��յ����ݵ��жϺ���
LRESULT CGrabDemoDlg::SerialRead(WPARAM, LPARAM)
{

	//AfxMessageBox(_T("��������"));
	SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
	int result = st.OnReceive();

	if (result == 0)
		FPGA_Receive();
	else
	{
		CString temp_value = _T("");   //temp_value��������intֵ
		temp_value.Format(_T("���ݴ��󣬴��룺%d"), result);//�̶���ʽ
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
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	st.CloseSerialPort();
	//::MessageBox(NULL, _T("�����ѹر�"), _T("��ʾ"), MB_OK);
	m_ComPortFlag = false;
}


void CGrabDemoDlg::OnBnClickedHe()
{
	FPGA_Send();
	if (m_ComPortFlag == true)
	{
		// ��ʱ��1��ʾ����λ��ͨ��
		SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
		bool status = 0;
		CString HE;
		GetDlgItemText(FPGA_HE, HE); //ȡ��ť����
		if (HE == _T("��ǿ_��"))
		{
			status = st.SendCommand(COMMAND_PE_MODULE_CONTROL, 0xff);
			if (status)
				GetDlgItem(FPGA_HE)->SetWindowText(_T("��ǿ_�ر�"));
			return;
		}
		else
		{
			status = st.SendCommand(COMMAND_PE_MODULE_CONTROL, 0xf0);
			if (status)
				GetDlgItem(FPGA_HE)->SetWindowText(_T("��ǿ_��"));
			return;
		}
	}
	::MessageBox(NULL, _T("����δ�򿪣��޷�����"), _T("��ʾ"), MB_OK);
}

//��λ�� �ɼ����±���
void CGrabDemoDlg::OnBnClickedGetlow()
{
	FPGA_Send();
	SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
	if (m_ComPortFlag == true)
	{
		bool status = st.SendCommand(COMMAND_NUC_CONTROL, 0x00);
		if (!status)		
			::MessageBox(NULL, _T("��������ʧ��"), _T("��ʾ"), MB_OK);
		return;
	}
	::MessageBox(NULL, _T("����δ�򿪣��޷�����"), _T("��ʾ"), MB_OK);
}
void CGrabDemoDlg::OnBnClickedGethigh()
{
	FPGA_Send();
	SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
	if (m_ComPortFlag == true)
	{
		bool status = st.SendCommand(COMMAND_NUC_CONTROL, 0x01);
		if (!status)
			::MessageBox(NULL, _T("��������ʧ��"), _T("��ʾ"), MB_OK);
		return;
	}
	::MessageBox(NULL, _T("����δ�򿪣��޷�����"), _T("��ʾ"), MB_OK);
}

void CGrabDemoDlg::OnBnClickedBlindCorrection()
{
	FPGA_Send();
	// ��ʱ��1��ʾ����λ��ͨ��
	SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
	if (m_ComPortFlag == true)
	{
		bool status = 0;
		CString HE;
		GetDlgItemText(FPGA_Blind_Correction, HE); //ȡ��ť����
		if (HE == _T("äԪ����_��"))
		{
			status = st.SendCommand(COMMAND_DY_BADPOINT_MODULE_CONTROL, 0xff);
			if (status)
				GetDlgItem(FPGA_Blind_Correction)->SetWindowText(_T("äԪ����_�ر�"));
			return;
		}
		else
		{
			status = st.SendCommand(COMMAND_DY_BADPOINT_MODULE_CONTROL, 0xf0);
			if (status)
				GetDlgItem(FPGA_Blind_Correction)->SetWindowText(_T("äԪ����_��"));
			return;
		}
	}
	::MessageBox(NULL, _T("����δ�򿪣��޷�����"), _T("��ʾ"), MB_OK);
}
//��λ�� �������
void CGrabDemoDlg::OnBnClickedTpCorrection()
{
	FPGA_Send();
	// ��ʱ��1��ʾ����λ��ͨ��
	SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
	if (m_ComPortFlag == true)
	{
		bool status = 0;
		CString HE;
		GetDlgItemText(FPGA_TP_Correction, HE); //ȡ��ť����
		if (HE == _T("�������_��"))
		{
			status = st.SendCommand(COMMAND_TWOPOINT_MODULE_CONTROL, 0xff);
			if (status)
				GetDlgItem(FPGA_TP_Correction)->SetWindowText(_T("�������_�ر�"));
			return;
		}
		else
		{
			status = st.SendCommand(COMMAND_TWOPOINT_MODULE_CONTROL, 0xf0);
			if (status)
				GetDlgItem(FPGA_TP_Correction)->SetWindowText(_T("�������_��"));
			return;
		}
	}
	::MessageBox(NULL, _T("����δ�򿪣��޷�����"), _T("��ʾ"), MB_OK);
}


void CGrabDemoDlg::OnBnClickedMedianFilter()
{
	FPGA_Send();
	// ��ʱ��1��ʾ����λ��ͨ��
	SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
	if (m_ComPortFlag == true)
	{
		bool status = 0;
		CString HE;
		GetDlgItemText(FPGA_Median_Filter, HE); //ȡ��ť����
		if (HE == _T("��ֵ�˲�_��"))
		{
			status = st.SendCommand(COMMAND_MID_MODULE_CONTROL, 0xff);
			if (status)
				GetDlgItem(FPGA_Median_Filter)->SetWindowText(_T("��ֵ�˲�_�ر�"));
			return;
		}
		else
		{
			status = st.SendCommand(COMMAND_MID_MODULE_CONTROL, 0xf0);
			if (status)
				GetDlgItem(FPGA_Median_Filter)->SetWindowText(_T("��ֵ�˲�_��"));
			return;
		}
	}
	::MessageBox(NULL, _T("����δ�򿪣��޷�����"), _T("��ʾ"), MB_OK);

}


void CGrabDemoDlg::OnBnClickedtest()
{
	// NETD���ԣ����������ͼ��

	//�����ֵ

	Flag_NETD = 1;
	for (int i = 0; i < 768; i++)
		for (int j = 0; j < 1024; j++)
			Image[j + i * 1024] = 80;
	m_DlgPointer->CalculateNETD(Flag_NETD, 768, 1024);

	

	//����ͼ��ɼ�
	//�����ֵ
	current = 0;
	Flag_NETD = 2;
	for (int k = 0; k < 50; k++)
	{
		//����ͼ��ɼ�
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
	// ��ʱ��1��ʾ����λ��ͨ��
	SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
	for (int i = 0; i < 50000; i++)
		parameters[i] = i;
	bool status = st.SendCommand(COMMAND_UPDATE_BADPOINT_TABLE, parameters,50000);

}


void CGrabDemoDlg::OnBnClickedBpmap()
{
	FPGA_Send();
	// ����λ��������λ�������äԪ��
	SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
	int Height = m_DlgPointer->m_Buffers->GetHeight();//��ȡ��ǰͼ���С  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
	long k = 0;

	// ����ʹ�� �� ���ڸ�äԪ���Զ��帳ֵ
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

	//����äԪ��ת��Ϊ��λ���ܽ��ܵ����ݸ�ʽ
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
	// ����λ��������λ�����������У����
	SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
	int Height = m_DlgPointer->m_Buffers->GetHeight();//��ȡ��ǰͼ���С  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024
	
	//���ڴ洢��λ���ɽ��յ���������
	//unsigned char parameters[1024 * 768];
	long k = 0;
	unsigned char* ptmp;
	//TP_Gain[0] = 0xa00b;
	//ptmp = (unsigned char*)&TP_Gain[0];

	//���Դ��룺
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
	//���ݽ�����ת��Ϊ��λ���ܽ��ܵ����ݸ�ʽ unsigned char 
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
	//-----���������ɫ------
	FPGA_Send();

	// ��ȡ�����ɼ���֡��
	CString nFrames;
	int numIntergral;		//�ɼ���֡Ƶ��s
						//GetDlgItem(IDC_Frame_Count)->GetWindowTextW(nFrames);
	GetDlgItemText(FPGA_InterNum, nFrames);
	numIntergral = _ttoi(nFrames);
	if (numIntergral >= 100)
	{
		MessageBox(_T("The Input of Intergral Time Warning !"));
		return;
	}

	//----------��������--------------
	if (m_ComPortFlag == true)
	{
		// ��ʱ��1��ʾ����λ��ͨ��
		SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
		bool status = 0;
		status = st.SendCommand(COMMAND_INTEGRATION_TIME_CONTROL, unsigned char(numIntergral));
		if (status)
			GetDlgItem(FPGA_HE)->SetWindowText(_T("��ǿ_�ر�"));
		return;
	}
	::MessageBox(NULL, _T("����δ�򿪣��޷�����"), _T("��ʾ"), MB_OK);
}




void CGrabDemoDlg::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: �ڴ������Ϣ�����������/�����Ĭ��ֵ
	//int x=0, y=0;
	//POINT PP = m_DlgPointer->m_View->GetScrollPos();
	//m_DlgPointer->m_View->OnHScroll(x);
	//m_DlgPointer->m_View->OnVScroll(y);

	//CRect  rect;
	//GetClientRect(&rect);//��ȡ�ͻ����Ĵ�С
	////CPoint point;
	//GetCursorPos(&point);//��ȡ��ǰָ������꣨ע�⣬������Ļ�ģ�
	//GetWindowRect(&rect);//��ȡ�ͻ������ͻ��������Ͻǣ��������Ļ��λ��
	//int x = (point.x - rect.left);//ͨ���任�ĵ��ͻ���������  
	//int y = (point.y - rect.top);
/*	SIZE Rsize = m_DlgPointer->m_View->GetScrollRange();

	CString str;
	point.x = point.x - Rsize.cx;
	point.y = point.y - Rsize.cy;
	str = m_ImageWnd.GetPixelString(point);
	SetDlgItemText(PointValue, str);

	//str.Format(_T("x=%d,y=%d"), Rsize.cx, Rsize.cy);
	////str.Format("��괦��x=%d,y=%d��λ��", point.x, point.y);
	//SetDlgItemText(PointValue, str);
*/
	__super::OnMouseMove(nFlags, point);
}


void CGrabDemoDlg::OnBnClickedChange1()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	// --------------------����I2C����-------------------
	IMyCalc *pMyCalc = NULL;
	IUnknown *pUnknown = NULL;
	// 1.��ʼ��COM��
	HRESULT hr = CoInitialize(NULL);
	// 2.������֪ProgID�Ҷ�ӦCLSID
	CLSID CalcID;
	hr = ::CLSIDFromProgID(L"MyCalc.math", &CalcID);   // �˴��ǿ�ʼ���õ�ID
	if (S_OK != hr)
		AfxMessageBox(_T("����CalcIDʧ��"));
	// 3.������Ӧ�Ľӿ�ʵ��
	hr = CoCreateInstance(CalcID, NULL, CLSCTX_LOCAL_SERVER, IID_IUnknown, (void **)&pUnknown);
	if (S_OK != hr)
		AfxMessageBox(_T("����ʵ��ʧ��"));
	// 4.��ѯ�ӿ�
	hr = pUnknown->QueryInterface(IID_IMyCalc, (void **)&pMyCalc);
	if (S_OK != hr)
		AfxMessageBox(_T("����IMyCalcʧ��"));
	// 5.�����ڲ��ӿ�

	//--------------------------���ʼ���---------------------------
	//  ���ʣ�Ru = Rs * b * CRv * CRrs * (Tu/Ts)
	double Rs = double(6048) / 896; double B = 2; double CRv = double(1 / 2); double CRrs = double(188) / 204; double TuTs = double(4) / 5;
	char InData[4];
	char OutData[3];
	char n_in = 2; char n_out = 3;  //������ٶ�ȡ���٣�

	int nIndex = Combe_I2CMode.GetCurSel();
	CString strCom; 
	Combe_I2CMode.GetLBText(nIndex, strCom);

	switch (nIndex)
	{
	case 0:
	{
		//-----д Rs ------ 2Kģʽ
		InData[0] = 0x1D; InData[1] = 0x00;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 1:
	{
		//-----д Rs ------ 8Kģʽ
		InData[0] = 0x1D; InData[1] = 0x01;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 2:
	{
		//-----д b ------ QPSK
		InData[0] = 0x1E; InData[1] = 0x00;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 3:
	{
		//-----д b ------ 16-QAM
		InData[0] = 0x1E; InData[1] = 0x01;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 4:
	{
		//-----д b ------ 64-QAM
		InData[0] = 0x1E; InData[1] = 0x02;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	default:
		AfxMessageBox(_T("δƥ�䵽ѡ��"));
		break;
	}

	//������Դ������ʼ��COM���
	pMyCalc->Release();
	pUnknown->Release();
	CoUninitialize();
}


void CGrabDemoDlg::OnBnClickedChange2()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	// --------------------����I2C����-------------------
	IMyCalc *pMyCalc = NULL;
	IUnknown *pUnknown = NULL;
	// 1.��ʼ��COM��
	HRESULT hr = CoInitialize(NULL);
	// 2.������֪ProgID�Ҷ�ӦCLSID
	CLSID CalcID;
	hr = ::CLSIDFromProgID(L"MyCalc.math", &CalcID);   // �˴��ǿ�ʼ���õ�ID
	if (S_OK != hr)
		AfxMessageBox(_T("����CalcIDʧ��"));
	// 3.������Ӧ�Ľӿ�ʵ��
	hr = CoCreateInstance(CalcID, NULL, CLSCTX_LOCAL_SERVER, IID_IUnknown, (void **)&pUnknown);
	if (S_OK != hr)
		AfxMessageBox(_T("����ʵ��ʧ��"));
	// 4.��ѯ�ӿ�
	hr = pUnknown->QueryInterface(IID_IMyCalc, (void **)&pMyCalc);
	if (S_OK != hr)
		AfxMessageBox(_T("����IMyCalcʧ��"));
	// 5.�����ڲ��ӿ�

	//--------------------------���ʼ���---------------------------
	//  ���ʣ�Ru = Rs * b * CRv * CRrs * (Tu/Ts)
	double Rs = double(6048) / 896; double B = 2; double CRv = double(1 / 2); double CRrs = double(188) / 204; double TuTs = double(4) / 5;
	char InData[4];
	char OutData[3];
	char n_in = 3; char n_out = 0;  //д3��0

	int nIndex = Combe_I2CBitSet.GetCurSel();
	CString strCom;
	Combe_I2CBitSet.GetLBText(nIndex, strCom);

	switch (nIndex)
	{
	case 0:
	{
		//-----д �ھ������������ CRv ------ 1/2
		InData[0] = 0x1F; InData[1] = 0x00;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 1:
	{
		//-----д �ھ������������ CRv ------ 2/3
		InData[0] = 0x1F; InData[1] = 0x01;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 2:
	{
		//-----д �ھ������������ CRv ------ 3/4
		InData[0] = 0x1F; InData[1] = 0x02;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 3:
	{
		//-----д �ھ������������ CRv ------ 5/6
		InData[0] = 0x1F; InData[1] = 0x03;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 4:
	{
		//-----д �ھ������������ CRv ------ 7/8
		InData[0] = 0x1F; InData[1] = 0x04;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	default:
		AfxMessageBox(_T("δƥ�䵽ѡ��"));
		break;
	}

	//������Դ������ʼ��COM���
	pMyCalc->Release();
	pUnknown->Release();
	CoUninitialize();
}


void CGrabDemoDlg::OnBnClickedChange3()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	// --------------------����I2C����-------------------
	IMyCalc *pMyCalc = NULL;
	IUnknown *pUnknown = NULL;
	// 1.��ʼ��COM��
	HRESULT hr = CoInitialize(NULL);
	// 2.������֪ProgID�Ҷ�ӦCLSID
	CLSID CalcID;
	hr = ::CLSIDFromProgID(L"MyCalc.math", &CalcID);   // �˴��ǿ�ʼ���õ�ID
	if (S_OK != hr)
		AfxMessageBox(_T("����CalcIDʧ��"));
	// 3.������Ӧ�Ľӿ�ʵ��
	hr = CoCreateInstance(CalcID, NULL, CLSCTX_LOCAL_SERVER, IID_IUnknown, (void **)&pUnknown);
	if (S_OK != hr)
		AfxMessageBox(_T("����ʵ��ʧ��"));
	// 4.��ѯ�ӿ�
	hr = pUnknown->QueryInterface(IID_IMyCalc, (void **)&pMyCalc);
	if (S_OK != hr)
		AfxMessageBox(_T("����IMyCalcʧ��"));
	// 5.�����ڲ��ӿ�

	//--------------------------���ʼ���---------------------------
	//  ���ʣ�Ru = Rs * b * CRv * CRrs * (Tu/Ts)
	char InData[4];
	char OutData[3];
	char n_in = 2; char n_out = 3;  //������ٶ�ȡ���٣�

	int nIndex = Combe_I2C_TimeSet.GetCurSel();
	CString strCom;
	Combe_I2C_TimeSet.GetLBText(nIndex, strCom);

	switch (nIndex)
	{
	case 0:
	{
		// �������1/4, 1/8, 1/16, 1/32
		InData[0] = 0x20; InData[1] = 0x00;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 1:
	{
		//�������1/8
		InData[0] = 0x20; InData[1] = 0x01;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 2:
	{
		//�������1/16
		InData[0] = 0x20; InData[1] = 0x02;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	case 3:
	{
		//-----�������1/32
		InData[0] = 0x20; InData[1] = 0x03;
		n_in = 3; n_out = 0;
		pMyCalc->COM32(InData[0], InData[1], InData[2], InData[3], n_in - 1, n_out, &OutData[0], &OutData[1], &OutData[2]);
		break;
	}
	default:
		AfxMessageBox(_T("δƥ�䵽ѡ��"));
		break;
	}

	//������Դ������ʼ��COM���
	pMyCalc->Release();
	pUnknown->Release();
	CoUninitialize();
}

// ������λ����Ϣ�������������������������ɫ
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
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	FPGA_Send();
	SetTimer(1, 50000, NULL);  //  ��λ��ms
	if (m_ComPortFlag == true)
	{
		bool status = st.SendCommand(COMMAND_NUC_CONTROL, 0x02);
		if (!status)
			::MessageBox(NULL, _T("��������ʧ��"), _T("��ʾ"), MB_OK);
		return;
	}
	::MessageBox(NULL, _T("����δ�򿪣��޷�����"), _T("��ʾ"), MB_OK);
}


void CGrabDemoDlg::OnBnClickedUpdate()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	CGrabDemoDlg::FPGA_Receive();
}

// �������CUDA�㷨�Ľӿ�
void CGrabDemoDlg::CUDA_Algorithm()
{
	//�ж��Ƿ��������㷨��û���򷵻�
	//--------�㷨����-------
	int Bilateral_On = ((CButton*)m_DlgPointer->GetDlgItem(PC_bilateralFilter))->GetCheck();
	int Enhance_On = ((CButton*)m_DlgPointer->GetDlgItem(IDC_H_Enhance))->GetCheck();

	if (!Bilateral_On && !Enhance_On)
		return;
	//------------ʹ��opencv+CUDA��д��---------
	int Height = m_DlgPointer->m_Buffers->GetHeight();//��ȡ��ǰͼ���С  768
	int Width = m_DlgPointer->m_Buffers->GetWidth();   //1024


	// ��16 bitsͼ��תΪ 8bits���� opencv + cuda ����
	for (int i = 0; i < Height; i++)
		for (int j = 0; j < Width; j++)
			rImage_uchar[i * Width + j] = rImage[i * Width + j] >> 8;
	// srcΪԭͼ��grayΪ�����ͼ��histΪֱ��ͼ
	Mat src_host = Mat(Height, Width, CV_8UC1, rImage_uchar);
	Mat gray_host;
	Mat hist_host;
	GpuMat src, gray, hist;

	// ˫���˲��㷨ʵ�֣�
	if (Bilateral_On > 0)
	{		
		src.upload(src_host);
		cv::cuda::bilateralFilter(src, gray, 8, 15, 15);
		gray.download(gray_host);
	}
	//ֱ��ͼ��ǿ�㷨ʵ��
	if (Enhance_On)
	{
		//û����˫���˲���ֱ�ӵ���ԭͼ || ����Ѿ�ִ��˫���˲�����ֱ���ϴ�ԭͼ��gray
		if (Bilateral_On <= 0)
		{
			gray.upload(src_host);
			gray.download(gray_host);
		}		
		cv::cuda::calcHist(gray, hist);
		hist.download(hist_host);
		//-------˫ƽ̨���� ���0-255�Ҷȼ� -------
		int L = 0; //���ڴ洢ֱ��ͼ�ķ�������Ŀ
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
		//------- ��˫ƽ̨���޽��� ����ֱ��ͼ ------------

		Sum = 0; //����ͳ�����������ں����������ֱ��ͼ
		for (int i = 0; i < 256; i++)
		{
			if (tmp_hist[i] < Tdown)
				tmp_hist[i] = Tdown;
			else if (tmp_hist[i] > Tup)
				tmp_hist[i] = Tup;
			Sum += tmp_hist[i];
		}
		// ���Ƹ����ܶ�ͼ
		for (int i = 1; i < 256; i++)
		{
			tmp_hist[i] = tmp_hist[i] + tmp_hist[i - 1];
		}
		//-------- ����ͼ�� ----------
		unsigned char* inData = src_host.ptr<unsigned char>(0); //ԭͼ
		unsigned char* outData = gray_host.ptr<unsigned char>(0); //�����ͼ��
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


	//------�����ݸ�ʽ 8 bits ��Ϊ 16 bits ------------
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
	// ������λ�����п�������
	//-----���������ɫ------
	FPGA_Send();

	// ��ȡ�����ɼ���֡��
	CString nFrames;
	int Win_H, Win_W;		//�ɼ���֡Ƶ��s
							//GetDlgItem(IDC_Frame_Count)->GetWindowTextW(nFrames);
	GetDlgItemText(FPGA_WinHeight, nFrames);
	Win_H = _ttoi(nFrames);
	GetDlgItemText(FPGA_WinWidth, nFrames);
	Win_W = _ttoi(nFrames);

	// ���ڴ�С���ƣ�0 <= H��Y���꣩<=511  ||   128 <= W (X����) <= 639
	if (Win_H < 0 || Win_H>511 || Win_W < 128 || Win_W > 639)
	{
		MessageBox(_T("The Window Size is Out of Range !"));
		return;
	}
	int index = 0;

	//------�����������ݣ�data[0],[1]ΪYmin��[2],[3]ΪYmax��[4]ΪXmin��[5]ΪXmax ���ǰ�λ���ݴ���
	transfer_data[index++] = 0x00;
	transfer_data[index++] = 0x00;
	unsigned short tmpValue = unsigned short(Win_H);
	transfer_data[index++] = tmpValue;
	transfer_data[index++] = tmpValue >> 8;
	transfer_data[index++] = 0x00;
	transfer_data[index++] = unsigned short(Win_W / 4); 

	//----------��������--------------
	if (m_ComPortFlag == true)
	{
		// ��ʱ��1��ʾ����λ��ͨ��
		SetTimer(1, Serial_Delay_Time, NULL);  //  ��λ��ms
		bool status = 0;
		status = st.SendCommand(COMMAND_WINDOW_MODE_SWITCH, transfer_data, 6);
		if (status)
			GetDlgItem(FPGA_HE)->SetWindowText(_T("��ǿ_�ر�"));
		return;
	}
	::MessageBox(NULL, _T("����δ�򿪣��޷�����"), _T("��ʾ"), MB_OK);
}


void CGrabDemoDlg::OnBnClickedLocalEnlarge()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void CGrabDemoDlg::OnStnClickedViewWnd2()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void CGrabDemoDlg::OnBnClickedCheck1()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void CGrabDemoDlg::OnBnClickedDepthCheck()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}
