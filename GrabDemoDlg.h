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

class CGrabDemoDlg : public CDialog, public CImageExWndEventHandler
{
// Construction
public:
	DistortionCailbration distortionCailbration;
	CGrabDemoDlg(CWnd* pParent = NULL);	// standard constructor

	BOOL CreateObjects();
	BOOL DestroyObjects();
	void UpdateMenu();
	static void XferCallback(SapXferCallbackInfo *pInfo);
	static void SignalCallback(SapAcqCallbackInfo *pInfo);
   void GetSignalStatus();
   void GetSignalStatus(SapAcquisition::SignalStatus signalStatus);
   void PixelChanged(int x, int y);
   void FPGA_Send();
   void FPGA_Receive();
// Dialog Data
	//{{AFX_DATA(CGrabDemoDlg)
	enum { IDD = IDD_GRABDEMO_DIALOG };
	float       m_BufferFrameRate;
	CStatic	m_statusWnd;
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGrabDemoDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON		m_hIcon;
	CString  m_appTitle;

	CString m_cstrWorkPath;    // 工作路径，存放采集数据和计算中间数据
	int m_imageHeight;             // 图像高度
	int m_imageWidth;              // 图像宽度

	CImageExWnd		m_ImageWnd;
	SapAcquisition	*m_Acq;
	SapBuffer		*m_Buffers;
	SapTransfer		*m_Xfer;
	SapView        *m_View;
	//SapView        *m_ViewProcessed;

	static CGrabDemoDlg   *m_DlgPointer;  //回调函数访问非静态成员
	static MyTensorRT* m_super_tensorRT;
	static MyTensorRT* m_detect_tensorRT;
	static MyTensorRT* m_depth_tensorRT;
   BOOL m_IsSignalDetected;   // TRUE if camera signal is detected

	// Generated message map functions
	//{{AFX_MSG(CGrabDemoDlg)
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnDestroy();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnSnap();
	afx_msg void OnGrab();
	afx_msg void OnFreeze();
	afx_msg void OnGeneralOptions();
	afx_msg void OnAreaScanOptions();
	afx_msg void OnLineScanOptions();
	afx_msg void OnCompositeOptions();
	afx_msg void OnLoadAcqConfig();
   afx_msg void OnImageFilterOptions();
	afx_msg void OnBufferOptions();
	afx_msg void OnViewOptions();
	afx_msg void OnFileLoad();
	afx_msg void OnFileNew();
	afx_msg void OnFileSave();
	afx_msg void OnExit();
   afx_msg void OnEndSession(BOOL bEnding);
   afx_msg BOOL OnQueryEndSession();
   afx_msg void OnKillfocusBufferFrameRate(void);

   afx_msg void OnTimer(UINT_PTR nIDEvent);

	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()

public:
	afx_msg void OnBnClickedSavemulti();

	afx_msg void OnBnClickedSaveTiming();
	afx_msg void OnBnClickedTimingStop();
	CButton GPU_State;
	CButton TP_Correction;
	afx_msg void OnBnClickedTpCold();
	afx_msg void OnBnClickedTpHot();
	CButton Blind_Correction;
	CButton H_Enhance;
	afx_msg void OnBnClickedHEnhance();
	afx_msg void OnBnClickedGpu();

	afx_msg void OnBnClickedGetlow();
	afx_msg void OnBnClickedNetd();

	void CalculateNETD(int & flag,int Height, int Width);

	afx_msg void OnBnClickedOk();
	CComboBox m_Combo;
	afx_msg void OnBnClickedOpen();
	afx_msg void OnBnClickedClose();
	afx_msg void OnBnClickedHe();
	// 下位机串口接收到数据的中断函数
	afx_msg void localEnlarge(int Height, int Widt);
	LRESULT CGrabDemoDlg::SerialRead(WPARAM, LPARAM);
	afx_msg void OnBnClickedGethigh();
	afx_msg void OnBnClickedBlindCorrection();
	afx_msg void OnBnClickedTpCorrection();
	afx_msg void OnBnClickedMedianFilter();
	afx_msg void OnBnClickedtest();
	afx_msg void OnBnClickedTest();
	afx_msg void OnBnClickedBpmap();
	afx_msg void OnBnClickedTpmap();
	afx_msg void OnBnClickedIntegral();
	afx_msg void OnBnClickedBitrate();
	int mHeight;
	int mWidth;
	int imageBits;

	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	CComboBox Comb_Rate;
	CComboBox Combe_I2CMode;
	CComboBox Combe_I2CBitSet;
	CComboBox Combe_I2C_TimeSet;
	afx_msg void OnBnClickedChange1();
	afx_msg void OnBnClickedChange2();
	afx_msg void OnBnClickedChange3();
	CButton PC_BilateralFilter;
	afx_msg void OnBnClickedNu();
	afx_msg void OnBnClickedUpdate();
	afx_msg void CUDA_Algorithm();
	afx_msg void OnBnClickedWinchange();
	afx_msg void OnBnClickedLocalEnlarge();
	afx_msg void OnStnClickedViewWnd();
	afx_msg void OnStnClickedCancel();
//	CImageExWnd m_ImageWnd2;
	afx_msg void OnStnClickedViewWnd2();
private:
	cv::Mat extractAndResizeCenterView(const cv::Mat& sourceImage);
public:
	afx_msg void OnBnClickedCheck1();
	afx_msg void OnBnClickedDepthCheck();
	afx_msg void OnBnClickedSuperCheck();
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Developer Studio will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GRABDEMODLG_H__82BFE149_F01E_11D1_AF74_00A0C91AC0FB__INCLUDED_)
