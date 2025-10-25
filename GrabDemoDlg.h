// GrabDemoDlg.h : header file
//
#include <stdio.h>  
#include "SapClassBasic.h"
#include "SapClassGui.h"

// GPU ͷ�ļ�
#include <stdio.h>  
#include "cuda_runtime.h"  
#include "device_launch_parameters.h"  
#include "afxwin.h"
#include "DistortionCailbration.h"
#include "MyTensorRT.h"
#include "tracker_utils.hpp"
// GPU ��������
extern "C"
cudaError_t Image_Solution(unsigned short *Image, int Length, int Width, float *pTP_Gain, float *pTP_Bias, int TP_On,int Blind_On, unsigned short *pBlind_Ram, int Histogram_On);  //  pTP_Gain,pTP_Biasָ�������������----TP_On��ʾ�Ƿ����������
//ֱ��ͼ��ǿ
extern "C" 
cudaError_t GPU_Histogram_Enhancement(unsigned short *Image, unsigned int *Histogram,float * Histogram_Float,unsigned short *dev_img, unsigned int *dev_Histogram,float* dev_Histogram_float, int Length, int Width);
// �����������
extern "C"
cudaError_t GPU_TwoPoint_Correction(unsigned short *Image, unsigned short *dev_img, float* dev_pTP_Gain, float* dev_pTP_Bias, int Length, int Width);
//äԪ��������
extern "C"
cudaError_t GPU_Blind_Correction(unsigned short *Image, unsigned short *dev_img, unsigned short *dev_pBlind_Ram, int Length, int Width);
#if !defined(AFX_GRABDEMODLG_H__82BFE149_F01E_11D1_AF74_00A0C91AC0FB__INCLUDED_)
#define AFX_GRABDEMODLG_H__82BFE149_F01E_11D1_AF74_00A0C91AC0FB__INCLUDED_

#if _MSC_VER >= 1000
#pragma once
#endif // _MSC_VER >= 1000


/////////////////////////////////////////////////////////////////////////////
// CGrabDemoDlg dialog

class CGrabDemoDlg : public CDialog, public CImageExWndEventHandler // �̳���CDialog��CImageExWndEventHandler
{
	// Construction
public:
	DistortionCailbration distortionCailbration; // ����У׼����
	CGrabDemoDlg(CWnd* pParent = NULL);	// ��׼���캯��

	BOOL CreateObjects(); // ��������
	BOOL DestroyObjects(); // ���ٶ���
	void UpdateMenu(); // ���²˵�
	static void XferCallback(SapXferCallbackInfo* pInfo); // ���ݴ���ص�����
	static void SignalCallback(SapAcqCallbackInfo* pInfo); // �źŻص�����
	void GetSignalStatus(); // ��ȡ�ź�״̬
	void GetSignalStatus(SapAcquisition::SignalStatus signalStatus); // ��ȡָ���ź�״̬
	void PixelChanged(int x, int y); // ���ظı��¼�����
	void FPGA_Send(); // �������ݵ�FPGA
	void FPGA_Receive(); // ��FPGA��������
 // Dialog Data
	 //{{AFX_DATA(CGrabDemoDlg)
	enum { IDD = IDD_GRABDEMO_DIALOG }; // �Ի�����ԴID
	float       m_BufferFrameRate; // ����֡��
	CStatic	m_statusWnd; // ״̬���ڿؼ�
	//}}AFX_DATA

	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CGrabDemoDlg)
protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV֧��
	//}}AFX_VIRTUAL

// Implementation
protected:
	HICON		m_hIcon; // ͼ����
	CString  m_appTitle; // Ӧ�ó������

	CString m_cstrWorkPath;    // ����·������Ųɼ����ݺͼ����м�����
	int m_imageHeight;             // ͼ��߶�
	int m_imageWidth;              // ͼ����

	CImageExWnd		m_ImageWnd; // ��ʾͼ��Ĵ���
	SapAcquisition* m_Acq; // �ɼ�����ָ��
	SapBuffer* m_Buffers; // ����������ָ��
	SapTransfer* m_Xfer; // �������ָ��
	SapView* m_View; // ��ͼ����ָ��
	//SapView        *m_ViewProcessed;

	static CGrabDemoDlg* m_DlgPointer;  // �ص��������ʷǾ�̬��Ա
	static MyTensorRT* m_super_tensorRT; // TensorRT���ֱ���ģ��ָ��
	static MyTensorRT* m_detect_tensorRT; // TensorRT���ģ��ָ��
	static MyTensorRT* m_depth_tensorRT; // TensorRT��ȹ���ģ��ָ��
	static MyTensorRT* m_track_tensorRT; // TensorRT��ȹ���ģ��ָ��
	BOOL m_IsSignalDetected;   // TRUE if camera signal is detected ����ź��Ƿ񱻼�⵽

	 // Generated message map functions
	 //{{AFX_MSG(CGrabDemoDlg)
	virtual BOOL OnInitDialog(); // ��ʼ���Ի���
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam); // ϵͳ������Ϣ
	afx_msg void OnPaint(); // ������Ϣ
	afx_msg HCURSOR OnQueryDragIcon(); // ��ѯ�϶�ͼ��
	afx_msg void OnDestroy(); // ������Ϣ
	afx_msg void OnSize(UINT nType, int cx, int cy); // ��С�仯��Ϣ
	afx_msg void OnSnap(); // ���հ�ť�����Ϣ
	afx_msg void OnGrab(); // ץȡ��ť�����Ϣ
	afx_msg void OnFreeze(); // ���ᰴť�����Ϣ
	afx_msg void OnGeneralOptions(); // ͨ��ѡ�ť�����Ϣ
	afx_msg void OnAreaScanOptions(); // ����ɨ��ѡ�ť�����Ϣ
	afx_msg void OnLineScanOptions(); // ��ɨ��ѡ�ť�����Ϣ
	afx_msg void OnCompositeOptions(); // ����ѡ�ť�����Ϣ
	afx_msg void OnLoadAcqConfig(); // ���زɼ����ð�ť�����Ϣ
	afx_msg void OnImageFilterOptions(); // ͼ���˲�ѡ�ť�����Ϣ
	afx_msg void OnBufferOptions(); // ������ѡ�ť�����Ϣ
	afx_msg void OnViewOptions(); // ��ͼѡ�ť�����Ϣ
	afx_msg void OnFileLoad(); // �ļ����ذ�ť�����Ϣ
	afx_msg void OnFileNew(); // �½��ļ���ť�����Ϣ
	afx_msg void OnFileSave(); // �ļ����水ť�����Ϣ
	afx_msg void OnExit(); // �˳���ť�����Ϣ
	afx_msg void OnEndSession(BOOL bEnding); // �����Ự��Ϣ
	afx_msg BOOL OnQueryEndSession(); // ��ѯ�����Ự��Ϣ
	afx_msg void OnKillfocusBufferFrameRate(void); // BufferFrameRateʧȥ����ʱ����Ϣ

	afx_msg void OnTimer(UINT_PTR nIDEvent); // ��ʱ����Ϣ

	 //}}AFX_MSG
	DECLARE_MESSAGE_MAP()

public:
	afx_msg void OnBnClickedSavemulti(); // "savemulti"��ť�����Ϣ

	afx_msg void OnBnClickedSaveTiming(); // "�����ʱ"��ť�����Ϣ
	afx_msg void OnBnClickedTimingStop(); // "ֹͣ��ʱ"��ť�����Ϣ
	CButton GPU_State; // GPU״̬��ť
	CButton TP_Correction; // ��ƯУ����ť
	afx_msg void OnBnClickedTpCold(); // ��Ư���У����ť�����Ϣ
	afx_msg void OnBnClickedTpHot(); // ��Ư�ȶ�У����ť�����Ϣ
	CButton Blind_Correction; // äԪУ����ť
	CButton H_Enhance; // ����ǿ��ť
	afx_msg void OnBnClickedHEnhance(); // ����ǿ��ť�����Ϣ
	afx_msg void OnBnClickedGpu(); // GPU��ť�����Ϣ

	afx_msg void OnBnClickedGetlow(); // ��ȡ��ֵ��ť�����Ϣ
	afx_msg void OnBnClickedNetd(); // NETD��ť�����Ϣ

	void CalculateNETD(int& flag, int Height, int Width); // ����NETD

	afx_msg void OnBnClickedOk(); // OK��ť�����Ϣ
	CComboBox m_Combo; // ��Ͽ�ؼ�
	afx_msg void OnBnClickedOpen(); // �򿪰�ť�����Ϣ
	afx_msg void OnBnClickedClose(); // �رհ�ť�����Ϣ
	afx_msg void OnBnClickedHe(); // HE��ť�����Ϣ
	// ��λ�����ڽ��յ����ݵ��жϺ���
	afx_msg void localEnlarge(int Height, int Widt); // �ֲ��Ŵ���
	LRESULT CGrabDemoDlg::SerialRead(WPARAM, LPARAM); // ���ڶ�ȡ��Ϣ
	afx_msg void OnBnClickedGethigh(); // ��ȡ��ֵ��ť�����Ϣ
	afx_msg void OnBnClickedBlindCorrection(); // äԪУ����ť�����Ϣ
	afx_msg void OnBnClickedTpCorrection(); // ��ƯУ����ť�����Ϣ
	afx_msg void OnBnClickedMedianFilter(); // ��ֵ�˲���ť�����Ϣ
	afx_msg void OnBnClickedtest(); // ���԰�ť�����Ϣ
	afx_msg void OnBnClickedTest(); // ��һ�����԰�ť�����Ϣ
	afx_msg void OnBnClickedBpmap(); // BPӳ�䰴ť�����Ϣ
	afx_msg void OnBnClickedTpmap(); // ��Ưӳ�䰴ť�����Ϣ
	afx_msg void OnBnClickedIntegral(); // ���ְ�ť�����Ϣ
	afx_msg void OnBnClickedBitrate(); // �����ʰ�ť�����Ϣ
	int mHeight; // ͼ��߶�
	int mWidth; // ͼ����
	int imageBits; // ͼ�������

	afx_msg void OnMouseMove(UINT nFlags, CPoint point); // ����ƶ���Ϣ
	CComboBox Comb_Rate; // ��Ͽ�ؼ�����ѡ������
	CComboBox Combe_I2CMode; // I2Cģʽ��Ͽ�
	CComboBox Combe_I2CBitSet; // I2C����������Ͽ�
	CComboBox Combe_I2C_TimeSet; // I2Cʱ��������Ͽ�
	afx_msg void OnBnClickedChange1(); // �ı�1��ť�����Ϣ
	afx_msg void OnBnClickedChange2(); // �ı�2��ť�����Ϣ
	afx_msg void OnBnClickedChange3(); // �ı�3��ť�����Ϣ
	CButton PC_BilateralFilter; // ˫���˲���ť
	afx_msg void OnBnClickedNu(); // NU��ť�����Ϣ
	afx_msg void OnBnClickedUpdate(); // ���°�ť�����Ϣ
	afx_msg void CUDA_Algorithm(); // CUDA�㷨
	afx_msg void OnBnClickedWinchange(); // ���ڸı䰴ť�����Ϣ
	afx_msg void OnBnClickedLocalEnlarge(); // �������
	//afx_msg void OnStnClickedViewWnd(); // ViewWnd�ؼ�������Ϣ
	afx_msg void OnStnClickedCancel(); // Cancel�ؼ�������Ϣ
//	CImageExWnd m_ImageWnd2;
	afx_msg void OnStnClickedViewWnd2(); // ViewWnd2�ؼ�������Ϣ
private:
	cv::Mat extractAndResizeCenterView(const cv::Mat& sourceImage); // ��ȡ������������ͼ��С
	TrackerUtils* m_tracker = nullptr;   // ����ģ��ָ��
	TrackingResult m_trackResult;        // �洢ÿ֡���ٽ��
	bool m_enableTracking = false;       // �Ƿ�������
public:
	afx_msg void OnBnClickedCheck1(); // Ŀ����
	afx_msg void OnBnClickedDepthCheck(); // ��ȼ�
	afx_msg void OnBnClickedSuperCheck(); // �ӽǺϳ�
	afx_msg void OnCbnSelchangeCombo1(); // Combo1ѡ����ı���Ϣ
	cv::Mat m_saiHost;   // ���ڱ��泬�ֱ��ʽ��					 
	CComboBox m_comboBox;  // �ƣ��꣬��������ѡ��ѡ��
	// CGrabDemoDlg.h ������
private:
	void UpdateSavePath(); // ���� m_cstrWorkPath
public:
	afx_msg void OnBnClickedTrackCheck();
	afx_msg void OnEnChangeframes();
};
//{{AFX_INSERT_LOCATION}}
// Microsoft Developer Studio will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_GRABDEMODLG_H__82BFE149_F01E_11D1_AF74_00A0C91AC0FB__INCLUDED_)
