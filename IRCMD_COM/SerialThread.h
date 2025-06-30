//SRIALTHREAD_H   ͷ�ļ���ʶ�������ظ����壬
//��ͷ�ļ���ֱ�ӹ�ϵ��������㶨�壬��һ�����ļ����Ĵ�д
#ifndef SRIALTHREAD_H
#define SRIALTHREAD_H


#include "CnComm.h"  
#include "ComFrame.h"
#include <string>
#include <vector>
#include <list>
using namespace std;
class SerialThread 
{

public:
	SerialThread();//��������������Ϊ��             
	virtual ~SerialThread();//��������������Ϊ��

	bool ThreadInit(CString dat, int Rate);//�򿪴���
	void CloseSerialPort(); //�رմ��� 
	void SendDatas(char *msg);//��������  
	void OnReceive(string &mes);  //��������  
	HWND pWnd; //��ȡ�Ի�����   hComm_
    CnComm Com;//ʵ����CnComm��

								  // ���ͼ򵥵Ĵ�������
	bool SendCommand(unsigned char order, unsigned char param);
	// ���ʹ�������
	bool SendCommand(unsigned char order, unsigned char* param, int len);
	bool sendFrame(CComFrame& frame);
	list<CComFrame*> m_listFrames;

	//-------------------------------------
	//CMDFRAME m_CmdFrame;     // ����֡�洢��
	ERRORTYPE m_CurState;    // ����֡��ǰ״̬
	virtual int OnReceive();
	// ���յ���λ��ACK�Ĵ���
	virtual int OnACK(unsigned char* buff, int len);
	// ���践�����ݵ�����ִ�����
	virtual void OnReportExecuteState(unsigned char cmd, ERRORTYPE et);
	// ��ѯ�¶�ֵ����ִ�����
	virtual void OnReportTemperature(int temp);
	// ��ѯϵͳ���ó�ʼ״̬����ִ�����
	virtual void OnReportInitState(const unsigned char* buff, int len);
	// ���ý�������ִ�����
	virtual void OnReportFocusLength(int pos);
	
	struct state
	{
		bool edge_switch;           // ��Ե�񻯿���
		bool pe_switch;             // ͼ����ǿ����
		bool mid_switch;            // ��ֵ�˲�����
		bool st_bp_switch;          // ��̬äԪ����
		bool dy_bp_switch;          // ��̬äԪ����
		bool tp_switch;             // ����У������


		int  P_Upper;               // ƽֵ̨����
		int  P_Lower;               // ƽֵ̨����
		int  brightness;            // ����
		int  contrast;              // �Աȶ�
		int  integration_time;      // ����ʱ��
		int  analog_signal_gain;    // ģ���ź�����
		int  GPol;                  // Gpolֵ��̽ͷƫѹ��

		int focusLength;            // ��ͷ����
	} m_state;

private:

	//���ú��������Ǳ����
	int Ascii2Hex(char* ascii, char* hex);   //asciiתʮ������  
	int Hex2Ascii(char* hex, char* ascii);  //ʮ������תascii  
											// ����Ϊ��֡������


};

#endif

