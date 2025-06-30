//SRIALTHREAD_H   头文件标识，避免重复定义，
//和头文件无直接关系，可以随便定义，但一般是文件名的大写
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
	SerialThread();//构建函数，可以为空             
	virtual ~SerialThread();//解析函数，可以为空

	bool ThreadInit(CString dat, int Rate);//打开串口
	void CloseSerialPort(); //关闭串口 
	void SendDatas(char *msg);//发送数据  
	void OnReceive(string &mes);  //接收数据  
	HWND pWnd; //获取对话框句柄   hComm_
    CnComm Com;//实例化CnComm类

								  // 发送简单的串口命令
	bool SendCommand(unsigned char order, unsigned char param);
	// 发送串口命令
	bool SendCommand(unsigned char order, unsigned char* param, int len);
	bool sendFrame(CComFrame& frame);
	list<CComFrame*> m_listFrames;

	//-------------------------------------
	//CMDFRAME m_CmdFrame;     // 命令帧存储区
	ERRORTYPE m_CurState;    // 命令帧当前状态
	virtual int OnReceive();
	// 接收到下位机ACK的处理
	virtual int OnACK(unsigned char* buff, int len);
	// 无需返回数据的命令执行完成
	virtual void OnReportExecuteState(unsigned char cmd, ERRORTYPE et);
	// 查询温度值命令执行完成
	virtual void OnReportTemperature(int temp);
	// 查询系统设置初始状态命令执行完成
	virtual void OnReportInitState(const unsigned char* buff, int len);
	// 设置焦距命令执行完成
	virtual void OnReportFocusLength(int pos);
	
	struct state
	{
		bool edge_switch;           // 边缘锐化开关
		bool pe_switch;             // 图像增强开关
		bool mid_switch;            // 中值滤波开关
		bool st_bp_switch;          // 静态盲元开关
		bool dy_bp_switch;          // 动态盲元开关
		bool tp_switch;             // 两点校正开关


		int  P_Upper;               // 平台值上限
		int  P_Lower;               // 平台值下限
		int  brightness;            // 亮度
		int  contrast;              // 对比度
		int  integration_time;      // 积分时间
		int  analog_signal_gain;    // 模拟信号增益
		int  GPol;                  // Gpol值（探头偏压）

		int focusLength;            // 镜头焦距
	} m_state;

private:

	//备用函数，不是必须的
	int Ascii2Hex(char* ascii, char* hex);   //ascii转十六进制  
	int Hex2Ascii(char* hex, char* ascii);  //十六进制转ascii  
											// 被分为多帧的命令


};

#endif

