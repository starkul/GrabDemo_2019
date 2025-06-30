
#include "stdafx.h"
#include <iostream>  
#include "SerialThread.h"  
#include "ComFrame.h"
using namespace std;


SerialThread::SerialThread()
{
}

SerialThread::~SerialThread()
{

}

bool SerialThread::ThreadInit(CString dat,int Rate)
{
	if (!Com.IsOpen())
	{
		//CnComm::Open函数就是用来打开串口
		//bool Open(DWORD dwPort, LPCTSTR szPortName, DWORD dwBaudRate, BYTE btParity = NOPARITY, BYTE btByteSize = 8, BYTE btStopBits = ONESTOPBIT)
		//dwPort，szPortName可以在“计算机管理->设备管理器->端口（COM和LPT）->双击设备名：位置”
		//if (!Com.Open(7, _T("COM1"), 9600))

		//----------------------------------------------------------------
		// 将选择的串口名字转换为LPCTSTR 类型
		string tmp(CW2A(dat.GetString()));
		char cTmp[50];
		tmp = "\\\\.\\" + tmp;
		strcpy(cTmp, tmp.c_str());
		int num = MultiByteToWideChar(0, 0, cTmp, -1, NULL, 0);
		wchar_t *wide = new wchar_t[num];
		MultiByteToWideChar(0, 0, cTmp, -1, wide, num);
		//----------------------------------------------------------------

	    //if (!Com.Open(17, _T("\\\\.\\Xcelera-CL_PX4_1_Serial_0"), 9600))
		//if (!Com.Open(17, wide, 115200))
		if (!Com.Open(17, wide, Rate))
	    //if (!Com.Open(2, wide, 9600))
		//if (!Com.Open(dat, 9600))//
		{
			//cout << "open fail" << endl;
			::MessageBox(NULL, _T("串口打开失败"), _T("错误"), MB_OK);
		}
		return true;
	}
	else {
		cout << "already open" << endl;
		::MessageBox(NULL, _T("串口已经打开"), _T("错误"), MB_OK);
	}

	//Com.SetWnd(pWnd);

	return false;
}

void SerialThread::CloseSerialPort()
{
	Com.Close();//关闭串口
	cout << "Close Successfully!" << endl;
}

void SerialThread::SendDatas(char *msg)
{
	if (Com.IsOpen())
	{
		Com.Write(msg);//写入字符串，实现控制

	}
}

void  SerialThread::OnReceive(string &msg)
{
	char buffer[1024];
	do {
		int len = Com.Read(buffer, 1023);//接受字符串
		buffer[len] = _T('\0');
		cout << buffer << endl;
	} while (Com.IsRxBufferMode() && Com.Input().SafeSize());
	string tmp(buffer); msg = tmp;
}

int SerialThread::Ascii2Hex(char* ascii, char* hex)
{
	int i, len = strlen(ascii);
	char chHex[] = "0123456789ABCDEF";

	for (i = 0; i < len; i++)
	{
		hex[i * 3] = chHex[((BYTE)ascii[i]) >> 4];
		hex[i * 3 + 1] = chHex[((BYTE)ascii[i]) & 0xf];
		hex[i * 3 + 2] = ' ';
	}

	hex[len * 3] = '\0';

	return len * 3;
}

int SerialThread::Hex2Ascii(char* hex, char* ascii)
{
	int len = strlen(hex), tlen, i, cnt;

	for (i = 0, cnt = 0, tlen = 0; i < len; i++)
	{
		char c = toupper(hex[i]);

		if ((c >= '0'&& c <= '9') || (c >= 'A'&& c <= 'F'))
		{
			BYTE t = (c >= 'A') ? c - 'A' + 10 : c - '0';

			if (cnt)
				ascii[tlen++] += t, cnt = 0;
			else
				ascii[tlen] = t << 4, cnt = 1;
		}
	}

	return tlen;
}

//**************************兼容原来通信帧信息**************
// 发送简单的串口命令
bool SerialThread::SendCommand(unsigned char order, unsigned char param)
{
	return SendCommand(order, &param, 1);
}

// 发送命令
bool SerialThread::SendCommand(unsigned char order, unsigned char* param, int len)
{
	// 如果有未处理完的命令帧，返回false
	if (!m_listFrames.empty())
	{
		return false;
	}

	int index = 0;
	unsigned char frameNum = 0;
	do  // 分帧
	{
		// 计算分段长度
		int segLen;
		if (len - index > MAX_CMD_SIZE - 1)
		{
			segLen = MAX_CMD_SIZE - 1;
		}
		else
		{
			segLen = len - index;
		}
		// 构造命令帧类
		CComFrame* pFrame = new CComFrame(order, param + index, segLen, frameNum);
		if (!pFrame->IsValid())
		{
			return false;
		}
		// 加入处理队列
		m_listFrames.push_back(pFrame);

		// 准备下一帧
		index += segLen;
		frameNum++;

	} while (index < len);

	// 发送第一帧
	if (!sendFrame(*(*m_listFrames.begin())))
	{
		//m_listFrames->clear();
		return false;
	}

	return true;
}
// 发送命令帧
bool SerialThread::sendFrame(CComFrame& frame)
{
	unsigned char buff[MAX_FRAME_LEN];
	int len;
	if (frame.Serialize(buff, len) != ET_OK)
	{
		return false;
	}

	if (Com.IsOverlappedMode() || Com.IsTxBufferMode())
	{
		Com.Write(buff, len);
	}
	else
	{//! 阻塞非缓冲区模式 必须检查返回值，确保数据完全发送出
		while (!Com.Write(buff, len));
	}

	// 发送计数更新的消息
	//Notify(ON_COM_UPDATE_COUNT);

	return true;
}

//-------------------------------------------------------
// 接收到数据时的处理
int SerialThread::OnReceive()
{
	// 读取接收到的数据
	unsigned char buff[1024];
	int len = Com.Read(buff, 1024);

	// 发送计数更新的消息
	//Notify(ON_COM_UPDATE_COUNT);

	// 如果没有正在等待ACK的命令，直接返回
	if (m_listFrames.empty())
	{
		return 66;
	}

	// 处理ACK
	return OnACK(buff, len);
}

// 接收到下位机ACK的处理
int SerialThread::OnACK(unsigned char* buff, int len)
{
	// 构造ACK帧
	CComFrame ack(buff, len);

	ERRORTYPE lastErr = ET_OK;

	// 如果ACK帧不正常，重发当前帧
	if (!ack)
	{
		// 报告错误
		lastErr = ack.GetErrorType();
	}

	// 如果ACK返回的结果不是正确执行完成，则作相应处理
	// 本身的格式不正确或ACK格式不正确，提前处理
/*	if (m_CurState != ET_OK || !ack)
	{
		// 重发
		// 丢弃发送完成的包
		sendFrame(*(*m_listFrames.begin()));
		return;
	}
*/
	lastErr = (*m_listFrames.begin())->CheckACK(ack);
	switch (lastErr)
	{
	case ET_OK:
		break;
	case ET_PARAM_ERR:
		//return ET_PARAM_ERR;
	case ET_EXECUSE_ERR:
		//return ET_EXECUSE_ERR;
	case ET_UNKNOWN_CMD:
		//return ET_UNKNOWN_CMD;
	case ET_BOC_ERR:
		//return ET_BOC_ERR;
	case ET_TIMEOUT:
		//return ET_TIMEOUT;
	case ET_CMD_MISSMATCH:
		//return ET_CMD_MISSMATCH;
	case ET_CHECK_ERR:
		//return ET_CHECK_ERR;
	case ET_SIZE_ERR:
		//return ET_SIZE_ERR;
	case ET_EOC_ERR:
		//return ET_EOC_ERR;
	default:
		// 对于多帧命令的中间帧，重发当前帧，其他命令直接报告错误
		if (m_listFrames.size() > 0)
		{
			// 重发
			sendFrame(*(*m_listFrames.begin()));
			return 0;
		}
		break;
	}
	// 丢弃发送完成的包
	delete (*m_listFrames.begin());
	m_listFrames.pop_front();

	// 如果有下一帧要发送则发送下一帧
	if (!m_listFrames.empty())
	{
		sendFrame(*(*m_listFrames.begin()));
		return 0;
	}

	// 没有下一帧要发送，则是命令执行完成了
	switch (ack.GetCmd())
	{
	case COMMAND_EDGE_MODULE_CONTROL:              // 边缘锐化模块开关
	case COMMAND_PE_MODULE_CONTROL:                // 图像增强模块开关
	case COMMAND_MID_MODULE_CONTROL:               // 中值滤波模块开关
	case COMMAND_ST_BADPOINT_MODULE_CONTROL:       // 静态盲元校正模块开关
												   //case COMMAND_DY_BADPOINT_MODULE_CONTROL:       // 动态盲元校正模块开关
	case COMMAND_TWOPOINT_MODULE_CONTROL:          // 两点校正模块开关
												   //case COMMAND_AD_SWITCH:                        // 片上AD开关
	case COMMAND_POLAR_CONTROL:                    // 极性控制
	case COMMAND_FRAME_FREQUENCY_CONTROL:          // 帧率控制
												   //case COMMAND_SCANNING_DIRECTION_CONTROL:       // 扫描方向控制
	case COMMAND_ANALOG_SIGNAL_GAIN_CONTROL:       // 模拟信号增益控制
												   //case COMMAND_BENDI_COLLECTION:                 // 本底采集
												   //case COMMAND_SET_MAX5625_VOLTAGE:              // 设置5625芯片电压
												   //case COMMAND_SET_RESET2INT:                    // 设置reset到INT下降沿的延时
												   //case COMMAND_SET_RESET2DISPLAYFLAG:            // 设置reset到displayflag的行数
												   //case COMMAND_PHASE8_CONTROL:                 // 8相位控制
	case COMMAND_SINGLE_POINT_CONTROL:             // 单点校正
												   //case COMMAND_UPLOAD_BENDI:                     // 上传本底
												   //case COMMAND_BP_GATE_CONTROL:                  // 盲元门限设置
												   //case COMMAND_BP_SIGMA_CONTROL:                 // 盲元sigma值设置
												   //case COMMAND_BP_READ_RAM_CONTROL:              // 读取动态盲元ram
												   //case COMMAND_RESET_SETTINGS:                   // 重置系统参数
	case COMMAND_NUC_CONTROL:                      // 进行内部非均匀校正  命令格式：0900：采集低温本底；0901：采集高温本底，并进行非均匀校正GO参数的计算和实时更新校正表
	case COMMAND_MOTOR_STEP_FORWARD:               // 步进电机向前步进
	case COMMAND_MOTOR_STEP_BACKWARD:              // 步进电机向后步进
	case COMMAND_INTEGRATION_TIME_CONTROL:         // 积分时间控制
	case COMMAND_INTEGRATION_TIME2_CONTROL:        // 第二积分时间控制
	case COMMAND_SET_PCONTROL:                     // 平台直方图，平台值设定
	case COMMAND_ZQ_DUIBI_CONTROL:                 // 亮度-对比度控制
	case COMMAND_GPOL_CONTROL:                     // 设置Gpol值
	case COMMAND_SERVO_ROUTE:                      // 舵机转动到指定角度
	case COMMAND_UPDATE_BADPOINT_TABLE:            // 更新盲元表
	case COMMAND_DATA_MERGE_CONTROL:                // 数据融合开关
													//case COMMAND_UPDATE_TWOPOINT_TABLE:            // 更新校正表
													//case COMMAND_UPDATE_PESUDOCOLOR_TABLE:         // 更新伪彩表
													// 这些命令的ACK不包含数据（前面把执行结果不是ET_OK的命令都进行了重发，这里只剩下ET_OK）
		OnReportExecuteState(ack.GetCmd(), lastErr);
		break;
	case COMMAND_TEMPERATURE_DISPLAY:              // 温度显示
		if (lastErr == ET_OK && ack.GetParamLen() - 3 == 4)
		{
			const unsigned char *tempBuff = ack.GetParam() + 3;
			int temp = tempBuff[0] * 0x1000000
				+ tempBuff[1] * 0x10000
				+ tempBuff[2] * 0x100
				+ tempBuff[3];
			OnReportTemperature(temp);
		}
		else
		{
			OnReportExecuteState(ack.GetCmd(), lastErr);
		}
		break;
	case COMMAND_QUERY_SETTINGS:                   // 请求系统参数
												   //if ( lastErr == ET_OK && ack.GetParamLen()-3 == 22 )
												   //{
		OnReportInitState(ack.GetParam() + 3, ack.GetParamLen() - 3);
		//}
		//else
		//{
		//	OnReportExecuteState( ack.GetCmd(), lastErr );
		//}
		break;
	case COMMAND_MOTOR_ROUTE_FORWARD:              // 直流电机正转
	case COMMAND_MOTOR_ROUTE_BACKWARD:             // 直流电机反转
		if (lastErr == ET_OK && ack.GetParamLen() - 3 == 4)
		{
			const unsigned char *posBuff = ack.GetParam() + 3;
			int pos = posBuff[0] * 0x1000000
				+ posBuff[1] * 0x10000
				+ posBuff[2] * 0x100
				+ posBuff[3];
			OnReportFocusLength(pos);
		}
		else
		{
			OnReportExecuteState(ack.GetCmd(), lastErr);
		}
		break;
	default:
		break;
	}
	return 0;
}

// 无需返回数据的命令执行完成
void SerialThread::OnReportExecuteState(unsigned char cmd, ERRORTYPE et)
{
	// 将命令和et码组合成消息参数
	LPARAM lp = cmd;
	lp <<= 24;
	lp |= et;
	// 发送消息
	//Notify(ON_REPORT_STATE, lp);
}

// 查询温度值命令执行完成
void SerialThread::OnReportTemperature(int temp)
{
	//Notify(ON_REPORT_TEMPERATURE, temp);
}

// 查询系统设置初始状态命令执行完成
void SerialThread::OnReportInitState(const unsigned char* buff, int len)
{
	m_state.GPol = buff[2] * 256 + buff[3];
	m_state.integration_time = buff[4] * 256 + buff[5];
	m_state.analog_signal_gain = buff[6] * 256 + buff[7];
	m_state.P_Upper = buff[8] * 256 + buff[9];
	m_state.P_Lower = buff[10] * 256 + buff[11];
	m_state.brightness = buff[12] * 256 + buff[13];
	m_state.contrast = buff[14] * 256 + buff[15];
	m_state.edge_switch = buff[17] & (0x01 << 4);
	m_state.pe_switch = buff[17] & (0x01 << 3);
	m_state.mid_switch = buff[17] & (0x01 << 2);
	m_state.st_bp_switch = buff[17] & (0x01 << 7);
	m_state.dy_bp_switch = buff[17] & (0x01 << 1);
	m_state.tp_switch = buff[17] & 0x01;
	m_state.focusLength = buff[18] * 0x1000000
		+ buff[19] * 0x10000 + buff[20] * 0x100 + buff[21];
	//Notify(ON_REPORT_FOCUS_LENGTH, m_state.focusLength);
	//Notify(ON_REPORT_INIT_STATE);
}

// 设置焦距命令执行完成
void SerialThread::OnReportFocusLength(int pos)
{
	//Notify(ON_REPORT_FOCUS_LENGTH, pos);
} //namespace IRCMD
