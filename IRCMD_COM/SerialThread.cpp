
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
		//CnComm::Open�������������򿪴���
		//bool Open(DWORD dwPort, LPCTSTR szPortName, DWORD dwBaudRate, BYTE btParity = NOPARITY, BYTE btByteSize = 8, BYTE btStopBits = ONESTOPBIT)
		//dwPort��szPortName�����ڡ����������->�豸������->�˿ڣ�COM��LPT��->˫���豸����λ�á�
		//if (!Com.Open(7, _T("COM1"), 9600))

		//----------------------------------------------------------------
		// ��ѡ��Ĵ�������ת��ΪLPCTSTR ����
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
			::MessageBox(NULL, _T("���ڴ�ʧ��"), _T("����"), MB_OK);
		}
		return true;
	}
	else {
		cout << "already open" << endl;
		::MessageBox(NULL, _T("�����Ѿ���"), _T("����"), MB_OK);
	}

	//Com.SetWnd(pWnd);

	return false;
}

void SerialThread::CloseSerialPort()
{
	Com.Close();//�رմ���
	cout << "Close Successfully!" << endl;
}

void SerialThread::SendDatas(char *msg)
{
	if (Com.IsOpen())
	{
		Com.Write(msg);//д���ַ�����ʵ�ֿ���

	}
}

void  SerialThread::OnReceive(string &msg)
{
	char buffer[1024];
	do {
		int len = Com.Read(buffer, 1023);//�����ַ���
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

//**************************����ԭ��ͨ��֡��Ϣ**************
// ���ͼ򵥵Ĵ�������
bool SerialThread::SendCommand(unsigned char order, unsigned char param)
{
	return SendCommand(order, &param, 1);
}

// ��������
bool SerialThread::SendCommand(unsigned char order, unsigned char* param, int len)
{
	// �����δ�����������֡������false
	if (!m_listFrames.empty())
	{
		return false;
	}

	int index = 0;
	unsigned char frameNum = 0;
	do  // ��֡
	{
		// ����ֶγ���
		int segLen;
		if (len - index > MAX_CMD_SIZE - 1)
		{
			segLen = MAX_CMD_SIZE - 1;
		}
		else
		{
			segLen = len - index;
		}
		// ��������֡��
		CComFrame* pFrame = new CComFrame(order, param + index, segLen, frameNum);
		if (!pFrame->IsValid())
		{
			return false;
		}
		// ���봦�����
		m_listFrames.push_back(pFrame);

		// ׼����һ֡
		index += segLen;
		frameNum++;

	} while (index < len);

	// ���͵�һ֡
	if (!sendFrame(*(*m_listFrames.begin())))
	{
		//m_listFrames->clear();
		return false;
	}

	return true;
}
// ��������֡
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
	{//! �����ǻ�����ģʽ �����鷵��ֵ��ȷ��������ȫ���ͳ�
		while (!Com.Write(buff, len));
	}

	// ���ͼ������µ���Ϣ
	//Notify(ON_COM_UPDATE_COUNT);

	return true;
}

//-------------------------------------------------------
// ���յ�����ʱ�Ĵ���
int SerialThread::OnReceive()
{
	// ��ȡ���յ�������
	unsigned char buff[1024];
	int len = Com.Read(buff, 1024);

	// ���ͼ������µ���Ϣ
	//Notify(ON_COM_UPDATE_COUNT);

	// ���û�����ڵȴ�ACK�����ֱ�ӷ���
	if (m_listFrames.empty())
	{
		return 66;
	}

	// ����ACK
	return OnACK(buff, len);
}

// ���յ���λ��ACK�Ĵ���
int SerialThread::OnACK(unsigned char* buff, int len)
{
	// ����ACK֡
	CComFrame ack(buff, len);

	ERRORTYPE lastErr = ET_OK;

	// ���ACK֡���������ط���ǰ֡
	if (!ack)
	{
		// �������
		lastErr = ack.GetErrorType();
	}

	// ���ACK���صĽ��������ȷִ����ɣ�������Ӧ����
	// ����ĸ�ʽ����ȷ��ACK��ʽ����ȷ����ǰ����
/*	if (m_CurState != ET_OK || !ack)
	{
		// �ط�
		// ����������ɵİ�
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
		// ���ڶ�֡������м�֡���ط���ǰ֡����������ֱ�ӱ������
		if (m_listFrames.size() > 0)
		{
			// �ط�
			sendFrame(*(*m_listFrames.begin()));
			return 0;
		}
		break;
	}
	// ����������ɵİ�
	delete (*m_listFrames.begin());
	m_listFrames.pop_front();

	// �������һ֡Ҫ����������һ֡
	if (!m_listFrames.empty())
	{
		sendFrame(*(*m_listFrames.begin()));
		return 0;
	}

	// û����һ֡Ҫ���ͣ���������ִ�������
	switch (ack.GetCmd())
	{
	case COMMAND_EDGE_MODULE_CONTROL:              // ��Ե��ģ�鿪��
	case COMMAND_PE_MODULE_CONTROL:                // ͼ����ǿģ�鿪��
	case COMMAND_MID_MODULE_CONTROL:               // ��ֵ�˲�ģ�鿪��
	case COMMAND_ST_BADPOINT_MODULE_CONTROL:       // ��̬äԪУ��ģ�鿪��
												   //case COMMAND_DY_BADPOINT_MODULE_CONTROL:       // ��̬äԪУ��ģ�鿪��
	case COMMAND_TWOPOINT_MODULE_CONTROL:          // ����У��ģ�鿪��
												   //case COMMAND_AD_SWITCH:                        // Ƭ��AD����
	case COMMAND_POLAR_CONTROL:                    // ���Կ���
	case COMMAND_FRAME_FREQUENCY_CONTROL:          // ֡�ʿ���
												   //case COMMAND_SCANNING_DIRECTION_CONTROL:       // ɨ�跽�����
	case COMMAND_ANALOG_SIGNAL_GAIN_CONTROL:       // ģ���ź��������
												   //case COMMAND_BENDI_COLLECTION:                 // ���ײɼ�
												   //case COMMAND_SET_MAX5625_VOLTAGE:              // ����5625оƬ��ѹ
												   //case COMMAND_SET_RESET2INT:                    // ����reset��INT�½��ص���ʱ
												   //case COMMAND_SET_RESET2DISPLAYFLAG:            // ����reset��displayflag������
												   //case COMMAND_PHASE8_CONTROL:                 // 8��λ����
	case COMMAND_SINGLE_POINT_CONTROL:             // ����У��
												   //case COMMAND_UPLOAD_BENDI:                     // �ϴ�����
												   //case COMMAND_BP_GATE_CONTROL:                  // äԪ��������
												   //case COMMAND_BP_SIGMA_CONTROL:                 // äԪsigmaֵ����
												   //case COMMAND_BP_READ_RAM_CONTROL:              // ��ȡ��̬äԪram
												   //case COMMAND_RESET_SETTINGS:                   // ����ϵͳ����
	case COMMAND_NUC_CONTROL:                      // �����ڲ��Ǿ���У��  �����ʽ��0900���ɼ����±��ף�0901���ɼ����±��ף������зǾ���У��GO�����ļ����ʵʱ����У����
	case COMMAND_MOTOR_STEP_FORWARD:               // ���������ǰ����
	case COMMAND_MOTOR_STEP_BACKWARD:              // ���������󲽽�
	case COMMAND_INTEGRATION_TIME_CONTROL:         // ����ʱ�����
	case COMMAND_INTEGRATION_TIME2_CONTROL:        // �ڶ�����ʱ�����
	case COMMAND_SET_PCONTROL:                     // ƽֱ̨��ͼ��ƽֵ̨�趨
	case COMMAND_ZQ_DUIBI_CONTROL:                 // ����-�Աȶȿ���
	case COMMAND_GPOL_CONTROL:                     // ����Gpolֵ
	case COMMAND_SERVO_ROUTE:                      // ���ת����ָ���Ƕ�
	case COMMAND_UPDATE_BADPOINT_TABLE:            // ����äԪ��
	case COMMAND_DATA_MERGE_CONTROL:                // �����ںϿ���
													//case COMMAND_UPDATE_TWOPOINT_TABLE:            // ����У����
													//case COMMAND_UPDATE_PESUDOCOLOR_TABLE:         // ����α�ʱ�
													// ��Щ�����ACK���������ݣ�ǰ���ִ�н������ET_OK������������ط�������ֻʣ��ET_OK��
		OnReportExecuteState(ack.GetCmd(), lastErr);
		break;
	case COMMAND_TEMPERATURE_DISPLAY:              // �¶���ʾ
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
	case COMMAND_QUERY_SETTINGS:                   // ����ϵͳ����
												   //if ( lastErr == ET_OK && ack.GetParamLen()-3 == 22 )
												   //{
		OnReportInitState(ack.GetParam() + 3, ack.GetParamLen() - 3);
		//}
		//else
		//{
		//	OnReportExecuteState( ack.GetCmd(), lastErr );
		//}
		break;
	case COMMAND_MOTOR_ROUTE_FORWARD:              // ֱ�������ת
	case COMMAND_MOTOR_ROUTE_BACKWARD:             // ֱ�������ת
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

// ���践�����ݵ�����ִ�����
void SerialThread::OnReportExecuteState(unsigned char cmd, ERRORTYPE et)
{
	// �������et����ϳ���Ϣ����
	LPARAM lp = cmd;
	lp <<= 24;
	lp |= et;
	// ������Ϣ
	//Notify(ON_REPORT_STATE, lp);
}

// ��ѯ�¶�ֵ����ִ�����
void SerialThread::OnReportTemperature(int temp)
{
	//Notify(ON_REPORT_TEMPERATURE, temp);
}

// ��ѯϵͳ���ó�ʼ״̬����ִ�����
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

// ���ý�������ִ�����
void SerialThread::OnReportFocusLength(int pos)
{
	//Notify(ON_REPORT_FOCUS_LENGTH, pos);
} //namespace IRCMD
