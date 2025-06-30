
#include "stdafx.h"
#include "ComFrame.h"
#include <string.h>
#include <assert.h>


// Ĭ�Ϲ��캯��
CComFrame::CComFrame()
	: m_CurState( ET_OK )
{
}

// ���ֽڿ鹹��
CComFrame::CComFrame( const unsigned char* buff, int len )
{
	Unserialize( buff, len );
}

// �������ֺͲ�������
CComFrame::CComFrame( unsigned char cmd, const unsigned char* param, int len /* = 1 */, const unsigned char addr /* = 0 */ )
{
	// �������(��һ���ֽ��������ţ�������ǲ���)
	if ( len > MAX_CMD_SIZE - 1 )
	{
		m_CurState = ET_SIZE_ERR;
		return;
	}

	// ��ʼ��֡��ʼ
	m_CmdFrame.BOC  = VALID_BOC;
	// ��ʼ��֡����
	m_CmdFrame.EOC  = VALID_EOC;
	// ��ʼ��֡���
	m_CmdFrame.ADDR = addr;

	// ����֡�غɣ�����֡У��
	m_CmdFrame.SIZE = len+1;
	m_CmdFrame.CMD[0] = cmd;
	m_CmdFrame.CHK = cmd;
	for ( int i = 0; i < len; i++ )
	{
		m_CmdFrame.CMD[i+1] = param[i];  
		m_CmdFrame.CHK ^= param[i];    //��żУ��λ����
	}
	
	// �����Ч��
	m_CurState = CheckValid();
}

// ��ֵ����
CComFrame& CComFrame::operator = (const CComFrame& src)
{
	memcpy( &(this->m_CmdFrame), &(src.m_CmdFrame), sizeof(CMDFRAME) );
	this->m_CurState = src.m_CurState;
	return *this;
}

// ��������
CComFrame::CComFrame( const CComFrame& src )
{
	*this = src;
}

// ��������
CComFrame::~CComFrame()
{
}

// ����������Ƿ���Ч
ERRORTYPE CComFrame::CheckValid()
{
	// �������ʼ�ֽڴ���
	if ( m_CmdFrame.BOC != VALID_BOC )
	{
		return ET_BOC_ERR;
	}

	// ���ȴ���
	if ( m_CmdFrame.SIZE > MAX_CMD_SIZE )
	{
		return ET_SIZE_ERR;
	}

	// У��ʹ���
	unsigned char chk = 0;
	for ( unsigned i = 0; i < m_CmdFrame.SIZE; i++ )
	{
		chk ^= m_CmdFrame.CMD[i];
	}
	if ( chk != m_CmdFrame.CHK )
	{
		return ET_CHECK_ERR;
	}

	// ����������ֽڴ���
	if ( m_CmdFrame.EOC != VALID_EOC )
	{
		return ET_EOC_ERR;
	}

	// ����ʽ
	//switch ( m_CmdFrame.CMD[0] )
 //   {
 //   case COMMAND_EDGE_MODULE_CONTROL:              // ��Ե��ģ�鿪��
 //   case COMMAND_PE_MODULE_CONTROL:                // ͼ����ǿģ�鿪��
 //   case COMMAND_MID_MODULE_CONTROL:               // ��ֵ�˲�ģ�鿪��
 //   case COMMAND_ST_BADPOINT_MODULE_CONTROL:       // ��̬äԪУ��ģ�鿪��
 //   //case COMMAND_DY_BADPOINT_MODULE_CONTROL:       // ��̬äԪУ��ģ�鿪��
 //   case COMMAND_TWOPOINT_MODULE_CONTROL:          // ����У��ģ�鿪��
	//case COMMAND_TEMPERATURE_DISPLAY:              // �¶���ʾ����
 //   //case COMMAND_AD_SWITCH:                        // Ƭ��AD����
 //   case COMMAND_POLAR_CONTROL:                    // ���Կ���
 //   case COMMAND_FRAME_FREQUENCY_CONTROL:          // ֡�ʿ���
 //   //case COMMAND_SCANNING_DIRECTION_CONTROL:       // ɨ�跽�����
 //   case COMMAND_ANALOG_SIGNAL_GAIN_CONTROL:       // ģ���ź��������
 //   //case COMMAND_BENDI_COLLECTION:                 // ���ײɼ�
 //   //case COMMAND_SET_MAX5625_VOLTAGE:              // ����5625оƬ��ѹ
 //   //case COMMAND_SET_RESET2INT:                    // ����reset��INT�½��ص���ʱ
 //   //case COMMAND_SET_RESET2DISPLAYFLAG:            // ����reset��displayflag������
 //   //case COMMAND_PHASE8_CONTROL:                 // 8��λ����
 //   case COMMAND_SINGLE_POINT_CONTROL:             // ����У��
	////case COMMAND_UPLOAD_BENDI:                     // �ϴ�����
 //   //case COMMAND_BP_GATE_CONTROL:                  // äԪ��������
 //   //case COMMAND_BP_SIGMA_CONTROL:                 // äԪsigmaֵ����
	////case COMMAND_BP_READ_RAM_CONTROL:              // ��ȡ��̬äԪram
	//case COMMAND_QUERY_SETTINGS:                   // ����ϵͳ����
	////case COMMAND_RESET_SETTINGS:                   // ����ϵͳ����
	//case COMMAND_NUC_CONTROL:                      // �����ڲ��Ǿ���У��  �����ʽ��0900���ɼ����±��ף�0901���ɼ����±��ף������зǾ���У��GO�����ļ����ʵʱ����У����
 //   case COMMAND_MOTOR_STEP_FORWARD:               // ���������ǰ����
 //   case COMMAND_MOTOR_STEP_BACKWARD:              // ���������󲽽�
	//case COMMAND_MOTOR_ROUTE_FORWARD:              // ֱ�������ת
	//case COMMAND_MOTOR_ROUTE_BACKWARD:             // ֱ�������ת
	//	// �غɳ��ȱ�����2�ֽ�
	//	if ( m_CmdFrame.SIZE != 2 )
	//	{
	//		return ET_SIZE_ERR;
	//	}
 //       break;
	//case COMMAND_INTEGRATION_TIME_CONTROL:         // ����ʱ�����
	//case COMMAND_SET_PCONTROL:                     // ƽֱ̨��ͼ��ƽֵ̨�趨
	//case COMMAND_ZQ_DUIBI_CONTROL:                 // ����-�Աȶȿ���
	//case COMMAND_GPOL_CONTROL:                     // ����Gpolֵ
	//case COMMAND_SERVO_ROUTE:                      // ���ת����ָ���Ƕ�
	//	// �غɳ��ȱ�����5�ֽ�
	//	if ( m_CmdFrame.SIZE != 5 )
	//	{
	//		return ET_SIZE_ERR;
	//	}
	//	break;
	//case COMMAND_UPDATE_BADPOINT_TABLE:            // ����äԪ��
	//case COMMAND_UPDATE_TWOPOINT_TABLE:            // ����У����
	////case COMMAND_UPDATE_PESUDOCOLOR_TABLE:         // ����α�ʱ�
	//	// ������������
	//	break;
 //   default:
	//	// δ��������
	//	return ET_UNKOWN_CMD;
 //       break;
 //   }

	// û�д���
	return ET_OK;
}

// ��ǰ������Ƿ���Ч
bool CComFrame::IsValid() const
{
	return m_CurState == ET_OK;
}

// bool ����
CComFrame::operator bool () const
{
	return m_CurState == ET_OK;
}

// ��ȡ������
ERRORTYPE CComFrame::GetErrorType() const
{
	return m_CurState;
}

// ת��Ϊ�ֽڿ�
ERRORTYPE CComFrame::Serialize( unsigned char * buff, int &len )
{
	if ( m_CurState != ET_OK )
	{
		return m_CurState;
	}

	buff[0] = m_CmdFrame.BOC;
	buff[1] = m_CmdFrame.ADDR;
	buff[2] = m_CmdFrame.SIZE>>16;
	buff[3] = m_CmdFrame.SIZE>>8;
	buff[4] = m_CmdFrame.SIZE;
	for ( unsigned i  = 0; i < m_CmdFrame.SIZE; i++ )
	{
		buff[i+5] = m_CmdFrame.CMD[i];
	}
	buff[m_CmdFrame.SIZE+5] = m_CmdFrame.CHK;
	buff[m_CmdFrame.SIZE+6] = m_CmdFrame.EOC;

	len = m_CmdFrame.SIZE+7;

	return m_CurState;
}

// ���ֽڿ鸳ֵ
ERRORTYPE CComFrame::Unserialize( const unsigned char* buff, int len )
{
	// ��ʼ״̬Ϊû�д���
	m_CurState = ET_OK;

	// �������
	if ( len < 7 )
	{
		m_CurState = ET_SIZE_ERR;
		return m_CurState;
	}

	// ����֡ͷ
	m_CmdFrame.BOC  = buff[0];
	// ��������֡���
	m_CmdFrame.ADDR = buff[1];
	// ��ȡ�غɳ���
	m_CmdFrame.SIZE = buff[2];
	m_CmdFrame.SIZE <<= 8;
	m_CmdFrame.SIZE |= buff[3];
	m_CmdFrame.SIZE <<= 8;
	m_CmdFrame.SIZE |= buff[4];


	// ����ȴ���
	if ( len != m_CmdFrame.SIZE + 7 )
	{
		m_CurState = ET_SIZE_ERR;
		return m_CurState;
	}

	// �����غ�
	for ( unsigned i = 0; i < m_CmdFrame.SIZE; i++ )
	{
		m_CmdFrame.CMD[i] = buff[i+5];
	}
	// �����غ�У��
	m_CmdFrame.CHK = buff[m_CmdFrame.SIZE+5];
	// ����֡β
	m_CmdFrame.EOC = buff[m_CmdFrame.SIZE+6];

	// �����������Ч��
	m_CurState |= CheckValid();
	return m_CurState;
}

// ���ACK
ERRORTYPE CComFrame::CheckACK( CComFrame& ack ) const
{
	// ����ʱ�������ĸ�ʽ����ȷ�������������ʧ��
	assert( m_CurState == ET_OK );

	// ACK��ʽ����ȷӦ���ڵ��ô˺���ǰ����
	if (!ack)
		return 16; //**********
	assert ( ack );

	// ������ƥ�䣬�򷵻�ƥ�����
	if ( ack.GetCmd() != GetCmd() )
	{
		return ET_CMD_MISSMATCH;
	}

	// ��ȡACK�غ�
	const unsigned char* buff = ack.GetParam();
	ERRORTYPE et = buff[0] * 0x10000
		+ buff[1] * 0x100
		+ buff[2];

	return et;
}

// ��ȡ�����֣�����ǰ���ȼ���������Ч�ԣ�
unsigned char CComFrame::GetCmd() const
{
	return m_CmdFrame.CMD[0];
}

// ��ȡ�������ȣ�����ǰ���ȼ���������Ч�ԣ�
int CComFrame::GetParamLen() const
{
	return m_CmdFrame.SIZE-1;
}

// ��ȡ����������ǰ���ȼ���������Ч�ԣ������GetParamLenʹ�ã�
const unsigned char* CComFrame::GetParam() const
{
	return m_CmdFrame.CMD+1;
}

