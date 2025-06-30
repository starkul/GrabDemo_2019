
#include "stdafx.h"
#include "ComFrame.h"
#include <string.h>
#include <assert.h>


// 默认构造函数
CComFrame::CComFrame()
	: m_CurState( ET_OK )
{
}

// 从字节块构造
CComFrame::CComFrame( const unsigned char* buff, int len )
{
	Unserialize( buff, len );
}

// 从命令字和参数构造
CComFrame::CComFrame( unsigned char cmd, const unsigned char* param, int len /* = 1 */, const unsigned char addr /* = 0 */ )
{
	// 命令过长(第一个字节是命令编号，后面的是参数)
	if ( len > MAX_CMD_SIZE - 1 )
	{
		m_CurState = ET_SIZE_ERR;
		return;
	}

	// 初始化帧起始
	m_CmdFrame.BOC  = VALID_BOC;
	// 初始化帧结束
	m_CmdFrame.EOC  = VALID_EOC;
	// 初始化帧编号
	m_CmdFrame.ADDR = addr;

	// 拷贝帧载荷，计算帧校验
	m_CmdFrame.SIZE = len+1;
	m_CmdFrame.CMD[0] = cmd;
	m_CmdFrame.CHK = cmd;
	for ( int i = 0; i < len; i++ )
	{
		m_CmdFrame.CMD[i+1] = param[i];  
		m_CmdFrame.CHK ^= param[i];    //奇偶校验位计算
	}
	
	// 检查有效性
	m_CurState = CheckValid();
}

// 赋值操作
CComFrame& CComFrame::operator = (const CComFrame& src)
{
	memcpy( &(this->m_CmdFrame), &(src.m_CmdFrame), sizeof(CMDFRAME) );
	this->m_CurState = src.m_CurState;
	return *this;
}

// 拷贝构造
CComFrame::CComFrame( const CComFrame& src )
{
	*this = src;
}

// 析构函数
CComFrame::~CComFrame()
{
}

// 检验命令包是否有效
ERRORTYPE CComFrame::CheckValid()
{
	// 命令包起始字节错误
	if ( m_CmdFrame.BOC != VALID_BOC )
	{
		return ET_BOC_ERR;
	}

	// 长度错误
	if ( m_CmdFrame.SIZE > MAX_CMD_SIZE )
	{
		return ET_SIZE_ERR;
	}

	// 校验和错误
	unsigned char chk = 0;
	for ( unsigned i = 0; i < m_CmdFrame.SIZE; i++ )
	{
		chk ^= m_CmdFrame.CMD[i];
	}
	if ( chk != m_CmdFrame.CHK )
	{
		return ET_CHECK_ERR;
	}

	// 命令包结束字节错误
	if ( m_CmdFrame.EOC != VALID_EOC )
	{
		return ET_EOC_ERR;
	}

	// 检查格式
	//switch ( m_CmdFrame.CMD[0] )
 //   {
 //   case COMMAND_EDGE_MODULE_CONTROL:              // 边缘锐化模块开关
 //   case COMMAND_PE_MODULE_CONTROL:                // 图像增强模块开关
 //   case COMMAND_MID_MODULE_CONTROL:               // 中值滤波模块开关
 //   case COMMAND_ST_BADPOINT_MODULE_CONTROL:       // 静态盲元校正模块开关
 //   //case COMMAND_DY_BADPOINT_MODULE_CONTROL:       // 动态盲元校正模块开关
 //   case COMMAND_TWOPOINT_MODULE_CONTROL:          // 两点校正模块开关
	//case COMMAND_TEMPERATURE_DISPLAY:              // 温度显示开关
 //   //case COMMAND_AD_SWITCH:                        // 片上AD开关
 //   case COMMAND_POLAR_CONTROL:                    // 极性控制
 //   case COMMAND_FRAME_FREQUENCY_CONTROL:          // 帧率控制
 //   //case COMMAND_SCANNING_DIRECTION_CONTROL:       // 扫描方向控制
 //   case COMMAND_ANALOG_SIGNAL_GAIN_CONTROL:       // 模拟信号增益控制
 //   //case COMMAND_BENDI_COLLECTION:                 // 本底采集
 //   //case COMMAND_SET_MAX5625_VOLTAGE:              // 设置5625芯片电压
 //   //case COMMAND_SET_RESET2INT:                    // 设置reset到INT下降沿的延时
 //   //case COMMAND_SET_RESET2DISPLAYFLAG:            // 设置reset到displayflag的行数
 //   //case COMMAND_PHASE8_CONTROL:                 // 8相位控制
 //   case COMMAND_SINGLE_POINT_CONTROL:             // 单点校正
	////case COMMAND_UPLOAD_BENDI:                     // 上传本底
 //   //case COMMAND_BP_GATE_CONTROL:                  // 盲元门限设置
 //   //case COMMAND_BP_SIGMA_CONTROL:                 // 盲元sigma值设置
	////case COMMAND_BP_READ_RAM_CONTROL:              // 读取动态盲元ram
	//case COMMAND_QUERY_SETTINGS:                   // 请求系统参数
	////case COMMAND_RESET_SETTINGS:                   // 重置系统参数
	//case COMMAND_NUC_CONTROL:                      // 进行内部非均匀校正  命令格式：0900：采集低温本底；0901：采集高温本底，并进行非均匀校正GO参数的计算和实时更新校正表
 //   case COMMAND_MOTOR_STEP_FORWARD:               // 步进电机向前步进
 //   case COMMAND_MOTOR_STEP_BACKWARD:              // 步进电机向后步进
	//case COMMAND_MOTOR_ROUTE_FORWARD:              // 直流电机正转
	//case COMMAND_MOTOR_ROUTE_BACKWARD:             // 直流电机反转
	//	// 载荷长度必须是2字节
	//	if ( m_CmdFrame.SIZE != 2 )
	//	{
	//		return ET_SIZE_ERR;
	//	}
 //       break;
	//case COMMAND_INTEGRATION_TIME_CONTROL:         // 积分时间控制
	//case COMMAND_SET_PCONTROL:                     // 平台直方图，平台值设定
	//case COMMAND_ZQ_DUIBI_CONTROL:                 // 亮度-对比度控制
	//case COMMAND_GPOL_CONTROL:                     // 设置Gpol值
	//case COMMAND_SERVO_ROUTE:                      // 舵机转动到指定角度
	//	// 载荷长度必须是5字节
	//	if ( m_CmdFrame.SIZE != 5 )
	//	{
	//		return ET_SIZE_ERR;
	//	}
	//	break;
	//case COMMAND_UPDATE_BADPOINT_TABLE:            // 更新盲元表
	//case COMMAND_UPDATE_TWOPOINT_TABLE:            // 更新校正表
	////case COMMAND_UPDATE_PESUDOCOLOR_TABLE:         // 更新伪彩表
	//	// 不检查参数长度
	//	break;
 //   default:
	//	// 未定义命令
	//	return ET_UNKOWN_CMD;
 //       break;
 //   }

	// 没有错误
	return ET_OK;
}

// 当前命令包是否有效
bool CComFrame::IsValid() const
{
	return m_CurState == ET_OK;
}

// bool 操作
CComFrame::operator bool () const
{
	return m_CurState == ET_OK;
}

// 获取错误码
ERRORTYPE CComFrame::GetErrorType() const
{
	return m_CurState;
}

// 转化为字节块
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

// 从字节块赋值
ERRORTYPE CComFrame::Unserialize( const unsigned char* buff, int len )
{
	// 初始状态为没有错误
	m_CurState = ET_OK;

	// 命令过短
	if ( len < 7 )
	{
		m_CurState = ET_SIZE_ERR;
		return m_CurState;
	}

	// 拷贝帧头
	m_CmdFrame.BOC  = buff[0];
	// 拷贝连续帧编号
	m_CmdFrame.ADDR = buff[1];
	// 读取载荷长度
	m_CmdFrame.SIZE = buff[2];
	m_CmdFrame.SIZE <<= 8;
	m_CmdFrame.SIZE |= buff[3];
	m_CmdFrame.SIZE <<= 8;
	m_CmdFrame.SIZE |= buff[4];


	// 命令长度错误
	if ( len != m_CmdFrame.SIZE + 7 )
	{
		m_CurState = ET_SIZE_ERR;
		return m_CurState;
	}

	// 拷贝载荷
	for ( unsigned i = 0; i < m_CmdFrame.SIZE; i++ )
	{
		m_CmdFrame.CMD[i] = buff[i+5];
	}
	// 拷贝载荷校验
	m_CmdFrame.CHK = buff[m_CmdFrame.SIZE+5];
	// 拷贝帧尾
	m_CmdFrame.EOC = buff[m_CmdFrame.SIZE+6];

	// 检验命令包有效性
	m_CurState |= CheckValid();
	return m_CurState;
}

// 检查ACK
ERRORTYPE CComFrame::CheckACK( CComFrame& ack ) const
{
	// 调用时如果本身的格式不正确，将会引起断言失败
	assert( m_CurState == ET_OK );

	// ACK格式不正确应该在调用此函数前处理
	if (!ack)
		return 16; //**********
	assert ( ack );

	// 如果命令不匹配，则返回匹配错误
	if ( ack.GetCmd() != GetCmd() )
	{
		return ET_CMD_MISSMATCH;
	}

	// 读取ACK载荷
	const unsigned char* buff = ack.GetParam();
	ERRORTYPE et = buff[0] * 0x10000
		+ buff[1] * 0x100
		+ buff[2];

	return et;
}

// 获取命令字（调用前请先检查命令包有效性）
unsigned char CComFrame::GetCmd() const
{
	return m_CmdFrame.CMD[0];
}

// 获取参数长度（调用前请先检查命令包有效性）
int CComFrame::GetParamLen() const
{
	return m_CmdFrame.SIZE-1;
}

// 获取参数（调用前请先检查命令包有效性，并配合GetParamLen使用）
const unsigned char* CComFrame::GetParam() const
{
	return m_CmdFrame.CMD+1;
}

