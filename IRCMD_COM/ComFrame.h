#ifndef COMFRAME_H_
#define COMFRAME_H_


/*-------------------------------*/
/*        最大命令包载荷长度      */
/*-------------------------------*/
//#define MAX_CMD_SIZE   512
//#define MAX_FRAME_LEN  (1+1+3+512+1+1)
#define MAX_CMD_SIZE   200
#define MAX_FRAME_LEN  (1+1+3+200+1+1)

/*-------------------------------*/
/*  上位机指令集（下位机状态集）   */
/*-------------------------------*/
#define IDLE                                       0x00         // 没有指令
#define COMMAND_EDGE_MODULE_CONTROL                0x30         // 边缘锐化模块开关
#define COMMAND_PE_MODULE_CONTROL                  0xd0         // 图像增强模块开关
#define COMMAND_MID_MODULE_CONTROL                 0xc0         // 中值滤波模块开关
#define COMMAND_ST_BADPOINT_MODULE_CONTROL         0xf0         // 静态盲元校正模块开关
#define COMMAND_DY_BADPOINT_MODULE_CONTROL         0x10         // 动态盲元校正模块开关
#define COMMAND_TWOPOINT_MODULE_CONTROL            0xe0         // 两点校正模块开关
#define COMMAND_TEMPERATURE_DISPLAY                0x50         // 查询温度
//#define COMMAND_AD_SWITCH                          0xad         // 片上AD开关
#define COMMAND_POLAR_CONTROL                      0xb0         // 极性控制
#define COMMAND_FRAME_FREQUENCY_CONTROL            0x12         // 帧率控制
//#define COMMAND_SCANNING_DIRECTION_CONTROL         0x90         // 扫描方向控制
#define COMMAND_INTEGRATION_TIME_CONTROL           0x80         // 积分时间控制
#define COMMAND_INTEGRATION_TIME2_CONTROL          0x81         // 第二积分时间控制
#define COMMAND_ANALOG_SIGNAL_GAIN_CONTROL         0x70         // 模拟信号增益控制
#define COMMAND_UPDATE_BADPOINT_TABLE              0x33         // 更新盲元表
#define COMMAND_UPDATE_TWOPOINT_TABLE              0x66         // 更新校正表

#define COMMAND_WINDOW_MODE_SWITCH				   0x22         // 开窗坐标设定
#define COMMAND_GAIN_MODE_SWITCH                   0x23         // 增益
#define COMMAND_SYSCHRONIZE_MODE_SWITCH            0x24         // 内外同步模式选择
#define COMMAND_RUN_MODE_SWITCH                    0x25         // 运行
#define COMMAND_RESET_CONTROL                      0x26         // 复位逻辑模块

//#define COMMAND_UPDATE_PESUDOCOLOR_TABLE           0x77         // 更新伪彩表
//#define COMMAND_BENDI_COLLECTION				   0x88         // 本底采集
//#define COMMAND_SET_MAX5625_VOLTAGE				   0x99	        // 设置5625芯片电压
#define COMMAND_SET_RESET2INT					   0xaa	        // 设置reset到INT下降沿的延时
#define COMMAND_SET_RESET2DISPLAYFLAG			   0xab	        // 设置reset到displayflag的行数
#define COMMAND_SET_PCONTROL			   		   0xac	        // 平台直方图，平台值设定
#define COMMAND_PHASE8_CONTROL                     0x11         // 8相位控制
#define COMMAND_SINGLE_POINT_CONTROL               0x01         // 单点校正
#define COMMAND_ZQ_DUIBI_CONTROL                   0xae         // 亮度-对比度控制
#define COMMAND_UPLOAD_BENDI					   0x0e         // 上传本底
#define COMMAND_BP_GATE_CONTROL                    0x03         // 盲元门限设置
#define COMMAND_BP_SIGMA_CONTROL                   0x04         // 盲元sigma值设置
#define COMMAND_BP_READ_RAM_CONTROL                0x05         // 读取动态盲元ram
#define COMMAND_GPOL_CONTROL                       0x06         // 设置Gpol值
#define COMMAND_QUERY_SETTINGS                     0x07         // 请求系统参数
#define COMMAND_RESET_SETTINGS                     0x08         // 重置系统参数
#define COMMAND_NUC_CONTROL                        0x09         // 进行内部非均匀校正  命令格式：0900：采集低温本底；0901：采集高温本底，0902:更新校正表 并进行非均匀校正GO参数的计算和实时更新校正表
#define COMMAND_MOTOR_STEP_FORWARD                 0x13         // 步进电机向前步进
#define COMMAND_MOTOR_STEP_BACKWARD                0x15         // 步进电机向后步进
#define COMMAND_MOTOR_ROUTE_FORWARD                0x14         // 直流电机正转
#define COMMAND_MOTOR_ROUTE_BACKWARD               0x17         // 直流电机反转
#define COMMAND_SERVO_ROUTE                        0x16         // 舵机转动到指定角度
#define COMMAND_DATA_MERGE_CONTROL                 0xdd         // 数据融合开关

/*-------------------------------*/
/*            错误码             */
/*-------------------------------*/
typedef unsigned int ERRORTYPE;
// 错误码只有3字节，所以最高8位将被忽略
#define	ET_OK                                      0x000000000  // 命令被正确执行
#define	ET_PARAM_ERR                               0x000000001  // 命令参数错误
#define	ET_EXECUSE_ERR                             0x000000002  // 命令执行出错
#define	ET_UNKNOWN_CMD                             0x000000004  // 未知命令
#define	ET_BOC_ERR                                 0x000000008  // 命令包起始字节错误
#define	ET_TIMEOUT                                 0x000000010  // 数据传输超时
#define ET_CMD_MISSMATCH                           0x000000020  // 返回的命令字不匹配
#define	ET_CHECK_ERR                               0x000000040  // 校验和错误
#define	ET_SIZE_ERR                                0x000000080  // 长度错误
#define	ET_EOC_ERR                                 0x000000100  // 命令包结束字节错误


/*-------------------------------*/
/*            BOC-EOC            */
/*-------------------------------*/
#define VALID_BOC 0x06      // 有效的命令包起始
#define VALID_EOC 0x08      // 有效的命令包结束

// 命令包类
class CComFrame
{
// 命令帧格式
typedef struct tag_CMDFRAME {
	unsigned char BOC;               // 命令帧起始
	unsigned char ADDR;              // 连续帧编号
	unsigned SIZE;          // 载荷长度
	unsigned char CMD[MAX_CMD_SIZE]; // 载荷
	unsigned char CHK;               // 载荷校验
	unsigned char EOC;               // 命令帧结束

	tag_CMDFRAME()          // 默认构造为一个有效无用命令帧
		: BOC( VALID_BOC )
		, ADDR( 0 )
		, SIZE( 1 )
		, CHK( IDLE )
		, EOC( VALID_EOC )
	{
		CMD[0] = IDLE;
	};
}CMDFRAME;

public:
	// 默认构造函数
	CComFrame();
	// 从字节块构造
	CComFrame( const unsigned char* buff, int len );
	// 从命令字和参数构造
	CComFrame( unsigned char cmd, const unsigned char* param, const int len = 1, const unsigned char addr = 0 );
	// 赋值操作
	CComFrame& operator = (const CComFrame& src);
	// 拷贝构造
	CComFrame( const CComFrame& src );
	// 析构函数
	~CComFrame();

private:
	CMDFRAME m_CmdFrame;     // 命令帧存储区
	ERRORTYPE m_CurState;    // 命令帧当前状态

protected:
	// 检验命令包是否有效
	ERRORTYPE CheckValid();

public:
	// 当前命令包是否有效
	bool IsValid() const;
	// bool 操作
	operator bool () const;
	// 获取错误码
	ERRORTYPE GetErrorType() const;
	// 转化为字节块
	ERRORTYPE Serialize( unsigned char * buff, int &len );
	// 从字节块赋值
	ERRORTYPE Unserialize( const unsigned char* buff, int len );

	// 检查ACK
	ERRORTYPE CheckACK( CComFrame& ack ) const;
	// 获取命令字（调用前请先检查命令包有效性）
	unsigned char GetCmd() const;
	// 获取参数长度（调用前请先检查命令包有效性）
	int GetParamLen() const;
	// 获取参数（调用前请先检查命令包有效性，并配合GetParamLen使用）
	const unsigned char* GetParam() const;
};

#endif // COMFRAME_H_
