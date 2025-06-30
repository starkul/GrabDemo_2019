#ifndef COMFRAME_H_
#define COMFRAME_H_


/*-------------------------------*/
/*        ���������غɳ���      */
/*-------------------------------*/
//#define MAX_CMD_SIZE   512
//#define MAX_FRAME_LEN  (1+1+3+512+1+1)
#define MAX_CMD_SIZE   200
#define MAX_FRAME_LEN  (1+1+3+200+1+1)

/*-------------------------------*/
/*  ��λ��ָ�����λ��״̬����   */
/*-------------------------------*/
#define IDLE                                       0x00         // û��ָ��
#define COMMAND_EDGE_MODULE_CONTROL                0x30         // ��Ե��ģ�鿪��
#define COMMAND_PE_MODULE_CONTROL                  0xd0         // ͼ����ǿģ�鿪��
#define COMMAND_MID_MODULE_CONTROL                 0xc0         // ��ֵ�˲�ģ�鿪��
#define COMMAND_ST_BADPOINT_MODULE_CONTROL         0xf0         // ��̬äԪУ��ģ�鿪��
#define COMMAND_DY_BADPOINT_MODULE_CONTROL         0x10         // ��̬äԪУ��ģ�鿪��
#define COMMAND_TWOPOINT_MODULE_CONTROL            0xe0         // ����У��ģ�鿪��
#define COMMAND_TEMPERATURE_DISPLAY                0x50         // ��ѯ�¶�
//#define COMMAND_AD_SWITCH                          0xad         // Ƭ��AD����
#define COMMAND_POLAR_CONTROL                      0xb0         // ���Կ���
#define COMMAND_FRAME_FREQUENCY_CONTROL            0x12         // ֡�ʿ���
//#define COMMAND_SCANNING_DIRECTION_CONTROL         0x90         // ɨ�跽�����
#define COMMAND_INTEGRATION_TIME_CONTROL           0x80         // ����ʱ�����
#define COMMAND_INTEGRATION_TIME2_CONTROL          0x81         // �ڶ�����ʱ�����
#define COMMAND_ANALOG_SIGNAL_GAIN_CONTROL         0x70         // ģ���ź��������
#define COMMAND_UPDATE_BADPOINT_TABLE              0x33         // ����äԪ��
#define COMMAND_UPDATE_TWOPOINT_TABLE              0x66         // ����У����

#define COMMAND_WINDOW_MODE_SWITCH				   0x22         // ���������趨
#define COMMAND_GAIN_MODE_SWITCH                   0x23         // ����
#define COMMAND_SYSCHRONIZE_MODE_SWITCH            0x24         // ����ͬ��ģʽѡ��
#define COMMAND_RUN_MODE_SWITCH                    0x25         // ����
#define COMMAND_RESET_CONTROL                      0x26         // ��λ�߼�ģ��

//#define COMMAND_UPDATE_PESUDOCOLOR_TABLE           0x77         // ����α�ʱ�
//#define COMMAND_BENDI_COLLECTION				   0x88         // ���ײɼ�
//#define COMMAND_SET_MAX5625_VOLTAGE				   0x99	        // ����5625оƬ��ѹ
#define COMMAND_SET_RESET2INT					   0xaa	        // ����reset��INT�½��ص���ʱ
#define COMMAND_SET_RESET2DISPLAYFLAG			   0xab	        // ����reset��displayflag������
#define COMMAND_SET_PCONTROL			   		   0xac	        // ƽֱ̨��ͼ��ƽֵ̨�趨
#define COMMAND_PHASE8_CONTROL                     0x11         // 8��λ����
#define COMMAND_SINGLE_POINT_CONTROL               0x01         // ����У��
#define COMMAND_ZQ_DUIBI_CONTROL                   0xae         // ����-�Աȶȿ���
#define COMMAND_UPLOAD_BENDI					   0x0e         // �ϴ�����
#define COMMAND_BP_GATE_CONTROL                    0x03         // äԪ��������
#define COMMAND_BP_SIGMA_CONTROL                   0x04         // äԪsigmaֵ����
#define COMMAND_BP_READ_RAM_CONTROL                0x05         // ��ȡ��̬äԪram
#define COMMAND_GPOL_CONTROL                       0x06         // ����Gpolֵ
#define COMMAND_QUERY_SETTINGS                     0x07         // ����ϵͳ����
#define COMMAND_RESET_SETTINGS                     0x08         // ����ϵͳ����
#define COMMAND_NUC_CONTROL                        0x09         // �����ڲ��Ǿ���У��  �����ʽ��0900���ɼ����±��ף�0901���ɼ����±��ף�0902:����У���� �����зǾ���У��GO�����ļ����ʵʱ����У����
#define COMMAND_MOTOR_STEP_FORWARD                 0x13         // ���������ǰ����
#define COMMAND_MOTOR_STEP_BACKWARD                0x15         // ���������󲽽�
#define COMMAND_MOTOR_ROUTE_FORWARD                0x14         // ֱ�������ת
#define COMMAND_MOTOR_ROUTE_BACKWARD               0x17         // ֱ�������ת
#define COMMAND_SERVO_ROUTE                        0x16         // ���ת����ָ���Ƕ�
#define COMMAND_DATA_MERGE_CONTROL                 0xdd         // �����ںϿ���

/*-------------------------------*/
/*            ������             */
/*-------------------------------*/
typedef unsigned int ERRORTYPE;
// ������ֻ��3�ֽڣ��������8λ��������
#define	ET_OK                                      0x000000000  // �����ȷִ��
#define	ET_PARAM_ERR                               0x000000001  // �����������
#define	ET_EXECUSE_ERR                             0x000000002  // ����ִ�г���
#define	ET_UNKNOWN_CMD                             0x000000004  // δ֪����
#define	ET_BOC_ERR                                 0x000000008  // �������ʼ�ֽڴ���
#define	ET_TIMEOUT                                 0x000000010  // ���ݴ��䳬ʱ
#define ET_CMD_MISSMATCH                           0x000000020  // ���ص������ֲ�ƥ��
#define	ET_CHECK_ERR                               0x000000040  // У��ʹ���
#define	ET_SIZE_ERR                                0x000000080  // ���ȴ���
#define	ET_EOC_ERR                                 0x000000100  // ����������ֽڴ���


/*-------------------------------*/
/*            BOC-EOC            */
/*-------------------------------*/
#define VALID_BOC 0x06      // ��Ч���������ʼ
#define VALID_EOC 0x08      // ��Ч�����������

// �������
class CComFrame
{
// ����֡��ʽ
typedef struct tag_CMDFRAME {
	unsigned char BOC;               // ����֡��ʼ
	unsigned char ADDR;              // ����֡���
	unsigned SIZE;          // �غɳ���
	unsigned char CMD[MAX_CMD_SIZE]; // �غ�
	unsigned char CHK;               // �غ�У��
	unsigned char EOC;               // ����֡����

	tag_CMDFRAME()          // Ĭ�Ϲ���Ϊһ����Ч��������֡
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
	// Ĭ�Ϲ��캯��
	CComFrame();
	// ���ֽڿ鹹��
	CComFrame( const unsigned char* buff, int len );
	// �������ֺͲ�������
	CComFrame( unsigned char cmd, const unsigned char* param, const int len = 1, const unsigned char addr = 0 );
	// ��ֵ����
	CComFrame& operator = (const CComFrame& src);
	// ��������
	CComFrame( const CComFrame& src );
	// ��������
	~CComFrame();

private:
	CMDFRAME m_CmdFrame;     // ����֡�洢��
	ERRORTYPE m_CurState;    // ����֡��ǰ״̬

protected:
	// ����������Ƿ���Ч
	ERRORTYPE CheckValid();

public:
	// ��ǰ������Ƿ���Ч
	bool IsValid() const;
	// bool ����
	operator bool () const;
	// ��ȡ������
	ERRORTYPE GetErrorType() const;
	// ת��Ϊ�ֽڿ�
	ERRORTYPE Serialize( unsigned char * buff, int &len );
	// ���ֽڿ鸳ֵ
	ERRORTYPE Unserialize( const unsigned char* buff, int len );

	// ���ACK
	ERRORTYPE CheckACK( CComFrame& ack ) const;
	// ��ȡ�����֣�����ǰ���ȼ���������Ч�ԣ�
	unsigned char GetCmd() const;
	// ��ȡ�������ȣ�����ǰ���ȼ���������Ч�ԣ�
	int GetParamLen() const;
	// ��ȡ����������ǰ���ȼ���������Ч�ԣ������GetParamLenʹ�ã�
	const unsigned char* GetParam() const;
};

#endif // COMFRAME_H_
