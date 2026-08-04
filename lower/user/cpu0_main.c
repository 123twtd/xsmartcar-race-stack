/* Copyright (c) 2026 清影/123twtd */
#include "zf_common_headfile.h"
#include "encoder_speed.h"
#include "pid.h"
#include "protocol.h"
#include "buzzer.h"
#include "key.h"
#include "battery.h"
#include <stdio.h>
#include <string.h>

#include "isr_config.h"

#pragma section all "cpu0_dsram"

// ======================= 引脚参数宏定义 =======================
#define UART_INDEX              (UART_0)
#define UART_BAUDRATE           (115200)//115200
#define UART_TX_PIN             (UART0_TX_P14_0)
#define UART_RX_PIN             (UART0_RX_P14_1)

#define ENC1_MODULE             (TIM5_ENCODER)
#define ENCODER_COUNT_PIN       (TIM5_ENCODER_CH1_P21_7)
#define ENCODER_DIR_PIN         (TIM5_ENCODER_CH2_P21_6)
#define ENCODER_LINES_PER_REV   (500*2)// 二倍频

#define MOTOR_PWM_PIN           (ATOM2_CH1_P11_2)//(ATOM2_CH1_P11_2)(ATOM0_CH4_P02_4)
#define MOTOR_DIR_PIN           (P11_3)//(ATOM2_CH2_P11_3)(P02_5)

#define SERVO_PIN               (ATOM1_CH1_P33_9)
#define SERVO_CENTER            (675)
#define SERVO_MAX_OFFSET        (150)

// ======================= 声明与定义全局核心参数 =======================
volatile float kp1 = KP1_DEFAULT;
volatile float kp2 = KP2_DEFAULT;
volatile float kp3 = KP3_DEFAULT;
volatile float servo_ki = SERVO_KI_DEFAULT;
volatile float servo_kd = SERVO_KD_DEFAULT;

// ======================= 声明外部协议变量 =======================
extern float near_offset;
extern float near_angle;
extern float det_offline;
extern int start_flag;
extern int speed_limit;

// ======================= 全局变量区 =======================
volatile uint8 rx_data = 0;
volatile uint8 rx_flag = 0;

pid_struct motor_pid; // 电机 PID (增量式)
pid_struct servo_pid; // 舵机 PID (位置式)

// 提取为全局变量，供中断与主循环共享
volatile int current_spd = 0;
volatile int target_servo_offset = 0;

// ======================= 底层驱动函数 =======================
void Motor_ctrl(int power) {
    if(power > 100) power = 100;
    if(power < -100) power = -100;

    if(power >= 0) {
        gpio_set_level(MOTOR_DIR_PIN, 1);
        pwm_set_duty(MOTOR_PWM_PIN, power * (PWM_DUTY_MAX / 100));
    } else {
        gpio_set_level(MOTOR_DIR_PIN, 0);
        pwm_set_duty(MOTOR_PWM_PIN, -power * (PWM_DUTY_MAX / 100));
    }
}

void Servo_ctrl(int offset) {
    pwm_set_duty(SERVO_PIN, SERVO_CENTER - offset);// 正左负右
}

// ======================= 主函数 =======================
int core0_main(void)
{
    clock_init();
    debug_init();

    // Init peripherals
    uart_init(UART_INDEX, UART_BAUDRATE, UART_TX_PIN, UART_RX_PIN);
    encoder_speed_init(ENC1_MODULE, ENCODER_COUNT_PIN, ENCODER_DIR_PIN, ENCODER_LINES_PER_REV);
    gpio_init(MOTOR_DIR_PIN, GPO, GPIO_LOW, GPO_PUSH_PULL);
    pwm_init(MOTOR_PWM_PIN, 17000, 0);
    pwm_init(SERVO_PIN, 50, SERVO_CENTER);
    Buzzer_Init();
    Key_Init();
    Battery_Init();

    // 初始化 PIT 定时器 CCU60_CH0，周期 10 毫秒
    pit_ms_init(CCU60_CH0, 10);

    PID_Init(&motor_pid, 0.8f, 0.04f, 0.0f, 100);
    PID_Init(&servo_pid, 1.0f, servo_ki, servo_kd, SERVO_MAX_OFFSET);

    // 开启全局中断，并同步双核（CPU1 在此处等待）
    cpu_wait_event_ready();

    uart_write_string(UART_INDEX, ">>> System Online!\r\n");

    Beep_Play(BEEP_LONG);

    uint32 print_time_cnt = 0;
    static int last_start_flag = 0;
    static int low_voltage_alarm_flag = 0;

    while (TRUE)
    {
        Key_Service();
        StartFlag_Guard();

        // ---------------------------------------------------
        // 发车/停车边沿检测
        if (start_flag == 1 && last_start_flag == 0) {
            Beep_Play(BEEP_DOUBLE);
        }
        else if (start_flag == 0 && last_start_flag == 1) {
            Beep_Play(BEEP_LONG);
            speed_limit = 0;
            near_offset = 0;
            near_angle  = 0;
            det_offline = 0;
        }
        last_start_flag = start_flag;

        // ---------------------------------------------------
        // 串口轮询接收处理
        // ---------------------------------------------------
        uint8 data = 0;
        while (uart_query_byte(UART_INDEX, &data) == 1) {
            uart_write_byte(UART_INDEX, data);
            if (data == '\r\n' || data == '\r'|| data == '\n') {
                uart_rx_buffer[uart_rx_index] = '\0';
                if(uart_rx_index > 0) {
                    // 安全门控：只有本地发车按键已置位后才接收上位机命令。
                    // C,0,0 完赛停车清除锁存后，必须再次按本地按键，串口不能远程发车。
                    if (start_flag == 1) {
                        Parse_Upper_Computer_Data(uart_rx_buffer);
                    }
                }
                uart_rx_index = 0;
            } else {
                if (uart_rx_index < 60) {
                    uart_rx_buffer[uart_rx_index++] = data;
                }
            }
        }

        // ---------------------------------------------------
        // 串口打印 (200ms) - TFT 打印已移交 CPU1
        // ---------------------------------------------------
        print_time_cnt++;
        if (print_time_cnt >= 20)
        {
            print_time_cnt = 0;
            char debug_buf[64];

            // 发车状态下，target_spd 等于 speed_limit
            int disp_target_spd = (start_flag == 1) ? speed_limit : 0;

            // 获取电池电压
            float bat_v = Battery_GetVoltage();
            if (bat_v < BATTERY_LOW_VOLTAGE)
            {
                if (low_voltage_alarm_flag == 0)
                {
                    Beep_Play(BEEP_LOW_VOLT);
                    low_voltage_alarm_flag = 1;
                }
            }
            else
            {
                low_voltage_alarm_flag = 0;
            }

            // 将电压格式化加入打印输出，例如：目标速度、当前速度、电机输出、8.25V
            sprintf(debug_buf, "%d,%d,%.2f,%.2fV\r\n",
                    disp_target_spd, current_spd, motor_pid.output, bat_v);
            uart_write_string(UART_INDEX, debug_buf);
        }

        // Buzzer service
        Buzzer_Service();

        system_delay_ms(10);
    }
}

// =========================================================================
// ============================= 中断服务函数 ==============================
// =========================================================================

// ================= 串口 0 接收中断 =================
IFX_INTERRUPT(uart0_rx_isr, 0, UART0_RX_INT_PRIO)
{
    uint8 get_data = 0;
    if(uart_query_byte(UART_INDEX, &get_data))
    {
        rx_data = get_data;
        rx_flag = 1;
    }
}

// ================= PIT 定时器中断 (10ms 周期) =================
IFX_INTERRUPT(cc60_pit_ch0_isr, 0, CCU6_0_CH0_ISR_PRIORITY)
{
    StartFlag_Guard();

    // ==========================================
    // 数据采集
    // ==========================================
    encoder_speed_service();
    float current_spd_float = -encoder_speed_get_cm_s();
    current_spd = (int)current_spd_float;

    // ==========================================
    // 舵机控制计算 (位置式 PID，充当限幅加法器)
    // ==========================================
    if (start_flag == 1)
    {
        // 计算基础巡线误差 (引入 kp1 与 kp2)
        float base_error = (kp1 * near_offset) +
                           (kp2 * near_angle);

        // 计算高级任务强制偏置 (引入 kp3)
        float adv_error = (kp3 * det_offline);

        // 合并总误差
        float total_error_float = base_error + adv_error;

        // 转换为整数舵机偏移
        int total_error_int = (int)(-total_error_float);

        servo_pid.kp = 1.0f;
        servo_pid.ki = servo_ki;
        servo_pid.kd = servo_kd;

        // 舵机 PID 输出自动框定在 -150 到 150 之间
        target_servo_offset = Position_PID_Realize(&servo_pid, 0, total_error_int);
    }
    else
    {
        target_servo_offset = 0; // 停车时舵机回中
    }
    Servo_ctrl(target_servo_offset);    // 执行底层舵机输出

    // ==========================================
    // 电机控制计算 (增量式 PID)
    // ==========================================
    int target_spd = 0;

    if (start_flag == 0)
    {
        // 停车时清空 PID 数据
        memset(&motor_pid, 0, sizeof(pid_struct));
        PID_Init(&motor_pid, 0.05f, 0.005f, 0.0f, 100);
        motor_pid.output = 0;
    }
    else
    {
        target_spd = speed_limit;
        Incremental_PID_Realize(&motor_pid, current_spd, target_spd);
    }

    Motor_ctrl(motor_pid.output);     // 执行底层电机输出

    pit_clear_flag(CCU60_CH0);    // 清除中断标志位（必须保留最后一步）
}

#pragma section all restore
