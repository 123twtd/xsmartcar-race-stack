/* Copyright (c) 2026 清影/123twtd */
/* Copyright (c) 2026 高志禹 */
/*
 * protocol.c
 *
 * Created on: 2026年6月6日
 * Author: 清影/123twtd
 */

#include "protocol.h"
#include <stdio.h>

float near_offset = 0.0f;
float near_angle = 0.0f;
float det_offline = 0.0f;

int start_flag = 0;
int speed_limit = 0;

char uart_rx_buffer[64];
int uart_rx_index = 0;

// =========================================================
// =========================================================
#define near_offset_max 150.0f//30.0f
#define near_angle_max  150.0f//45.0f
#define det_offline_max 150.0f

static volatile int start_flag_latched = 0;
static volatile int start_flag_allow_zero = 0;

// 请求运行并锁存；本地按键授权和已启动状态下的 C,1,... 都会走这条路径。
void StartFlag_RequestStart(void)
{
    start_flag = 1;
    start_flag_latched = 1;
    start_flag_allow_zero = 0;
}

// 清除运行锁存；之后必须重新按本地发车键授权。
void StartFlag_RequestStop(void)
{
    start_flag_allow_zero = 1;
    start_flag_latched = 0;
    start_flag = 0;
}

// 防止普通赋值或异常值绕过发车/停车锁存规则。
void StartFlag_Guard(void)
{
    if (start_flag > 1) {
        start_flag = 1;
        start_flag_latched = 1;
        start_flag_allow_zero = 0;
    }
    else if (start_flag == 1) {
        start_flag_latched = 1;
        start_flag_allow_zero = 0;
    }
    else if (start_flag == 0) {
        if (start_flag_latched && !start_flag_allow_zero) {
            start_flag = 1;
        }
        else {
            start_flag_latched = 0;
            start_flag_allow_zero = 0;
        }
    }
    else {
        start_flag = start_flag_latched ? 1 : 0;
    }
}

// 核心协议解析函数。
void Parse_Upper_Computer_Data(char *str) {
    // L 帧：视觉横向偏差、航向偏差和高级行为附加偏置。
    if (str[0] == 'L') {
        // 当前格式：L,near_offset,near_angle,det_offline
        sscanf(str, "L,%f,%f,%f", &near_offset, &near_angle, &det_offline);

        // 对三个控制量分别限幅。
        if(near_offset > near_offset_max) { near_offset = near_offset_max; }
        else if (near_offset < -near_offset_max) { near_offset = -near_offset_max; }

        if(near_angle > near_angle_max) { near_angle = near_angle_max; }
        else if (near_angle < -near_angle_max) { near_angle = -near_angle_max; }

        if(det_offline > det_offline_max) { det_offline = det_offline_max; }
        else if (det_offline < -det_offline_max) { det_offline = -det_offline_max; }
    }
    // C 帧：运行锁存和速度上限。
    else if (str[0] == 'C') {
        int rx_start_flag = start_flag;
        int rx_speed_limit = speed_limit;

        if (sscanf(str, "C,%d,%d", &rx_start_flag, &rx_speed_limit) == 2) {
            if (rx_start_flag == 0) {
                // C,0,...：完赛/安全停车，清除锁存。
                speed_limit = rx_speed_limit;
                StartFlag_RequestStop();
            }
            else if (rx_start_flag > 0) {
                // C,1,0 是保留锁存的软暂停；C,1,speed 设置运行速度。
                speed_limit = rx_speed_limit;
                StartFlag_RequestStart();
            }
            else {
                StartFlag_Guard();
            }
        }
    }
}
