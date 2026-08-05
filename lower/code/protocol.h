/* Copyright (c) 2026 清影/123twtd */
/* Copyright (c) 2026 高志禹 */
/*
 * protocol.h
 *
 *  Created on: 2026年6月6日
 *      Author: A
 */

#ifndef PROTOCOL_H
#define PROTOCOL_H

// 暴露给主程序的全局变量（加 extern）
extern float near_angle;
extern float far_angle;
extern float near_offset;

extern int start_flag;
extern int speed_limit;

// 串口接收缓存区与索引
extern char uart_rx_buffer[64];
extern int uart_rx_index;

// 解析函数声明
void Parse_Upper_Computer_Data(char *str);
void StartFlag_RequestStart(void);
void StartFlag_RequestStop(void);
void StartFlag_Guard(void);

#endif
