/* Copyright (c) 2026 清影/123twtd */
/* Copyright (c) 2026 高志禹 */
/*
 * PID.h
 *
 *  Created on: 2026年5月3日
 *      Author: A
 */



#ifndef _PID_H_
#define _PID_H_

#include "zf_common_headfile.h"

// 定义 PID 结构体，把参数和中间变量打包在一起
typedef struct
{
    float kp;              // 比例系数
    float ki;              // 积分系数
    float kd;              // 微分系数

    int target;            // 目标值
    int current;           // 当前实际值
    int error;             // 当前误差
    int last_error;        // 上次误差
    int prev_error;        // 上上次误差 (增量式 PID 专用)

    float output;          // 最终输出值
    float i_sum;           // 积分累加值 (位置式 PID 专用)

    int out_max;           // 输出上限（限幅）
    int out_min;           // 输出下限
} pid_struct;

// 函数声明
void PID_Init(pid_struct *pid, float p, float i, float d, int max);
int Position_PID_Realize(pid_struct *pid, int current, int target);              //位置式
int Incremental_PID_Realize(pid_struct *pid, int current, int target);           //增量式

#endif

