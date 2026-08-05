/* Copyright (c) 2026 清影/123twtd */
/* Copyright (c) 2026 高志禹 */
/*
 * 限幅逻辑由清影/123twtd提出并审查，当前比赛版本的具体实现由高志禹完成。
 */

/*
 * PID.c
 *
 *  Created on: 2026年5月3日
 *      Author: A
 */


#include "pid.h"

// PID 初始化：给参数赋初值
void PID_Init(pid_struct *pid, float p, float i, float d, int max)
{
    pid->kp = p;
    pid->ki = i;
    pid->kd = d;
    pid->target = 0;
    pid->current = 0;
    pid->error = 0;
    pid->last_error = 0;
    pid->prev_error = 0;
    pid->output = 0;
    pid->i_sum = 0;
    pid->out_max = max;
    pid->out_min = -max;
}

// 位置式 PID
int Position_PID_Realize(pid_struct *pid, int current, int target)
{
    pid->error = target - current;
    pid->i_sum += pid->ki * pid->error;  // 积分累加
    if (pid->i_sum > 6.0f) pid->i_sum = 6.0f;
    if (pid->i_sum < -6.0f) pid->i_sum = -6.0f;

    // 计算输出
    pid->output = pid->kp * pid->error +
                  pid->i_sum +
                  pid->kd * (pid->error - pid->last_error);

    pid->last_error = pid->error;

    // 限幅
    if(pid->output > pid->out_max) pid->output = pid->out_max;
    if(pid->output < pid->out_min) pid->output = pid->out_min;

    return (int)pid->output;
}

// 增量式 PID
int Incremental_PID_Realize(pid_struct *pid, int current, int target)
{

    //  滚存误差
    pid->prev_error = pid->last_error;
    pid->last_error = pid->error;
    pid->error = target - current;

//    // 如果 Kp 和 Kd 都是 0（纯 I 控制），直接走纯积分路线，杜绝历史脏数据干扰！
//    float delta = 0.0f;
//    if (pid->kp == 0.0f && pid->kd == 0.0f)
//    {
//        delta = pid->ki * pid->error;
//    }
//    else
//    {
//        // 正常增量公式
//        delta = pid->kp * (pid->error - pid->last_error) +
//                pid->ki * pid->error +
//                pid->kd * (pid->error - 2 * pid->last_error + pid->prev_error);
//    }
//
    pid->output +=  pid->kp * (pid->error - pid->last_error) +
                    pid->ki * pid->error +
                    pid->kd * (pid->error - 2 * pid->last_error + pid->prev_error);


    // 输出强力限幅
    if(pid->output > pid->out_max) pid->output = pid->out_max;
    if(pid->output < pid->out_min) pid->output = pid->out_min;

    // 静态死区保护
    if (target == 0 && current < 5 && current > -5) {
        pid->output = 0;
        pid->i_sum = 0; // 清空
        return 0;
    }

    return (int)pid->output;
}
