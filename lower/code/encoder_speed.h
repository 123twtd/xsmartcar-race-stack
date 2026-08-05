/* Copyright (c) 2026 清影/123twtd */
/* Copyright (c) 2026 高志禹 */
/*
 * encoder_speed.h
 *
 *  Created on: 2026年6月5日
 *      Author: A
 */

#ifndef CODE_ENCODER_SPEED_H_
#define CODE_ENCODER_SPEED_H_

#include "zf_common_headfile.h"

// ======================= 核心机械参数配置 =======================
#define ENCODER_SAMPLE_MS   10U     // 采样周期 10ms
#define WHEEL_RADIUS_CM     3.0f    // 轮胎半径 3cm

#define ENCODER_DIR_SIGN    1       // 方向符号 (如果发现往前推车，速度是负的，就把这改成 -1)
#define ENC_DELTA_DEADZONE  0       // 死区 (静止时允许的脉冲抖动，设0即可)
#define ENC_DELTA_MAX_TICK  10000   // 单次最大允许脉冲数 (防爆冲溢出)
// ================================================================

void encoder_speed_init(encoder_index_enum ch, encoder_channel1_enum count_pin, encoder_channel2_enum dir_pin, uint16 lines_per_rev);
void encoder_speed_service(void);
float encoder_speed_get_rpm(void);
float encoder_speed_get_cm_s(void);

#endif
