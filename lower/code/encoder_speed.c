/* Copyright (c) 2026 清影/123twtd */
/* Copyright (c) 2026 高志禹 */
/*
 * encoder_speed.c
 *
 *  Created on: 2026年6月5日
 *      Author: A
 */




#include "encoder_speed.h"
#include "zf_driver_encoder.h"

static encoder_index_enum s_enc_ch;
static uint16 s_lines_per_rev;
static float  s_rpm = 0.0f;
static float  s_rpm_filtered = 0.0f;
static int16  s_last_count = 0;
static int32  s_last_delta = 0;
static uint8  s_encoder_inited = 0U;

#define ENC_LP_ALPHA 0.50f  // 低通滤波系数

static int32 encoder_delta_unwrap(int16 curr, int16 prev)
{
    uint16 curr_u = (uint16)curr;
    uint16 prev_u = (uint16)prev;
    int32  d      = (int32)curr_u - (int32)prev_u;

    if (d > 32767) d -= 65536;
    else if (d < -32768) d += 65536;
    return d;
}

void encoder_speed_init(encoder_index_enum ch, encoder_channel1_enum count_pin, encoder_channel2_enum dir_pin, uint16 lines_per_rev)
{
    encoder_dir_init(ch, count_pin, dir_pin);
    s_enc_ch = ch;
    s_lines_per_rev = lines_per_rev;
    s_rpm = 0.0f;
    s_rpm_filtered = 0.0f;
    s_last_delta = 0;
    encoder_clear_count(ch);
    s_last_count = encoder_get_count(ch);
    s_encoder_inited = 1U;
}

void encoder_speed_service(void)
{
    int16 raw;
    int32 delta32;
    float rpm_inst;

    if (s_encoder_inited == 0U) return;

    raw = encoder_get_count(s_enc_ch);
    delta32 = encoder_delta_unwrap(raw, s_last_count);
    delta32 *= (int32)ENCODER_DIR_SIGN;

    if (delta32 > -(int32)ENC_DELTA_DEADZONE && delta32 < (int32)ENC_DELTA_DEADZONE) delta32 = 0;
    if (delta32 > (int32)ENC_DELTA_MAX_TICK) delta32 = (int32)ENC_DELTA_MAX_TICK;
    else if (delta32 < -(int32)ENC_DELTA_MAX_TICK) delta32 = -(int32)ENC_DELTA_MAX_TICK;

    s_last_count = raw;
    s_last_delta = delta32;

    if (s_lines_per_rev == 0U) return;

    rpm_inst = ((float)delta32 * 60000.0f) / ((float)s_lines_per_rev * (float)ENCODER_SAMPLE_MS);

    s_rpm = rpm_inst;
    s_rpm_filtered = ENC_LP_ALPHA * rpm_inst + (1.0f - ENC_LP_ALPHA) * s_rpm_filtered;
}

float encoder_speed_get_rpm(void) { return s_rpm_filtered; }
float encoder_speed_get_cm_s(void) { return s_rpm_filtered * (2.0f * 3.14159265f * WHEEL_RADIUS_CM) / 60.0f; }
