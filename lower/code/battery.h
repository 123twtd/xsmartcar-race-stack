/* Copyright (c) 2026 清影/123twtd */
#ifndef _BATTERY_H_
#define _BATTERY_H_

#include "zf_common_headfile.h"

#define BATTERY_ADC_CHANNEL    (ADC0_CH11_A11)

#define BATTERY_CONVERT_COEFF  (117.0f)
#define BATTERY_LOW_VOLTAGE    (10.0f)

void Battery_Init(void);

float Battery_GetVoltage(void);

#endif /* _BATTERY_H_ */
