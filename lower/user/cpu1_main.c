/* Copyright (c) 2026 清影/123twtd */
#include "zf_common_headfile.h"
#include "tft_ui.h"
#include "pid.h"
#include "key.h"
#include "battery.h"
#include <stdio.h>
#include <math.h>

int Safe_Float_Split(float val, int *out_int, int *out_dec);

#pragma section all "cpu1_dsram"

extern volatile int current_spd;
extern volatile int target_servo_offset;
extern pid_struct motor_pid;

extern volatile float kp1;
extern volatile float kp2;
extern volatile float kp3;
extern volatile float servo_ki;
extern volatile float servo_kd;

extern int start_flag;
extern int speed_limit;

extern int ui_select_index;

int Safe_Float_Split(float val, int *out_int, int *out_dec) {
    if (isnan(val) || isinf(val)) {
        *out_int = 0;
        *out_dec = 0;
        return 0;
    }

    if (val > 30000.0f) val = 30000.0f;
    if (val < -30000.0f) val = -30000.0f;

    *out_int = (int)val;

    float dec_part = (val - (float)(*out_int)) * 100.0f;
    if (dec_part < 0) dec_part = -dec_part;

    *out_dec = (int)(dec_part + 0.5f);
    if (*out_dec > 99) *out_dec = 99;

    return 1;
}

void core1_main(void)
{
    disable_Watchdog();
    interrupt_global_enable(0);

    cpu_wait_event_ready();

    UI_Init();

    while (TRUE)
    {
        char disp_buf[32];
        int val_i, val_d;

        float bat_v = Battery_GetVoltage();
        Safe_Float_Split(bat_v, &val_i, &val_d);
        sprintf(disp_buf, "%cStr:%3d B:%d.%02dV ",
                ui_select_index == 0 ? '>' : ' ', start_flag, val_i, val_d);
        tft180_show_string(0, 0, disp_buf);

        sprintf(disp_buf, "%c Set: %4d     ", ui_select_index == 1 ? '>' : ' ', speed_limit);
        tft180_show_string(0, 16, disp_buf);

        Safe_Float_Split(kp1, &val_i, &val_d);
        sprintf(disp_buf, "%c kp1: %d.%02d   ", ui_select_index == 2 ? '>' : ' ', val_i, val_d);
        tft180_show_string(0, 32, disp_buf);

        Safe_Float_Split(kp2, &val_i, &val_d);
        sprintf(disp_buf, "%c kp2: %d.%02d   ", ui_select_index == 3 ? '>' : ' ', val_i, val_d);
        tft180_show_string(0, 48, disp_buf);

        Safe_Float_Split(kp3, &val_i, &val_d);
        sprintf(disp_buf, "%c kp3: %d.%02d   ", ui_select_index == 4 ? '>' : ' ', val_i, val_d);
        tft180_show_string(0, 64, disp_buf);

        Safe_Float_Split(servo_ki, &val_i, &val_d);
        sprintf(disp_buf, "%c sKi: %d.%02d   ", ui_select_index == 5 ? '>' : ' ', val_i, val_d);
        tft180_show_string(0, 80, disp_buf);

        Safe_Float_Split(servo_kd, &val_i, &val_d);
        sprintf(disp_buf, "%c sKd: %d.%02d   ", ui_select_index == 6 ? '>' : ' ', val_i, val_d);
        tft180_show_string(0, 96, disp_buf);

        sprintf(disp_buf, "  Spd: %4d cm/s  ", current_spd);
        tft180_show_string(0, 112, disp_buf);

        system_delay_ms(150);
    }
}

#pragma section all restore
