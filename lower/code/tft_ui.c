/* Copyright (c) 2026 清影/123twtd */
/* Copyright (c) 2026 高志禹 */
#include "tft_ui.h"
#include "zf_device_tft180.h"
#include "zf_common_headfile.h"
#include <stdio.h>

static uint32 print_time_cnt = 0;

void UI_Init(void) {
    tft180_init();

    system_delay_ms(300);

    tft180_set_dir(TFT180_CROSSWISE_180);           // 设置为横屏 180 度
    tft180_set_color(RGB565_MAGENTA, RGB565_BLACK); // 设置为品红字，黑底

    tft180_clear();
    tft180_show_string(0, 0, "ShunZhi Team");
}

void UI_Task(float speed_cm_s, float rpm) {
    char tft_buf[32];

    print_time_cnt++;

    // 每 200ms (20个10ms周期) 刷新一次屏幕
    if (print_time_cnt >= 20) {
        print_time_cnt = 0;

        // 防崩溃处理：提取整数部分和小数部分，用 %d 整数来打印浮点数！
        // 处理 cm/s
        int spd_int = (int)speed_cm_s;                             // 整数部分
        int spd_dec = (int)(speed_cm_s * 100.0f) % 100;            // 小数部分 (保留两位)
        if (spd_dec < 0) spd_dec = -spd_dec;                       // 解决负数时小数多出负号的问题

        // 处理 RPM
        int rpm_int = (int)rpm;
        int rpm_dec = (int)(rpm * 100.0f) % 100;
        if (rpm_dec < 0) rpm_dec = -rpm_dec;

        // 每次刷字前重新激活一下专属画笔
        tft180_set_color(RGB565_MAGENTA, RGB565_BLACK);

        // 拼接字符串（注意后面的空格，是为了覆盖掉数字变小时的残留字符）
        sprintf(tft_buf, "Speed: %d.%02d cm/s   ", spd_int, spd_dec);
        tft180_show_string(0, 30, tft_buf);

        sprintf(tft_buf, "RPM  : %d.%02d r/m    ", rpm_int, rpm_dec);
        tft180_show_string(0, 50, tft_buf);
    }
}
