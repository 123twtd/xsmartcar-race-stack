/* Copyright (c) 2026 清影/123twtd */
/* Copyright (c) 2026 高志禹 */
#include "battery.h"

void Battery_Init(void)
{
    // 【关键修复】：逐飞库必须初始化 ADC 模块。
    // 新版逐飞库 adc_init 需要两个参数，第二个通常是分辨率宏定义（如 ADC_12BIT）。
    // 提示：如果烧录时提示找不到 ADC_12BIT，请将它直接改为数字 12，或者在 IDE 里按住 Ctrl 左键点击 adc_init 查看它第二个参数到底叫什么。
    adc_init(BATTERY_ADC_CHANNEL, ADC_12BIT);
}

float Battery_GetVoltage(void)
{
    static float s_bat_v = 0.0f;
    static uint8 s_bat_div = 0U;

    // 降频刷新防抖：每调用 10 次才真正读取一次底层硬件
    if (s_bat_div == 0U)
    {
        // 采集 5 次求平均，并除以转换系数得到真实电压
        s_bat_v = (float)(adc_mean_filter_convert(BATTERY_ADC_CHANNEL, 5) / BATTERY_CONVERT_COEFF);
    }

    s_bat_div++;
    if (s_bat_div >= 10U)
    {
        s_bat_div = 0U;
    }

    return s_bat_v;
}
