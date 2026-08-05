/* Copyright (c) 2026 清影/123twtd */
/* Copyright (c) 2026 高志禹 */
/*
 * 按键交互与发车控制逻辑由清影/123twtd提出并审查，当前比赛版本的具体实现由高志禹完成。
 */

#include "key.h"
#include "buzzer.h"
#include "pid.h"
#include "protocol.h"

extern int start_flag;
extern int speed_limit;

int ui_select_index = 0;
#define MAX_UI_SELECT 6      // 0:Start 1:Speed 2:kp1 3:kp2 4:kp3 5:sKi 6:sKd
#define KEY_DEBOUNCE_SAMPLES 3

typedef struct {
    uint8 stable_pressed;
    uint8 sampled_pressed;
    uint8 stable_count;
} key_filter_t;

static int Key_Press_Event(key_filter_t *filter, gpio_pin_enum pin) {
    uint8 pressed = (gpio_get_level(pin) == 0) ? 1 : 0;

    if (pressed != filter->sampled_pressed) {
        filter->sampled_pressed = pressed;
        filter->stable_count = 1;
        return 0;
    }

    if (filter->stable_count < KEY_DEBOUNCE_SAMPLES) {
        filter->stable_count++;
    }

    if (filter->stable_count >= KEY_DEBOUNCE_SAMPLES &&
        filter->stable_pressed != pressed) {
        filter->stable_pressed = pressed;
        return pressed ? 1 : 0;
    }

    return 0;
}

static float Key_Adjust_Hundredth(float value, int direction) {
    int hundredths = (int)(value * 100.0f + 0.5f);
    hundredths += direction;
    if (hundredths < 0) hundredths = 0;
    return (float)hundredths / 100.0f;
}

void Key_Init(void) {
    gpio_init(KEY_START_PIN, GPI, GPIO_HIGH, GPI_PULL_UP);
    gpio_init(KEY_SPEED_ADD_PIN, GPI, GPIO_HIGH, GPI_PULL_UP);
    gpio_init(KEY_SPEED_SUB_PIN, GPI, GPIO_HIGH, GPI_PULL_UP);
    gpio_init(KEY_RESET_PIN, GPI, GPIO_HIGH, GPI_PULL_UP);
}

void Key_Service(void) {
    static key_filter_t key_start = {0};
    static key_filter_t key_add = {0};
    static key_filter_t key_sub = {0};
    static key_filter_t key_reset = {0};

    int start_event = Key_Press_Event(&key_start, KEY_START_PIN);
    int add_event = Key_Press_Event(&key_add, KEY_SPEED_ADD_PIN);
    int sub_event = Key_Press_Event(&key_sub, KEY_SPEED_SUB_PIN);
    int reset_event = Key_Press_Event(&key_reset, KEY_RESET_PIN);

    // 复位和菜单切换优先，防止多个按键同一周期修改参数。
    if (reset_event) {
        StartFlag_RequestStop();
        speed_limit = 0;
        ui_select_index = 0;

        kp1 = KP1_DEFAULT;
        kp2 = KP2_DEFAULT;
        kp3 = KP3_DEFAULT;
        servo_ki = SERVO_KI_DEFAULT;
        servo_kd = SERVO_KD_DEFAULT;

        Beep_Play(BEEP_TRIPLE);
        return;
    }

    if (start_event) {
        ui_select_index++;
        if (ui_select_index > MAX_UI_SELECT) ui_select_index = 0;
        Beep_Play(BEEP_DOUBLE);
        return;
    }

    // 加减键同时处于按下状态时不调参，避免抖动时间不同导致先加后减。
    if (gpio_get_level(KEY_SPEED_ADD_PIN) == 0 &&
        gpio_get_level(KEY_SPEED_SUB_PIN) == 0) return;

    if (add_event) {
        switch(ui_select_index) {
            case 0:
                StartFlag_RequestStart();
                Beep_Play(BEEP_DOUBLE);
                break;
            case 1:
                speed_limit += 10;
                if (speed_limit > 400) speed_limit = 400;
                Beep_Play(BEEP_DOUBLE);
                break;
            case 2:
                kp1 = Key_Adjust_Hundredth(kp1, 1);
                Beep_Play(BEEP_DOUBLE);
                break;
            case 3:
                kp2 = Key_Adjust_Hundredth(kp2, 1);
                Beep_Play(BEEP_DOUBLE);
                break;
            case 4:
                kp3 = Key_Adjust_Hundredth(kp3, 1);
                Beep_Play(BEEP_DOUBLE);
                break;
            case 5:
                servo_ki = Key_Adjust_Hundredth(servo_ki, 1);
                Beep_Play(BEEP_DOUBLE);
                break;
            case 6:
                servo_kd = Key_Adjust_Hundredth(servo_kd, 1);
                Beep_Play(BEEP_DOUBLE);
                break;
        }
        return;
    }

    if (sub_event) {
        switch(ui_select_index) {
            case 0:
                StartFlag_RequestStop();
                Beep_Play(BEEP_DOUBLE);
                break;
            case 1:
                speed_limit -= 10;
                if (speed_limit < 0) speed_limit = 0;
                Beep_Play(BEEP_DOUBLE);
                break;
            case 2:
                kp1 = Key_Adjust_Hundredth(kp1, -1);
                Beep_Play(BEEP_DOUBLE);
                break;
            case 3:
                kp2 = Key_Adjust_Hundredth(kp2, -1);
                Beep_Play(BEEP_DOUBLE);
                break;
            case 4:
                kp3 = Key_Adjust_Hundredth(kp3, -1);
                Beep_Play(BEEP_DOUBLE);
                break;
            case 5:
                servo_ki = Key_Adjust_Hundredth(servo_ki, -1);
                Beep_Play(BEEP_DOUBLE);
                break;
            case 6:
                servo_kd = Key_Adjust_Hundredth(servo_kd, -1);
                Beep_Play(BEEP_DOUBLE);
                break;
        }
    }
}
