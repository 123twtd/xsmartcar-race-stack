/* Copyright (c) 2026 清影/123twtd */
/*
 * buzzer.c
 *
 *  Created on: 2026年6月8日
 *      Author: A
 */




#include "buzzer.h"

static int beep_pattern = 0;
static int buzzer_ticks = 0;

void Buzzer_Init(void) {
    gpio_init(BUZZER_PIN, GPO, GPIO_LOW, GPO_PUSH_PULL);
}

void Beep_Play(int pattern) {
    if (beep_pattern == BEEP_LOW_VOLT && pattern != BEEP_LOW_VOLT) {
        return;
    }

    beep_pattern = pattern;
    buzzer_ticks = 0;
}

void Buzzer_Service(void) {
    if (beep_pattern == BEEP_OFF) {
        gpio_set_level(BUZZER_PIN, 0);
        return;
    }

    buzzer_ticks++;

    // 长鸣半秒
    if (beep_pattern == BEEP_LONG) {
        if (buzzer_ticks == 1)       gpio_set_level(BUZZER_PIN, 1);
        else if (buzzer_ticks >= 50) {
            gpio_set_level(BUZZER_PIN, 0);
            beep_pattern = BEEP_OFF;
        }
    }
    //短促双声
    else if (beep_pattern == BEEP_DOUBLE) {
        if (buzzer_ticks == 1)       gpio_set_level(BUZZER_PIN, 1);
        else if (buzzer_ticks == 10) gpio_set_level(BUZZER_PIN, 0);
        else if (buzzer_ticks == 20) gpio_set_level(BUZZER_PIN, 1);
        else if (buzzer_ticks == 30) {
            gpio_set_level(BUZZER_PIN, 0);
            beep_pattern = BEEP_OFF;
        }
    }
    // 短促三声
    else if (beep_pattern == BEEP_TRIPLE) {
        if (buzzer_ticks == 1)       gpio_set_level(BUZZER_PIN, 1);
        else if (buzzer_ticks == 10) gpio_set_level(BUZZER_PIN, 0);
        else if (buzzer_ticks == 20) gpio_set_level(BUZZER_PIN, 1);
        else if (buzzer_ticks == 30) gpio_set_level(BUZZER_PIN, 0);
        else if (buzzer_ticks == 40) gpio_set_level(BUZZER_PIN, 1);
        else if (buzzer_ticks == 50) {
            gpio_set_level(BUZZER_PIN, 0);
            beep_pattern = BEEP_OFF;
        }
    }
    // 低压报警：Buzzer_Service 每 10ms 调用一次，1000 次约等于 10 秒
    else if (beep_pattern == BEEP_LOW_VOLT) {
        if (buzzer_ticks == 1) {
            gpio_set_level(BUZZER_PIN, 1);
        }
        else if (buzzer_ticks >= 1000) {
            gpio_set_level(BUZZER_PIN, 0);
            beep_pattern = BEEP_OFF;
        }
    }
}
