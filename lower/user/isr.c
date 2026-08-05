/* Copyright (c) 2026 清影/123twtd */
/* Copyright (c) 2026 高志禹 */
/*
 * 中断组织逻辑由清影/123twtd提出并审查，当前比赛版本的具体实现由高志禹完成。
 */

#include "isr_config.h"
#include "isr.h"

// 注意：
// uart0_rx_isr (串口0接收中断) 和 cc60_pit_ch0_isr (PIT定时器中断)
// 已经移动到了 cpu0_main.c 文件底部处理。

// ================= 以下为防报错的空壳中断 =================
IFX_INTERRUPT(cc60_pit_ch1_isr, 0, CCU6_0_CH1_ISR_PRIORITY) { pit_clear_flag(CCU60_CH1); }
IFX_INTERRUPT(cc61_pit_ch0_isr, 0, CCU6_1_CH0_ISR_PRIORITY) { pit_clear_flag(CCU61_CH0); }
IFX_INTERRUPT(cc61_pit_ch1_isr, 0, CCU6_1_CH1_ISR_PRIORITY) { pit_clear_flag(CCU61_CH1); }

IFX_INTERRUPT(uart0_tx_isr, 0, UART0_TX_INT_PRIO) {}
IFX_INTERRUPT(uart1_tx_isr, 0, UART1_TX_INT_PRIO) {}
IFX_INTERRUPT(uart1_rx_isr, 0, UART1_RX_INT_PRIO) {}
IFX_INTERRUPT(uart2_tx_isr, 0, UART2_TX_INT_PRIO) {}
IFX_INTERRUPT(uart2_rx_isr, 0, UART2_RX_INT_PRIO) {}
IFX_INTERRUPT(uart3_tx_isr, 0, UART3_TX_INT_PRIO) {}
IFX_INTERRUPT(uart3_rx_isr, 0, UART3_RX_INT_PRIO) {}

IFX_INTERRUPT(uart0_er_isr, 0, UART0_ER_INT_PRIO) {}
IFX_INTERRUPT(uart1_er_isr, 0, UART1_ER_INT_PRIO) {}
IFX_INTERRUPT(uart2_er_isr, 0, UART2_ER_INT_PRIO) {}
IFX_INTERRUPT(uart3_er_isr, 0, UART3_ER_INT_PRIO) {}

IFX_INTERRUPT(exti_ch0_ch4_isr, 0, EXTI_CH0_CH4_INT_PRIO)
{
    if(exti_flag_get(ERU_CH0_REQ0_P15_4)) { exti_flag_clear(ERU_CH0_REQ0_P15_4); }
    if(exti_flag_get(ERU_CH4_REQ13_P15_5)){ exti_flag_clear(ERU_CH4_REQ13_P15_5);}
}
IFX_INTERRUPT(exti_ch1_ch5_isr, 0, EXTI_CH1_CH5_INT_PRIO)
{
    if(exti_flag_get(ERU_CH1_REQ10_P14_3)){ exti_flag_clear(ERU_CH1_REQ10_P14_3); }
    if(exti_flag_get(ERU_CH5_REQ1_P15_8)) { exti_flag_clear(ERU_CH5_REQ1_P15_8); }
}
IFX_INTERRUPT(exti_ch3_ch7_isr, 0, EXTI_CH3_CH7_INT_PRIO)
{
    if(exti_flag_get(ERU_CH3_REQ6_P02_0)) { exti_flag_clear(ERU_CH3_REQ6_P02_0); }
    if(exti_flag_get(ERU_CH7_REQ16_P15_1)){ exti_flag_clear(ERU_CH7_REQ16_P15_1); }
}
IFX_INTERRUPT(dma_ch5_isr, 0, DMA_INT_PRIO) {}
