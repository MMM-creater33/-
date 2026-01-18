import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 参数设置 ---
T = 2000  # 模拟时间步长
dt = 1    # 采样时间
t = np.arange(T)  # t 的长度为 2000

# --- 2. 生成原始波动功率 (模拟需要平抑的负荷/新能源波动) ---
# 目标：让系统输出变成 15000 的水平直线
P_target = 15000
# 原始功率包含低频趋势和高频冲击
P_original = (P_target + 5000 * np.sin(0.005 * t) +  # 低频波动 (大趋势)
              3000 * np.sin(0.02 * t) +             # 中频波动
              np.random.normal(0, 1000, T))         # 高频噪声 (随机脉冲)

# --- 3. 混合储能模型参数 (5个模型) ---
# Model 1: 抽水蓄能 (慢速, 大容量, 负责基荷)
max_p_1 = 10000; capacity_1 = 500000; soc_1 = [0.5]  # 初始SOC 50%
# Model 2: 压缩空气 (慢速, 中容量)
max_p_2 = 8000; capacity_2 = 300000; soc_2 = [0.5]
# Model 3: 熔盐储热 (中速, 较小容量)
max_p_3 = 5000; capacity_3 = 100000; soc_3 = [0.5]
# Model 4: 锂电池 (快速, 小容量, 负责脉冲)
max_p_4 = 3000; capacity_4 = 50000; soc_4 = [0.5]
# Model 5: 超级电容 (极速, 极小容量, 负责尖峰)
max_p_5 = 2000; capacity_5 = 10000; soc_5 = [0.5]

# --- 4. 功率分配逻辑 (核心控制算法) ---
# 目标：让 P_hybrid 成为一条水平线 (P_target)
# 原理：利用低通滤波器将波动切分，慢速机承担低频，快速机承担高频。

# 1. 计算系统需要提供的总补偿功率 (让输出变直线的关键)
# 如果原始功率 > 目标 -> 系统充电 (P_compensate < 0)
# 如果原始功率 < 目标 -> 系统放电 (P_compensate > 0)
P_error = P_target - P_original # 系统需要"吞吐"的功率

# 2. 设计滤波器 (切分频率)
# --- 调参关键点 ---
# 让 Model 1 & 2 (慢速) 承担绝大部分能量，让 Model 4 & 5 (快速) 只承担残差
# 这里的分子 [1] 分母 [1, 0.02] 构成了低通滤波器

# 慢速通道 (分配给 Model 1, 2, 3)
b_slow, a_slow = signal.butter(2, 0.005, 'low') # 频率极低，只让大趋势通过
P_slow_cmd = signal.filtfilt(b_slow, a_slow, P_error)

# 中速通道 (分配给 Model 3)
b_med, a_med = signal.butter(2, 0.01, 'low')
P_med_cmd = signal.filtfilt(b_med, a_med, P_error)

# 高速通道 (残差，分配给 Model 4, 5)
P_fast_cmd = P_error - P_slow_cmd

# 3. 初始化功率记录列表 (长度将与 t 保持一致)
p1_list, p2_list, p3_list, p4_list, p5_list = [], [], [], [], []

# 循环计算 (注意：range(T) 会产生 2000 个数据点)
for i in range(T):
    # --- 5. 功率分配策略 ---
    # Model 1 (抽水蓄能): 承担主要的低频波动 (基荷)
    p1 = P_slow_cmd[i] * 0.6  # 分配 60% 的低频任务

    # Model 2 (压缩空气): 承担剩余的低频波动
    p2 = P_slow_cmd[i] * 0.4  # 分配 40% 的低频任务

    # Model 3 (储热): 承担中频波动 (作为补充)
    p3 = P_med_cmd[i] * 0.2   # 参与一部分

    # Model 4 & 5 (电池/电容): 承担高频脉冲 (平抑波动)
    # 两者分工：电池扛大脉冲，电容扛微小抖动
    p4 = P_fast_cmd[i] * 0.7  # 电池承担大部分高频
    p5 = P_fast_cmd[i] * 0.3  # 电容承担小部分

    # --- 6. 物理限制 (限幅) ---
    # 简单限幅逻辑 (实际应用需更复杂判断)
    p1 = np.clip(p1, -max_p_1, max_p_1)
    p2 = np.clip(p2, -max_p_2, max_p_2)
    p3 = np.clip(p3, -max_p_3, max_p_3)
    p4 = np.clip(p4, -max_p_4, max_p_4)
    p5 = np.clip(p5, -max_p_5, max_p_5)

    # --- 7. 更新 SOC (荷电状态) ---
    # SOC变化 = (功率 * 效率) / 容量
    # 注意：此处需要根据充放电方向调整符号，简单起见假设效率为1

    # 抽水蓄能
    delta_soc1 = -p1 * dt / capacity_1
    new_soc1 = soc_1[-1] + delta_soc1
    new_soc1 = np.clip(new_soc1, 0.1, 0.9) # 限制在 10%-90%
    soc_1.append(new_soc1)

    # 压缩空气
    delta_soc2 = -p2 * dt / capacity_2
    new_soc2 = soc_2[-1] + delta_soc2
    new_soc2 = np.clip(new_soc2, 0.1, 0.9)
    soc_2.append(new_soc2)

    # 熔盐储热
    delta_soc3 = -p3 * dt / capacity_3
    new_soc3 = soc_3[-1] + delta_soc3
    new_soc3 = np.clip(new_soc3, 0.1, 0.9)
    soc_3.append(new_soc3)

    # 锂电池
    delta_soc4 = -p4 * dt / capacity_4
    new_soc4 = soc_4[-1] + delta_soc4
    new_soc4 = np.clip(new_soc4, 0.2, 0.8) # 电池范围稍宽
    soc_4.append(new_soc4)

    # 超级电容
    delta_soc5 = -p5 * dt / capacity_5
    new_soc5 = soc_5[-1] + delta_soc5
    new_soc5 = np.clip(new_soc5, 0.1, 0.9)
    soc_5.append(new_soc5)

    # 记录功率 (用于画图)
    p1_list.append(p1)
    p2_list.append(p2)
    p3_list.append(p3)
    p4_list.append(p4)
    p5_list.append(p5)

# --- 修复维度的关键步骤 ---
# 我们的 soc 列表长度是 2001 (初始值 + 2000次循环)
# 而 t 的长度是 2000。为了画图，我们需要把 soc 的初始值去掉，或者把 t 对齐。
# 方法：去掉 soc 的第一个元素，使其长度变为 2000
soc_1 = soc_1[1:]
soc_2 = soc_2[1:]
soc_3 = soc_3[1:]
soc_4 = soc_4[1:]
soc_5 = soc_5[1:]

# --- 8. 计算混合系统总输出 ---
# 混合输出 = 原始功率 + 储能系统提供的功率 (P_error)
# 理论上应该是一条直线
P_hybrid = P_original + np.array(p1_list) + np.array(p2_list) + np.array(p3_list) + np.array(p4_list) + np.array(p5_list)

# --- 9. 画图 ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

# 图1: 系统功率平衡 (目标是水平直线)
axs[0].plot(t, P_original, label='原始波动功率', color='red', alpha=0.6)
axs[0].plot(t, P_hybrid, label='混合系统输出 (目标: 水平直线)', color='black', linewidth=2)
axs[0].axhline(y=P_target, color='green', linestyle='--', label='目标基准线 (15MW)')
axs[0].set_ylabel('功率 (kW)')
axs[0].set_title('图1: 系统功率平衡控制 (脉冲消除, 曲线变水平)')
axs[0].legend()
axs[0].grid(True)

# 图2: 功率分配细节
axs[1].plot(t, p1_list, label='Model 1 (抽蓄/慢)', alpha=0.8)
axs[1].plot(t, p4_list, label='Model 4 (电池/快)', alpha=0.8)
axs[1].plot(t, p5_list, label='Model 5 (电容/极快)', alpha=0.8)
axs[1].set_ylabel('功率 (kW)')
axs[1].set_title('图2: 功率分配细节 (慢速承担基荷, 快速承担脉冲)')
axs[1].legend()
axs[1].grid(True)

# 图3: SOC 变化 (安全性)
axs[2].plot(t, soc_1, label='抽水蓄能 SOC')
axs[2].plot(t, soc_4, label='锂电池 SOC')
axs[2].plot(t, soc_5, label='超级电容 SOC')
axs[2].set_ylabel('SOC')
axs[2].set_xlabel('时间 (s)')
axs[2].set_title('图3: 储能荷电状态 (SOC) 变化')
axs[2].legend()
axs[2].grid(True)

plt.show()
