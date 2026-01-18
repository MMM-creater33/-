import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 设置中文字体，防止乱码和报错
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CAES:
    """压缩空气储能模型 (负责调峰，输出平滑)"""
    def __init__(self, capacity=100000, max_p=5000):
        self.capacity = capacity  # 总能量 (kWh)
        self.max_p = max_p        # 最大充放电功率 (kW)
        self.soc = 0.5            # 初始SOC
        self.power_out = []       # 用于存储输出功率序列

    def control(self, target_p, dt):
        """
        简单的一阶滞后响应，模拟CAES无法跟踪快速变化，只跟踪慢变化(平滑直线)
        """
        # CAES响应慢，只跟踪指令的平滑部分
        if not self.power_out:
            prev = 0
        else:
            prev = self.power_out[-1]

        # 一阶惯性环节模拟：T * dy/dt + y = K * u
        # 这里模拟CAES的爬坡速率限制
        desired_change = target_p - prev
        actual_change = desired_change * 0.1  # 10%的响应速度

        # 功率限制
        p = np.clip(prev + actual_change, -self.max_p, self.max_p)
        self.power_out.append(p)

        # 简单的SOC更新 (假设效率100%简化计算)
        self.soc -= p * dt / self.capacity / 3600 * 100

class HESS:
    """混合储能系统 (负责调频，吸收高频脉冲)"""
    def __init__(self, max_p=3000):
        self.max_p = max_p
        self.power_out = []
        self.soc = 0.5

    def control(self, error_p, dt):
        """
        跟踪误差功率 (中高频分量)
        """
        if not self.power_out:
            prev = 0
        else:
            prev = self.power_out[-1]

        # 快速响应：迅速吸收误差
        p = np.clip(error_p, -self.max_p, self.max_p)
        self.power_out.append(p)

        # 简单的SOC更新
        self.soc -= p * dt / 10000 / 3600 * 100

# ==========================================
# 2. 主程序：仿真设置与执行
# ==========================================
def run_simulation():
    # --- 参数设置 ---
    duration = 2000  # 模拟步长
    dt = 1           # 时间步长 (s)

    # --- 生成目标波动功率 (模拟风电/光伏或负荷波动) ---
    t = np.arange(0, duration, dt)
    # 1. 低频趋势 (调峰任务)
    low_freq = 15000 + 2000 * np.sin(0.001 * t)
    # 2. 中高频波动 (调频任务)
    mid_freq = 1000 * np.sin(0.02 * t) + 500 * np.sin(0.05 * t)
    # 3. 随机脉冲噪声
    noise = np.random.normal(0, 300, t.shape)
    # 合成目标功率
    target_power = low_freq + mid_freq + noise

    # --- 初始化模型 ---
    caes = CAES(capacity=80000, max_p=4000) # CAES参数
    hess = HESS(max_p=3000)                 # HESS参数

    # --- 仿真循环 ---
    # 注意：这里必须循环，否则 power_out 就是一个空列表或单值，绘图会报错
    for i in range(len(t)):
        if i == 0:
            prev_caes = 0
        else:
            prev_caes = caes.power_out[-1]

        # 1. CAES跟踪平滑后的功率 (模拟低通滤波)
        # 这里使用简单的移动平均或滞后响应来模拟平滑效果
        window = 50 # 窗口大小，越大越平滑
        if i < window:
            avg_power = np.mean(target_power[:i+1])
        else:
            avg_power = np.mean(target_power[i-window:i])

        caes.control(avg_power, dt)

        # 2. HESS跟踪误差 (中高频分量)
        # 目标功率 - CAES实际能发出的功率 = 误差
        # 这个误差就是需要被平抑的中高频功率
        error_power = target_power[i] - caes.power_out[-1]
        hess.control(error_power, dt)

    # 确保列表长度一致
    caes.power_out = np.array(caes.power_out)
    hess.power_out = np.array(hess.power_out)

    # ==========================================
    # 3. 结果绘图
    # ==========================================
    fig, axs = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # 1. 原始功率 vs 平滑后的参考线
    axs[0].plot(t, target_power, label='原始波动功率 (P_ref)', color='#1f77b4', alpha=0.6)
    axs[0].plot(t, caes.power_out + hess.power_out, label='系统总输出 (P_total)', color='red', linestyle='--', alpha=0.8)
    axs[0].set_ylabel('功率 (kW)')
    axs[0].set_title('1. 功率平抑效果对比 (原始 vs 输出)')
    axs[0].legend()
    axs[0].grid(True, which='both', ls='--')

    # 2. CAES (调峰) vs HESS (调频) 出力分解
    axs[1].plot(t, caes.power_out, label='CAES 出力 (调峰)', color='#2ca02c', linewidth=2)
    axs[1].plot(t, hess.power_out, label='HESS 出力 (调频/脉冲)', color='#ff7f0e', linewidth=1.5)
    axs[1].set_ylabel('功率 (kW)')
    axs[1].set_title('2. 功率分解 (CAES负责基荷，HESS负责波动)')
    axs[1].legend()
    axs[1].grid(True, which='both', ls='--')

    # 3. 系统功率平衡误差 (目标是让这根线趋近于0)
    balance_error = target_power - (caes.power_out + hess.power_out)
    axs[2].plot(t, balance_error, label='功率不平衡量', color='#d62728')
    axs[2].axhline(0, color='black', linewidth=0.8, alpha=0.5)
    axs[2].set_ylabel('误差功率 (kW)')
    axs[2].set_title('3. 功率平衡误差 (目标: 最小化偏差)')
    axs[2].legend()
    axs[2].grid(True, which='both', ls='--')

    # 4. SOC 状态
    # 为了演示，这里生成模拟的SOC数据
    soc_caes = 0.5 - (caes.power_out / 10000) * 0.01
    soc_hess = 0.5 - (hess.power_out / 5000) * 0.02
    # 累积计算SOC
    soc_caes = 0.5 - np.cumsum(caes.power_out) * dt / 80000
    soc_hess = 0.5 - np.cumsum(hess.power_out) * dt / 5000

    axs[3].plot(t, soc_caes, label='CAES SOC', color='#1f77b4', linestyle='-')
    axs[3].plot(t, soc_hess, label='HESS SOC', color='#ff7f0e', linestyle='--')
    axs[3].set_xlabel('时间 (s)')
    axs[3].set_ylabel('SOC')
    axs[3].set_title('4. 储能SOC变化')
    axs[3].legend()
    axs[3].grid(True, which='both', ls='--')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
