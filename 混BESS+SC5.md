import matplotlib.pyplot as plt
import numpy as np

# --- 1. 全局配置 (解决中文乱码) ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 模型类定义 (整合自你的截图) ---
class LithiumIonBattery:
    def __init__(self, nominal_voltage, capacity, internal_resistance):
        self.nominal_voltage = nominal_voltage
        self.capacity = capacity  # Ah
        self.internal_resistance = internal_resistance
        self.initial_soc = 1.0
        self.current_soc = self.initial_soc
        self.current = 0
        self.time = 0

    def calculate_soc(self, current, time):
        # 核心算法：计算SOC
        self.current = current
        self.time = time
        # 计算充放电电量变化 (Ah)
        charge_change = (current * time) / 3600
        # 更新SOC
        self.current_soc = self.current_soc - charge_change / self.capacity

        # 边界检查
        if self.current_soc < 0:
            self.current_soc = 0
        elif self.current_soc > 1:
            self.current_soc = 1
        return self.current_soc

    def calculate_terminal_voltage(self, ocv):
        # 计算端电压: V = OCV - I*R
        return ocv - self.current * self.internal_resistance

class Supercapacitor:
    def __init__(self, capacitance, esr, max_voltage, min_voltage):
        self.capacitance = capacitance  # F
        self.esr = esr  # 等效串联电阻
        self.max_voltage = max_voltage
        self.min_voltage = min_voltage
        self.current_voltage = (max_voltage + min_voltage) / 2 # 初始化为中间电压

    def discharge(self, current, time):
        # 放电逻辑：电压下降
        voltage_drop = current * self.esr
        self.current_voltage -= voltage_drop

        # 电压边界限制
        if self.current_voltage < self.min_voltage:
            self.current_voltage = self.min_voltage
        return self.current_voltage

    def charge(self, voltage):
        # 充电逻辑：电压上升
        if voltage <= self.max_voltage:
            self.current_voltage = voltage
        else:
            self.current_voltage = self.max_voltage
        return self.current_voltage

# --- 3. 混合控制策略 (核心逻辑) ---
def run_hybrid_simulation():
    # 1. 初始化参数
    dt = 1  # 时间步长 (s)
    duration = 1000  # 模拟时长
    time_array = np.arange(0, duration, dt)

    # 创建电池实例 (3.7V, 20Ah, 0.01欧姆)
    battery = LithiumIonBattery(3.7, 20.0, 0.01)
    # 创建电容实例 (3000F, ESR 0.01, 电压范围 1.0-2.7V)
    supercap = Supercapacitor(3000, 0.01, 2.7, 1.0)

    # 2. 生成模拟负载功率 (包含平滑趋势 + 突变脉冲)
    # 模拟一个缓慢下降的基荷 (调峰部分)
    base_power = 500 + 100 * np.sin(time_array / 50)
    # 模拟随机脉冲负载 (调频部分)
    pulse_power = 200 * np.sin(time_array / 5) + 100 * np.sin(time_array / 2)
    # 组合功率 (目标功率)
    P_target = base_power + pulse_power

    # 3. 功率分配算法 (一阶低通滤波)
    # 目标：让电池出力 P_battery 变成水平直线，平抑脉冲
    # tau (时间常数): 控制电池响应速度。值越大，电池出力越平滑(越像直线)。
    tau = 20.0
    alpha = dt / (tau + dt) # 滤波系数

    P_battery = np.zeros_like(P_target)
    P_sc = np.zeros_like(P_target) # 电容功率
    soc_history = [] # 记录SOC
    v_sc_history = [] # 记录电容电压

    # 初始状态
    P_battery[0] = P_target[0]

    print(f"模拟开始，总步长: {len(time_array)}")

    # 4. 循环计算 (核心循环)
    for i in range(1, len(time_array)):
        t = time_array[i]

        # --- 功率分配逻辑 (关键点) ---
        # 1. 电池出力通过低通滤波器平滑 -> 变成近似水平的直线
        P_battery[i] = alpha * P_target[i] + (1 - alpha) * P_battery[i-1]

        # 2. 电容出力 = 总目标 - 电池出力 -> 跟踪脉冲
        P_sc[i] = P_target[i] - P_battery[i]

        # --- 电池模型计算 ---
        # 假设电压恒定计算电流 (简化模型)
        battery_current = P_battery[i] / battery.nominal_voltage
        battery.calculate_soc(battery_current, dt)
        soc_history.append(battery.current_soc)

        # --- 电容模型计算 ---
        # 功率 = 电压 * 电流 -> 计算电容电流
        if supercap.current_voltage > 0:
            sc_current = P_sc[i] / supercap.current_voltage
        else:
            sc_current = 0

        # 根据电流和时间更新电容电压 (简化模型，实际需积分)
        # dV = (I * dt) / C
        delta_v = (sc_current * dt) / supercap.capacitance
        supercap.current_voltage += delta_v
        # 边界限制
        if supercap.current_voltage > supercap.max_voltage:
            supercap.current_voltage = supercap.max_voltage
        elif supercap.current_voltage < supercap.min_voltage:
            supercap.current_voltage = supercap.min_voltage

        v_sc_history.append(supercap.current_voltage)

    # --- 5. 结果可视化 ---
    # 创建画布，3行1列
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 图1: 功率对比 (目标 vs 电池出力)
    axs[0].plot(time_array[1:], P_target[1:], label='目标功率 (含脉冲)', color='#1f77b4', alpha=0.6)
    axs[0].plot(time_array[1:], P_battery[1:], label='电池出力 (调峰/平滑)', color='#d62728', linewidth=2)
    axs[0].set_ylabel('功率 (W)')
    axs[0].set_title('1. 功率分配与脉冲平抑效果')
    axs[0].legend()
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # 图2: SOC 变化
    axs[1].plot(time_array[1:], soc_history, label='电池 SOC', color='#2ca02c')
    axs[1].set_ylabel('SOC (p.u.)')
    axs[1].set_title('2. 电池荷电状态 (SOC) 变化')
    axs[1].legend()
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[1].set_ylim(-0.05, 1.05) # 限制Y轴范围

    # 图3: 系统功率平衡验证
    # 混合输出 = 电池 + 电容
    P_hybrid = np.array(P_battery[1:]) + np.array(P_sc[1:])
    error = np.array(P_hybrid) - np.array(P_target[1:])
    axs[2].plot(time_array[1:], P_hybrid, label='混合系统总输出', color='#9467bd', linewidth=2)
    axs[2].plot(time_array[1:], error, label='平衡误差', color='gray', linestyle='--', alpha=0.5)
    axs[2].set_ylabel('功率 (W)')
    axs[2].set_xlabel('时间 (s)')
    axs[2].set_title('3. 系统功率平衡验证 (总输出应接近目标功率)')
    axs[2].legend()
    axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # 打印关键指标
    print(f"模拟结束。")
    print(f"电池最终SOC: {battery.current_soc:.2f}")
    print(f"电容最终电压: {supercap.current_voltage:.2f} V")

if __name__ == "__main__":
    run_hybrid_simulation()
