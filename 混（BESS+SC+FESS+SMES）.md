import math
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 全局配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (18, 14)


# --- 2. 模型类定义 (基于你提供的文档整合) ---

class SuperconductingMagneticEnergyStorage:
    """超导磁储能模型 (负责极高频脉冲/微秒级响应)"""

    def __init__(self, inductance=10, critical_current=5000, max_current=4000):
        self.inductance = inductance  # 电感 (H)
        self.critical_current = critical_current  # 临界电流 (A)
        self.max_current = max_current  # 运行上限
        self.current = 0  # 当前电流 (A)

    def calculate_energy(self):
        return 0.5 * self.inductance * self.current ** 2

    def get_soc(self):
        max_energy = 0.5 * self.inductance * self.max_current ** 2
        current_energy = 0.5 * self.inductance * self.current ** 2
        return min(current_energy / max_energy, 1.0) if max_energy > 0 else 0

    def update(self, power, dt):
        # 理想SMES效率极高，忽略损耗
        if abs(power) < 1e-5:  # 防止除零
            return self.get_soc()

        # 根据功率计算电流变化 dI/dt = P / (L * I)
        # 简化处理：计算能量变化 -> 电流变化
        energy_change = power * dt
        current_energy = 0.5 * self.inductance * self.current ** 2
        new_energy = current_energy + energy_change

        # 边界限制
        if new_energy <= 0:
            self.current = 0
        else:
            self.current = math.sqrt(2 * new_energy / self.inductance)
            if self.current > self.max_current:
                self.current = self.max_current
            elif self.current < -self.max_current:
                self.current = -self.max_current

        return self.get_soc()


class FlywheelModel:
    """飞轮储能系统模型 (负责中频调频)"""

    def __init__(self, radius=0.6, mass=800, max_angular_vel=1200,
                 efficiency=0.92):
        self.radius = radius
        self.mass = mass
        self.max_angular_vel = max_angular_vel
        self.efficiency = efficiency
        self.moment_of_inertia = 0.5 * mass * radius ** 2
        self.current_angular_vel = 0
        self.max_energy = 0.5 * self.moment_of_inertia * max_angular_vel ** 2

    def calculate_kinetic_energy(self):
        return 0.5 * self.moment_of_inertia * self.current_angular_vel ** 2

    def get_soc(self):
        current_energy = self.calculate_kinetic_energy()
        return current_energy / self.max_energy if self.max_energy > 0 else 0

    def update(self, power, dt):
        energy_change = power * dt
        # 考虑充放电效率
        if energy_change > 0:  # 充电
            effective_energy = energy_change * self.efficiency
        else:  # 放电
            effective_energy = energy_change / self.efficiency

        current_energy = self.calculate_kinetic_energy()
        new_energy = current_energy + effective_energy

        # 边界限制
        new_energy = max(0, min(new_energy, self.max_energy))

        # 更新转速
        self.current_angular_vel = math.sqrt(2 * new_energy / self.moment_of_inertia) if new_energy > 0 else 0
        return self.get_soc()


class LithiumIonBattery:
    """锂离子电池模型 (负责调峰/基荷)"""

    def __init__(self, nominal_voltage=3.7, capacity=50.0, internal_resistance=0.01):
        self.nominal_voltage = nominal_voltage
        self.capacity = capacity  # Ah
        self.internal_resistance = internal_resistance
        self.current_soc = 1.0

    def update(self, power, dt):
        # 简化模型：根据功率更新SOC
        energy_change = power * dt  # J
        # 电池容量转换为焦耳 (近似)
        max_energy_joule = self.capacity * self.nominal_voltage * 3600
        delta_soc = energy_change / max_energy_joule
        self.current_soc -= delta_soc

        # 边界限制 (防止过充过放)
        self.current_soc = max(0.2, min(0.95, self.current_soc))
        return self.current_soc


class Supercapacitor:
    """超级电容模型 (负责高频瞬时响应)"""

    def __init__(self, capacitance=3000, esr=0.005, max_voltage=2.7, min_voltage=1.0):
        self.capacitance = capacitance  # F
        self.esr = esr  # Ohm
        self.max_voltage = max_voltage
        self.min_voltage = min_voltage
        self.voltage = (max_voltage + min_voltage) / 2  # 初始电压

    def get_soc(self):
        return (self.voltage - self.min_voltage) / (self.max_voltage - self.min_voltage)

    def update(self, power, dt):
        if self.voltage <= 0:
            self.voltage = self.min_voltage

        # 根据功率计算电流 I = P / V
        current = power / self.voltage
        # 电容电压变化 dV/dt = I / C
        delta_v = (current * dt) / self.capacitance
        self.voltage += delta_v

        # 电压限制
        self.voltage = max(self.min_voltage, min(self.max_voltage, self.voltage))
        return self.get_soc()


# --- 3. 混合系统仿真核心 ---

def run_hybrid_simulation():
    # 1. 初始化参数
    dt = 1  # 时间步长 (s)
    duration = 2400  # 模拟时长 (s)
    time_array = np.arange(0, duration, dt)

    # --- 创建各储能单元实例 ---
    battery = LithiumIonBattery(capacity=100.0)  # 电池容量加大以支撑基荷
    fess = FlywheelModel()
    supercap = Supercapacitor()
    smes = SuperconductingMagneticEnergyStorage(inductance=10, max_current=3000)

    # --- 模拟负载功率 (包含基荷 + 随机脉冲) ---
    # 基荷 (缓慢变化 - 模拟日内负荷变化)
    base_trend = 2000 + 500 * np.sin(2 * np.pi * time_array / (24 * 3600))  # 日周期趋势
    # 随机波动 (模拟日内随机负荷)
    random_fluctuation = 400 * np.sin(time_array / 50) * (1 + 0.5 * np.sin(time_array / 10))
    # 随机尖峰脉冲 (模拟瞬时冲击/故障)
    pulses = np.zeros_like(time_array)
    # 生成一些随机的尖峰
    for i in range(50):
        pos = np.random.randint(100, len(time_array) - 100)
        pulses[pos:pos + 10] = np.random.uniform(500, 1500)  # 高幅值短时脉冲

    P_target = base_trend + random_fluctuation + pulses

    # --- 功率分配滤波器参数 (关键：分级分解) ---
    # 目标：让电池出力尽可能是一条水平直线 (调峰)
    tau_bat = 120.0  # 电池滤波常数 (极大，只留直流分量)
    tau_fess = 25.0  # 飞轮滤波常数 (提取中频)
    tau_sc = 8.0  # 电容滤波常数 (提取高频)
    # SMES 不需要滤波器，直接承担最后的残差 (极高频/脉冲)

    alpha_bat = dt / (tau_bat + dt)
    alpha_fess = dt / (tau_fess + dt)
    alpha_sc = dt / (tau_sc + dt)

    # 初始化数组
    P_bat_out = np.zeros_like(P_target)
    P_fess_out = np.zeros_like(P_target)
    P_sc_out = np.zeros_like(P_target)
    P_smes_out = np.zeros_like(P_target)

    SOC_bat_hist = []
    SOC_fess_hist = []
    SOC_sc_hist = []
    SOC_smes_hist = []

    print(f"开始四维混合储能系统仿真，总步数: {len(time_array)}")

    # --- 4. 核心仿真循环 ---
    for i in range(1, len(time_array)):
        # --- 多级功率分解 (级联低通滤波) ---

        # Step 1: 电池 (最平滑的直线 - 调峰)
        # 使用一阶低通滤波，提取直流分量
        P_bat_out[i] = alpha_bat * P_target[i] + (1 - alpha_bat) * P_bat_out[i - 1]

        # Step 2: 飞轮 (中频分量 - 一次调频)
        # 计算除去电池后的剩余功率
        P_residual_1 = P_target[i] - P_bat_out[i]
        # 对剩余功率滤波，提取中频给飞轮
        P_fess_out[i] = alpha_fess * P_residual_1 + (1 - alpha_fess) * P_fess_out[i - 1]

        # Step 3: 超级电容 (高频分量 - 二次调频)
        # 计算除去电池和飞轮后的剩余功率
        P_residual_2 = P_residual_1 - P_fess_out[i]
        # 对剩余功率滤波，提取高频给电容
        P_sc_out[i] = alpha_sc * P_residual_2 + (1 - alpha_sc) * P_sc_out[i - 1]

        # Step 4: SMES (极高频脉冲/残差 - 脉冲平抑)
        # SMES 承担所有未被前三者吸收的剩余功率 (即最后的尖峰)
        P_smes_out[i] = P_residual_2 - P_sc_out[i]

        # --- 储能单元状态更新 ---
        # 注意：这里传入的 power 是系统指令，正值为充电，负值为放电
        # 需要根据系统连接方式确定符号，这里假设指令直接对应功率流
        battery.update(P_bat_out[i], dt)
        fess.update(P_fess_out[i], dt)
        supercap.update(P_sc_out[i], dt)
        smes.update(P_smes_out[i], dt)

        # 记录状态
        SOC_bat_hist.append(battery.current_soc)
        SOC_fess_hist.append(fess.get_soc())
        SOC_sc_hist.append(supercap.get_soc())
        SOC_smes_hist.append(smes.get_soc())

    # --- 5. 结果可视化 ---
    fig, axs = plt.subplots(4, 1, figsize=(18, 14))

    # 图 1: 调峰效果 (目标：电池出力为直线)
    axs[0].plot(time_array, P_target, label='原始负载功率 (波动大)', color='gray', alpha=0.5, linewidth=1)
    axs[0].plot(time_array, P_bat_out, label='电池出力 (调峰 - 接近直线)', color='red', linewidth=3)
    axs[0].set_ylabel('功率 (W)')
    axs[0].set_title('1. 调峰效果验证：锂离子电池出力被强制平抑为水平直线')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # 图 2: 功率分解细节 (中高频分量)
    axs[1].plot(time_array, P_fess_out, label='飞轮出力 (中频)', color='green', alpha=0.8)
    axs[1].plot(time_array, P_sc_out, label='超级电容出力 (高频)', color='purple', alpha=0.8)
    axs[1].plot(time_array, P_smes_out, label='SMES出力 (极高频脉冲)', color='blue', linewidth=2)
    axs[1].set_ylabel('功率 (W)')
    axs[1].set_title('2. 调频与脉冲平衡：中、高频及尖峰脉冲分解')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # 图 3: 系统功率平衡验证
    P_total_out = P_bat_out + P_fess_out + P_sc_out + P_smes_out
    error = P_total_out - P_target
    axs[2].plot(time_array, P_total_out, label='混合储能总输出', color='black', linewidth=2)
    axs[2].plot(time_array, P_target, label='目标负载', color='blue', linestyle=':', linewidth=2)
    axs[2].set_ylabel('功率 (W)')
    axs[2].set_title('3. 系统功率平衡验证 (总输出 = 目标)')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    # 图 4: 各单元SOC状态监测
    axs[3].plot(time_array[:-1], SOC_bat_hist, label='电池 SOC', color='red')
    axs[3].plot(time_array[:-1], SOC_fess_hist, label='飞轮 SOC', color='green')
    axs[3].plot(time_array[:-1], SOC_sc_hist, label='电容 SOC', color='purple')
    axs[3].plot(time_array[:-1], SOC_smes_hist, label='SMES SOC', color='blue')
    axs[3].set_xlabel('时间 (s)')
    axs[3].set_ylabel('SOC (p.u.)')
    axs[3].set_title('4. 储能单元状态监测 (SOC)')
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)
    axs[3].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print(f"\\n--- 仿真结束 ---")
    print(f"电池出力波动标准差 (衡量平直度): {np.std(P_bat_out):.2f} W")
    print(f"SMES 承担的最大脉冲功率: {np.max(np.abs(P_smes_out)):.2f} W")
    print(f"系统功率平衡误差 (RMS): {np.sqrt(np.mean(error ** 2)):.2f} W")
    print(f"电池最终SOC: {battery.current_soc:.2%}")


if __name__ == "__main__":
    run_hybrid_simulation()
