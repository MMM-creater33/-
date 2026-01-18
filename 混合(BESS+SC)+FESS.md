import math
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 全局配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 10)


# --- 2. 模型类定义 (整合版) ---

class FlywheelModel:
    """飞轮储能系统模型 (负责中频调频)"""

    def __init__(self, radius=0.6, mass=800, max_angular_vel=1200,
                 efficiency=0.92, friction_coeff=0.005):
        self.radius = radius
        self.mass = mass
        self.max_angular_vel = max_angular_vel
        self.efficiency = efficiency
        self.friction_coeff = friction_coeff
        self.moment_of_inertia = 0.5 * mass * radius ** 2
        self.current_angular_vel = 0
        self.max_energy = self.calculate_kinetic_energy(max_angular_vel)

    def calculate_kinetic_energy(self, angular_vel=None):
        if angular_vel is None: angular_vel = self.current_angular_vel
        return 0.5 * self.moment_of_inertia * angular_vel ** 2

    def get_soc(self):
        # 飞轮SOC定义为当前能量/最大能量
        current_energy = self.calculate_kinetic_energy()
        return current_energy / self.max_energy if self.max_energy > 0 else 0

    def update(self, power, dt):
        # power > 0 充电 (吸收能量), power < 0 放电 (释放能量)
        energy_change = power * dt

        if energy_change > 0:  # 充电
            effective_energy = energy_change * self.efficiency
        else:  # 放电
            effective_energy = energy_change / self.efficiency

        current_energy = self.calculate_kinetic_energy()
        new_energy = current_energy + effective_energy

        # 边界限制
        if new_energy < 0: new_energy = 0
        if new_energy > self.max_energy:
            new_energy = self.max_energy

        # 更新转速
        if new_energy == 0:
            self.current_angular_vel = 0
        else:
            self.current_angular_vel = math.sqrt(2 * new_energy / self.moment_of_inertia)

        return self.calculate_kinetic_energy() - current_energy


class LithiumIonBattery:
    """锂离子电池模型 (负责调峰/基荷)"""

    def __init__(self, nominal_voltage, capacity, internal_resistance):
        self.nominal_voltage = nominal_voltage
        self.capacity = capacity  # Ah
        self.internal_resistance = internal_resistance
        self.current_soc = 1.0
        self.max_capacity_joule = capacity * nominal_voltage * 3600  # 近似最大能量 (J)

    def update(self, power, dt):
        # 简化模型：根据功率更新SOC
        energy_change = power * dt  # J
        delta_soc = energy_change / (self.capacity * self.nominal_voltage * 3600)
        self.current_soc -= delta_soc / 3600  # 简化处理，实际需考虑库伦效率

        # 边界限制
        self.current_soc = max(0.1, min(0.9, self.current_soc))  # 限制在10%-90%
        return self.current_soc


class Supercapacitor:
    """超级电容模型 (负责高频瞬时响应)"""

    def __init__(self, capacitance, esr, max_voltage, min_voltage):
        self.capacitance = capacitance  # F
        self.esr = esr  # Ohm
        self.max_voltage = max_voltage
        self.min_voltage = min_voltage
        self.voltage = (max_voltage + min_voltage) / 2

    def get_soc(self):
        return (self.voltage - self.min_voltage) / (self.max_voltage - self.min_voltage)

    def update(self, power, dt):
        # 功率模型更新电压
        if abs(power) > 0 and self.voltage > 0:
            current = power / self.voltage
            # dV/dt = I/C (忽略ESR动态，简化为平均值)
            delta_v = (current * dt) / self.capacitance
            self.voltage += delta_v

            # 电压限制
            self.voltage = max(self.min_voltage, min(self.max_voltage, self.voltage))
        return self.voltage


# --- 3. 混合系统仿真核心 ---

def run_hybrid_simulation():
    # 1. 初始化参数
    dt = 1  # 时间步长 (s)
    duration = 1200  # 模拟时长 (s)
    time_array = np.arange(0, duration, dt)

    # --- 创建各储能单元实例 ---
    # 电池：负责最平滑的基荷 (调峰)
    battery = LithiumIonBattery(nominal_voltage=3.7, capacity=50.0, internal_resistance=0.01)

    # 飞轮：负责中频波动 (调频)
    fess = FlywheelModel(radius=0.6, mass=800, max_angular_vel=1200, efficiency=0.92)

    # 超级电容：负责最高频脉冲 (瞬时响应)
    supercap = Supercapacitor(capacitance=3000, esr=0.005, max_voltage=2.7, min_voltage=1.0)

    # --- 模拟负载功率 (包含基荷 + 中频波动 + 高频脉冲) ---
    # 基荷 (缓慢变化)
    base_trend = 800 + 100 * np.sin(time_array / 150)
    # 中频波动 (飞轮处理)
    mid_freq = 150 * np.sin(time_array / 20)
    # 高频脉冲 (电容处理)
    high_freq = 100 * np.sin(time_array / 2) + 50 * np.sin(time_array / 0.5)
    # 目标总功率
    P_target = base_trend + mid_freq + high_freq

    # --- 功率分配滤波器参数 (关键：分解中高频) ---
    # 1. 电池滤波器 (时间常数大，提取直流/基荷)
    tau_bat = 50.0  # 电池响应最慢
    alpha_bat = dt / (tau_bat + dt)

    # 2. 飞轮滤波器 (时间常数中等，提取中频)
    # 先计算除去电池承担部分后的剩余功率
    # 再通过飞轮滤波器提取中频分量
    tau_fess = 10.0  # 飞轮响应中等
    alpha_fess = dt / (tau_fess + dt)

    # 初始化数组
    P_bat_out = np.zeros_like(P_target)
    P_fess_out = np.zeros_like(P_target)
    P_sc_out = np.zeros_like(P_target)

    SOC_bat_hist = []
    SOC_fess_hist = []
    V_sc_hist = []

    print(f"开始混合储能系统仿真，总步数: {len(time_array)}")

    # --- 4. 核心仿真循环 ---
    for i in range(1, len(time_array)):
        # --- 功率分解算法 (一阶低通滤波级联) ---

        # Step 1: 电池出力 (最平滑的直线)
        # 目标：让电池出力接近水平直线，只承担缓慢变化的功率
        P_bat_out[i] = alpha_bat * P_target[i] + (1 - alpha_bat) * P_bat_out[i - 1]

        # Step 2: 计算剩余功率 (需要由飞轮和电容承担)
        P_residual_1 = P_target[i] - P_bat_out[i]

        # Step 3: 飞轮出力 (承担中频分量)
        # 对剩余功率进行第二次滤波，提取中频给飞轮，高频给电容
        # 这里使用滤波后的值作为飞轮指令
        P_fess_filtered = alpha_fess * P_residual_1 + (1 - alpha_fess) * (P_target[i - 1] - P_bat_out[i - 1])
        P_fess_out[i] = P_fess_filtered

        # Step 4: 超级电容出力 (承担所有剩余高频脉冲)
        P_sc_out[i] = P_residual_1 - P_fess_out[i]

        # --- 储能单元状态更新 ---

        # 电池更新 (模拟调峰)
        battery.update(P_bat_out[i], dt)
        SOC_bat_hist.append(battery.current_soc)

        # 飞轮更新 (模拟中频调频)
        fess.update(P_fess_out[i], dt)
        SOC_fess_hist.append(fess.get_soc())

        # 电容更新 (模拟高频瞬时响应)
        supercap.update(P_sc_out[i], dt)
        V_sc_hist.append(supercap.voltage)

    # --- 5. 结果可视化 ---
    fig, axs = plt.subplots(4, 1, figsize=(16, 12))

    # 图 1: 功率分配总览 (目标 vs 各单元)
    axs[0].plot(time_array, P_target, label='目标功率 $P_{load}$', color='gray', linewidth=1, alpha=0.7)
    axs[0].plot(time_array, P_bat_out, label='电池出力 $P_{bat}$ (调峰)', color='red', linewidth=2)
    axs[0].plot(time_array, P_bat_out + P_fess_out, label='电池+飞轮 (平抑后)', color='orange', linewidth=2,
                linestyle='--')
    axs[0].set_ylabel('功率 (W)')
    axs[0].set_title('1. 功率分配总览：目标功率 vs 储能系统出力')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # 图 2: 详细功率分解 (展示中高频分离)
    axs[1].plot(time_array, P_target - P_bat_out, label='剩余功率 (需平抑)', color='blue', alpha=0.5)
    axs[1].plot(time_array, P_fess_out, label='飞轮出力 $P_{fess}$ (中频调频)', color='green', linewidth=2)
    axs[1].plot(time_array, P_sc_out, label='电容出力 $P_{sc}$ (高频脉冲)', color='purple', linewidth=2)
    axs[1].set_ylabel('功率 (W)')
    axs[1].set_title('2. 功率分解细节：中频(飞轮)与高频(电容)分离')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # 图 3: 状态监测 (SOC/电压)
    axs[2].plot(time_array[:-1], SOC_bat_hist, label='电池 SOC', color='red')
    axs[2].plot(time_array[:-1], SOC_fess_hist, label='飞轮 SOC', color='green')
    axs[2].plot(time_array[:-1], np.array(V_sc_hist) / (2.7 - 1.0), label='电容归一化电压', color='purple',
                linestyle='--')  # 归一化便于显示
    axs[2].set_ylabel('状态 (p.u.)')
    axs[2].set_title('3. 储能单元状态监测 (SOC & 电压)')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    axs[2].set_ylim(0, 1)

    # 图 4: 系统功率平衡验证
    P_total_out = P_bat_out + P_fess_out + P_sc_out
    error = P_total_out - P_target
    axs[3].plot(time_array, P_total_out, label='系统总输出', color='black', linewidth=2)
    axs[3].plot(time_array, P_target, label='目标负载', color='blue', linestyle=':', linewidth=2)
    axs[3].plot(time_array, error * 10, label='误差 x10 (右轴)', color='gray', linestyle='-.')  # 放大10倍显示误差
    axs[3].set_xlabel('时间 (s)')
    axs[3].set_ylabel('功率 (W)')
    axs[3].set_title('4. 系统功率平衡验证 (总输出应完美跟踪目标)')
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印最终状态
    print(f"\\n--- 仿真结束 ---")
    print(f"电池最终SOC: {battery.current_soc:.2%}")
    print(f"飞轮最终SOC: {fess.get_soc():.2%}")
    print(f"电容最终电压: {supercap.voltage:.2f}V")
    print(f"功率平衡误差 (RMS): {np.sqrt(np.mean(error ** 2)):.2f} W")


if __name__ == "__main__":
    run_hybrid_simulation()
