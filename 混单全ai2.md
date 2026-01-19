import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.optimize import minimize

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 1. 各储能系统类（保持原样） ====================

class CAESSystem:
    def __init__(self, M_air_max=1000, M_air_min=100, P_max=100):
        self.M_air_max = M_air_max
        self.M_air_min = M_air_min
        self.P_max = P_max
        self.M_air_current = (M_air_max + M_air_min) / 2
        self.history = []
        self.eta = 0.85
        self.soc_history = []

    def step(self, P_cmd, dt=1):
        P_act = np.clip(P_cmd, -self.P_max, self.P_max)
        m_change = P_act * dt / (self.eta * 3600)
        M_air_next = self.M_air_current + m_change

        if M_air_next > self.M_air_max:
            M_air_next = self.M_air_max
            P_act = 0
        elif M_air_next < self.M_air_min:
            M_air_next = self.M_air_min
            P_act = 0

        self.M_air_current = M_air_next
        soc = (self.M_air_current - self.M_air_min) / (self.M_air_max - self.M_air_min)
        self.soc_history.append(soc)
        self.history.append(P_act)
        return P_act

    def get_soc(self):
        return (self.M_air_current - self.M_air_min) / (self.M_air_max - self.M_air_min)


class SupercapacitorSystem:
    def __init__(self, capacitance=3000, max_voltage=2.7, min_voltage=1.0, P_max=50):
        self.capacitance = capacitance
        self.max_voltage = max_voltage
        self.min_voltage = min_voltage
        self.P_max = P_max
        self.voltage = max_voltage
        self.history = []
        self.R_esr = 0.01
        self.soc_history = []

    def step(self, P_cmd, dt=1):
        P_act = np.clip(P_cmd, -self.P_max, self.P_max)

        if self.voltage > 0:
            I = P_act / self.voltage
        else:
            I = 0

        self.voltage -= I * self.R_esr * dt
        self.voltage = np.clip(self.voltage, self.min_voltage, self.max_voltage)

        soc = (self.voltage - self.min_voltage) / (self.max_voltage - self.min_voltage)
        self.soc_history.append(soc)
        self.history.append(P_act)
        return P_act

    def get_soc(self):
        return (self.voltage - self.min_voltage) / (self.max_voltage - self.min_voltage)


class FlywheelSystem:
    def __init__(self, max_energy=1e6, efficiency=0.9, P_max=30):
        self.max_energy = max_energy
        self.efficiency = efficiency
        self.P_max = P_max
        self.energy = max_energy / 2
        self.history = []
        self.soc_history = []

    def step(self, P_cmd, dt=1):
        P_act = np.clip(P_cmd, -self.P_max, self.P_max)

        if P_act > 0:  # 放电
            energy_out = P_act * dt / self.efficiency
        else:  # 充电
            energy_out = P_act * dt * self.efficiency

        self.energy -= energy_out
        self.energy = np.clip(self.energy, 0, self.max_energy)

        soc = self.energy / self.max_energy
        self.soc_history.append(soc)
        self.history.append(P_act)
        return P_act

    def get_soc(self):
        return self.energy / self.max_energy


class SMESSystem:
    def __init__(self, inductance=100, max_current=1000, P_max=40):
        self.inductance = inductance
        self.max_current = max_current
        self.P_max = P_max
        self.current = max_current / 2
        self.history = []
        self.soc_history = []

    def step(self, P_cmd, dt=1):
        P_act = np.clip(P_cmd, -self.P_max, self.P_max)

        if self.current > 0:
            di = P_act * dt / (self.inductance * self.current)
        else:
            di = 0

        self.current += di
        self.current = np.clip(self.current, 0, self.max_current)

        soc = self.current / self.max_current
        self.soc_history.append(soc)
        self.history.append(P_act)
        return P_act

    def get_soc(self):
        return self.current / self.max_current


class BESSSystem:
    def __init__(self, capacity=2000, max_power=80):
        self.capacity = capacity  # Wh
        self.max_power = max_power
        self.soc = 0.5
        self.history = []
        self.soc_history = []

    def step(self, P_cmd, dt=1):
        P_act = np.clip(P_cmd, -self.max_power, self.max_power)
        delta_energy = P_act * dt / 3600  # Wh
        self.soc -= delta_energy / self.capacity
        self.soc = np.clip(self.soc, 0, 1)

        self.soc_history.append(self.soc)
        self.history.append(P_act)
        return P_act

    def get_soc(self):
        return self.soc


# ==================== 2. 改进的混合储能系统协同控制器 ====================

class AdvancedHybridEnergyStorageSystem:
    def __init__(self):
        # 初始化各储能单元
        self.caes = CAESSystem(P_max=80)
        self.sc = SupercapacitorSystem(P_max=40)
        self.fw = FlywheelSystem(P_max=30)
        self.smes = SMESSystem(P_max=40)
        self.bess = BESSSystem(max_power=60)

        self.storages = [self.caes, self.sc, self.fw, self.smes, self.bess]

        # 响应特性分类
        self.slow_storages = [self.caes, self.bess]  # 响应慢，适合低频分量
        self.fast_storages = [self.sc, self.fw, self.smes]  # 响应快，适合高频分量

        # 历史记录
        self.power_history = []
        self.soc_history = []
        self.balance_history = []
        self.time_history = []
        self.demand_history = []

        # 控制参数
        self.low_pass_cutoff = 0.02  # 低通滤波器截止频率
        self.high_pass_cutoff = 0.1  # 高通滤波器截止频率
        self.soc_target = 0.5  # SOC目标值

        # 平滑滤波器
        self.filter_order = 3

    def butter_lowpass(self, cutoff, fs, order=3):
        """创建低通滤波器"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_highpass(self, cutoff, fs, order=3):
        """创建高通滤波器"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def filter_signal(self, signal, cutoff, fs, filter_type='low'):
        """滤波信号"""
        # 检查信号长度是否足够
        min_length = 12  # 最小信号长度要求

        if len(signal) < min_length:
            # 信号太短，返回简单处理结果
            if filter_type == 'low':
                return np.array(signal)  # 返回原始信号作为低频
            else:
                return np.zeros_like(signal)  # 返回零作为高频

        try:
            if filter_type == 'low':
                b, a = self.butter_lowpass(cutoff, fs, 2)  # 降低阶数
            else:
                b, a = self.butter_highpass(cutoff, fs, 2)  # 降低阶数

            filtered = filtfilt(b, a, signal)
            return filtered
        except Exception as e:
            # 滤波失败，返回简化结果
            if filter_type == 'low':
                return np.array(signal)
            else:
                return np.zeros_like(signal)

    def power_decomposition(self, demand_signal, fs=1):
        """功率分解：将需求功率分解为低频、中频和高频分量"""
        # 检查信号长度
        if len(demand_signal) < 20:
            # 信号太短，使用简化分解
            # 使用移动平均作为低频分量
            window_size = min(5, len(demand_signal))
            if window_size > 1:
                low_freq = np.convolve(demand_signal, np.ones(window_size) / window_size, mode='same')
            else:
                low_freq = np.array(demand_signal)

            # 高频分量
            high_freq = demand_signal - low_freq

            # 中频分量设为0
            mid_freq = np.zeros_like(demand_signal)
        else:
            # 1. 低通滤波得到低频分量（慢速变化部分）
            low_freq = self.filter_signal(demand_signal, self.low_pass_cutoff, fs, 'low')

            # 2. 高通滤波得到高频分量（快速变化部分）
            high_freq = self.filter_signal(demand_signal, self.high_pass_cutoff, fs, 'high')

            # 3. 中频分量 = 总需求 - 低频 - 高频
            mid_freq = np.array(demand_signal) - low_freq - high_freq

        return low_freq, mid_freq, high_freq

    def allocate_power_smart(self, P_demand, t, demand_series):
        """智能功率分配策略"""
        # 获取当前SOC状态
        socs = [s.get_soc() for s in self.storages]

        # 响应速度权重：响应越快，权重越大
        response_weights = [0.2, 0.3, 0.25, 0.15, 0.1]  # CAES, SC, FW, SMES, BESS

        # SOC平衡权重：SOC偏离目标越多，权重调整越大
        soc_deviations = [abs(soc - self.soc_target) for soc in socs]
        soc_weights = [1 + dev for dev in soc_deviations]  # 偏离越大，权重越大

        # 综合权重
        combined_weights = np.array(response_weights) * np.array(soc_weights)
        combined_weights = combined_weights / np.sum(combined_weights)

        # 根据功率需求的正负调整权重
        if P_demand > 0:  # 放电需求
            # 对于放电，SOC高的应该多放电
            soc_factor = np.array(socs) / np.sum(socs)
            combined_weights = combined_weights * soc_factor
        else:  # 充电需求
            # 对于充电，SOC低的应该多充电
            soc_factor = (1 - np.array(socs)) / np.sum(1 - np.array(socs))
            combined_weights = combined_weights * soc_factor

        # 归一化
        combined_weights = combined_weights / np.sum(combined_weights)

        # 功率分配
        allocations = P_demand * combined_weights

        return allocations

    def allocate_frequency_components(self, low_freq, mid_freq, high_freq):
        """频率分量分配"""
        # 低频分量分配给慢速储能
        low_alloc = np.array([0.6, 0.4])  # CAES和BESS分配比例
        low_power_slow = low_freq * low_alloc[0] / 2
        low_power_bess = low_freq * low_alloc[1] / 2

        # 中频分量均匀分配
        mid_power_all = mid_freq / len(self.storages)

        # 高频分量分配给快速储能
        high_alloc = np.array([0.4, 0.35, 0.25])  # SC, FW, SMES分配比例
        high_power_sc = high_freq * high_alloc[0]
        high_power_fw = high_freq * high_alloc[1]
        high_power_smes = high_freq * high_alloc[2]

        # 组合分配
        allocations = np.zeros(5)
        allocations[0] = low_power_slow + mid_power_all  # CAES
        allocations[1] = high_power_sc + mid_power_all  # SC
        allocations[2] = high_power_fw + mid_power_all  # FW
        allocations[3] = high_power_smes + mid_power_all  # SMES
        allocations[4] = low_power_bess + mid_power_all  # BESS

        return allocations

    def step(self, P_demand, t, demand_series=None):
        """执行一步协同控制"""
        # 如果提供了需求序列，使用频率分解方法
        if demand_series is not None and len(demand_series) > 15:
            # 获取最近一段时间的需求序列进行频率分解
            window_size = min(50, len(demand_series))
            recent_demand = demand_series[-window_size:]

            # 频率分解
            fs = 1  # 采样频率
            low_freq, mid_freq, high_freq = self.power_decomposition(recent_demand, fs)

            # 使用当前时刻的分解结果
            if len(low_freq) > 0:
                current_low = low_freq[-1]
                current_mid = mid_freq[-1]
                current_high = high_freq[-1]

                # 频率分量分配
                allocations = self.allocate_frequency_components(
                    current_low, current_mid, current_high
                )
            else:
                allocations = self.allocate_power_smart(P_demand, t, demand_series)
        else:
            allocations = self.allocate_power_smart(P_demand, t, [])

        # 各储能单元执行
        total_output = 0
        socs = []

        for i, (storage, P_alloc) in enumerate(zip(self.storages, allocations)):
            # 考虑功率限制
            if hasattr(storage, 'P_max'):
                P_max = storage.P_max
            elif hasattr(storage, 'max_power'):
                P_max = storage.max_power
            else:
                P_max = 100  # 默认值

            P_alloc = np.clip(P_alloc, -P_max, P_max)

            P_out = storage.step(P_alloc)
            total_output += P_out
            socs.append(storage.get_soc())

        # 计算平衡误差
        balance_error = P_demand - total_output

        # 记录
        self.power_history.append(total_output)
        self.soc_history.append(socs)
        self.balance_history.append(balance_error)
        self.time_history.append(t)
        self.demand_history.append(P_demand)

        return total_output, balance_error

    def run_simulation(self, steps=200, demand_signal=None):
        """运行仿真"""
        if demand_signal is None:
            # 生成更复杂的测试信号（模拟真实波动）
            t = np.linspace(0, 10, steps)
            # 多种频率成分叠加
            demand_signal = (
                    50 * np.sin(0.5 * t) +  # 低频基础
                    30 * np.sin(3 * t) +  # 中频波动
                    20 * np.sin(10 * t) +  # 高频波动
                    15 * np.random.randn(steps) +  # 随机噪声
                    10 * np.sin(20 * t)  # 更高频波动
            )

        for t in range(steps):
            P_demand = demand_signal[t]
            self.step(P_demand, t, demand_signal[:t + 1])

    def plot_detailed_results(self):
        """绘制详细结果图"""
        fig, axes = plt.subplots(5, 1, figsize=(16, 18))

        time_array = np.array(self.time_history)
        power_array = np.array(self.power_history)
        demand_array = np.array(self.demand_history)

        # 1. 原始需求 vs 实际出力
        axes[0].plot(time_array, demand_array, 'b-', alpha=0.6, label='需求功率', linewidth=1.5)
        axes[0].plot(time_array, power_array, 'r-', label='实际出力', linewidth=1.5)
        axes[0].fill_between(time_array, demand_array, power_array, alpha=0.2, color='gray')
        axes[0].set_ylabel('功率 (kW)')
        axes[0].set_title('需求功率 vs 实际出力')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 功率平衡误差
        error_array = np.array(self.balance_history)
        axes[1].plot(time_array, error_array, 'g-', label='平衡误差', linewidth=1)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].fill_between(time_array, error_array, 0, where=(error_array >= 0),
                             alpha=0.3, color='red', label='正误差')
        axes[1].fill_between(time_array, error_array, 0, where=(error_array < 0),
                             alpha=0.3, color='blue', label='负误差')
        axes[1].set_ylabel('误差 (kW)')
        axes[1].set_title('功率平衡误差分布')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. 各储能单元SOC变化
        soc_array = np.array(self.soc_history)
        labels = ['CAES', '超级电容', '飞轮', 'SMES', '锂电']
        colors = ['blue', 'green', 'red', 'purple', 'orange']

        for i in range(soc_array.shape[1]):
            axes[2].plot(time_array, soc_array[:, i], color=colors[i],
                         label=labels[i], linewidth=1.5)

        axes[2].axhline(y=0.2, color='gray', linestyle='--', alpha=0.5)
        axes[2].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_ylabel('SOC')
        axes[2].set_title('各储能单元SOC变化')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # 4. 平滑效果对比
        # 计算移动平均平滑
        if len(power_array) > 1:
            window_size = min(10, len(power_array))
            if window_size > 1:
                smoothed = np.convolve(power_array, np.ones(window_size) / window_size, mode='same')

                # Savitzky-Golay滤波
                if len(power_array) > 11:
                    try:
                        sg_smoothed = savgol_filter(power_array, 11, 3)
                    except:
                        sg_smoothed = smoothed
                else:
                    sg_smoothed = smoothed

                axes[3].plot(time_array, power_array, 'b-', alpha=0.4, label='原始出力', linewidth=1)
                axes[3].plot(time_array, smoothed, 'r-', label=f'移动平均 (窗宽={window_size})', linewidth=2)
                axes[3].plot(time_array, sg_smoothed, 'g-', label='Savitzky-Golay滤波', linewidth=2)
                axes[3].axhline(y=np.mean(power_array), color='orange', linestyle='--',
                                label='平均值', linewidth=2)
                axes[3].set_ylabel('功率 (kW)')
                axes[3].set_title('功率平滑效果对比')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)

        # 5. 统计指标
        axes[4].axis('off')

        # 计算统计指标
        if len(power_array) > 0:
            mean_power = np.mean(power_array)
            std_power = np.std(power_array)
            max_error = np.max(np.abs(error_array))
            mean_error = np.mean(np.abs(error_array))

            # 计算平滑度指标
            if len(power_array) > 1:
                power_diff = np.diff(power_array)
                smoothness = np.mean(np.abs(power_diff))
            else:
                smoothness = 0
        else:
            mean_power = std_power = max_error = mean_error = smoothness = 0

        # 显示统计信息
        stats_text = (
            f"=== 混合储能系统性能统计 ===\n\n"
            f"平均出力: {mean_power:.2f} kW\n"
            f"出力标准差: {std_power:.2f} kW\n"
            f"出力变异系数: {std_power / abs(mean_power + 1e-10):.3f}\n"
            f"最大平衡误差: {max_error:.2f} kW\n"
            f"平均绝对误差: {mean_error:.2f} kW\n"
            f"功率变化平滑度: {smoothness:.3f}\n\n"
            f"各储能平均SOC:\n"
        )

        if len(soc_array) > 0 and soc_array.shape[1] == 5:
            for i, label in enumerate(labels):
                mean_soc = np.mean(soc_array[:, i])
                stats_text += f"  {label}: {mean_soc:.3f}\n"
        else:
            stats_text += "  SOC数据不可用\n"

        axes[4].text(0.05, 0.95, stats_text, transform=axes[4].transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show()

        return {
            'mean_power': mean_power,
            'std_power': std_power,
            'max_error': max_error,
            'mean_error': mean_error,
            'smoothness': smoothness
        }


# ==================== 3. 主程序 ====================

if __name__ == "__main__":
    print("=== 改进的混合储能系统仿真 ===")
    print("采用多层次功率分解与协同控制策略...")

    # 创建改进的混合储能系统
    hess = AdvancedHybridEnergyStorageSystem()

    # 生成复杂的测试需求信号
    steps = 300
    t = np.linspace(0, 15, steps)

    # 多频率成分叠加
    demand_signal = (
            60 * np.sin(0.3 * t) +  # 低频基础波动
            40 * np.sin(2 * t) * np.exp(-0.05 * t) +  # 衰减的中频波动
            30 * np.sin(5 * t) +  # 中高频波动
            20 * np.sin(15 * t) +  # 高频波动
            25 * np.random.randn(steps) * (1 + 0.1 * t) +  # 时变噪声
            10 * np.sign(np.sin(0.8 * t))  # 方波分量
    )

    # 运行仿真
    hess.run_simulation(steps=steps, demand_signal=demand_signal)

    # 绘制详细结果
    stats = hess.plot_detailed_results()

    # 输出关键性能指标
    print(f"\n=== 关键性能指标 ===")
    print(f"出力波动标准差: {stats['std_power']:.2f} kW")
    print(f"功率平滑度: {stats['smoothness']:.3f}")
    print(f"最大平衡误差: {stats['max_error']:.2f} kW")

    # 计算改进效果
    print(f"\n=== 平滑效果评估 ===")
    original_std = np.std(demand_signal)
    output_std = stats['std_power']
    improvement = (original_std - output_std) / original_std * 100
    print(f"原始需求标准差: {original_std:.2f} kW")
    print(f"混合系统出力标准差: {output_std:.2f} kW")
    print(f"波动减少: {improvement:.1f}%")
