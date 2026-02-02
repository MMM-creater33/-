import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 1. 储能设备参数类 ====================
class StorageParameters:
    """存储所有储能设备的参数"""

    def __init__(self):
        # 参考电价和天然气价格文件中的数据
        self.electricity_price = {
            'valley': 0.21,  # 元/kWh (00:00-06:00, 11:00-13:00)
            'flat': 0.62,  # 元/kWh
            'peak': 1.12,  # 元/kWh (14:00-22:00)
            'sharp': 1.34  # 元/kWh (18:00-20:00)
        }

        self.gas_price = 3.6  # 元/立方米 (南京冬季工商业)
        self.gas_energy_density = 10  # kWh/立方米 (低热值)

        # 设备造价 (元/单位)
        self.capex = {
            'BESS': {'power': 800, 'energy': 0.7},  # 元/kW, 元/Wh -> 700元/kWh
            'SC': {'power': 2000, 'energy': 2.0},  # 元/kW, 元/Wh -> 2000元/kWh
            'FESS': {'power': 10000, 'energy': 0},  # 元/kW (能量成本已包含)
            'SMES': {'power': 0, 'energy': 65000},  # 元/kWh
            'CAES': {'power': 0, 'energy': 2750}  # 元/kWh
        }

        # 运维成本 (元/kWh/年)
        self.opex = {
            'BESS': 0.025,  # 2.5分/kWh
            'SC': 0.015,  # 1.5分/kWh
            'FESS': 0.02,  # 2.0分/kWh
            'SMES': 0.05,  # 5.0分/kWh (冷却成本高)
            'CAES': 0.01  # 1.0分/kWh
        }

        # 效率参数 (充放电)
        self.efficiency = {
            'BESS': {'charge': 0.95, 'discharge': 0.95},
            'SC': {'charge': 0.98, 'discharge': 0.98},
            'FESS': {'charge': 0.92, 'discharge': 0.92},
            'SMES': {'charge': 0.97, 'discharge': 0.97},
            'CAES': {'charge': 0.85, 'discharge': 0.65}  # 放电效率包含发电效率
        }

        # 响应时间 (秒)
        self.response_time = {
            'SMES': 0.001,  # 1毫秒
            'SC': 0.01,  # 10毫秒
            'FESS': 0.1,  # 100毫秒
            'BESS': 1.0,  # 1秒
            'CAES': 10.0  # 10秒
        }

        # 自放电率 (%/小时)
        self.self_discharge = {
            'BESS': 0.1,
            'SC': 5.0,
            'FESS': 0.5,
            'SMES': 0.01,
            'CAES': 0.05
        }


# ==================== 2. 储能设备模型类 ====================
class StorageDevice:
    """通用储能设备基类"""

    def __init__(self, name, params, P_rated, E_capacity, SOC_init=0.5):
        self.name = name
        self.params = params
        self.P_rated = P_rated  # 额定功率 (kW)
        self.E_capacity = E_capacity  # 额定容量 (kWh)
        self.SOC = SOC_init  # 当前SOC (0-1)
        self.SOC_max = 0.9
        self.SOC_min = 0.1
        self.cost = 0  # 累计成本 (元)
        self.energy_stored = E_capacity * SOC_init  # 当前储能 (kWh)

    def charge(self, P_charge, dt):
        """充电过程"""
        if P_charge <= 0:
            return 0

        # 功率限制
        P_actual = min(P_charge, self.P_rated)

        # 考虑效率
        energy_input = P_actual * dt / 3600  # kWh
        energy_stored = energy_input * self.params.efficiency[self.name]['charge']

        # 容量限制
        available_space = self.E_capacity * self.SOC_max - self.energy_stored
        energy_stored = min(energy_stored, available_space)

        # 更新状态
        self.energy_stored += energy_stored
        self.SOC = self.energy_stored / self.E_capacity

        # 成本计算 (充电成本)
        self.cost += energy_input * self.params.electricity_price['valley']

        return P_actual

    def discharge(self, P_discharge, dt):
        """放电过程"""
        if P_discharge <= 0:
            return 0

        # 功率限制
        P_actual = min(P_discharge, self.P_rated)

        # 能量限制
        energy_available = (self.energy_stored - self.E_capacity * self.SOC_min) * 3600 / dt
        P_actual = min(P_actual, energy_available)

        if P_actual <= 0:
            return 0

        # 考虑效率
        energy_output = P_actual * dt / 3600  # kWh
        energy_used = energy_output / self.params.efficiency[self.name]['discharge']

        # 更新状态
        self.energy_stored -= energy_used
        self.SOC = self.energy_stored / self.E_capacity

        # 成本计算 (放电收益)
        self.cost -= energy_output * self.params.electricity_price['peak']

        return P_actual

    def idle_loss(self, dt):
        """闲置损耗"""
        loss_rate = self.params.self_discharge[self.name] / 100 / 3600  # 每秒损耗率
        self.energy_stored *= (1 - loss_rate * dt)
        self.SOC = self.energy_stored / self.E_capacity


# ==================== 3. PWM调制器类 ====================
class PWMModulator:
    """PWM调制器，用于分配高频脉冲"""

    def __init__(self, devices, params):
        self.devices = devices
        self.params = params
        self.duty_cycles = {}  # 各设备的占空比

    def calculate_duty_cycles(self, pulse_power, pulse_duration):
        """根据脉冲特性计算最优占空比"""
        # 按响应时间排序
        sorted_devices = sorted(self.devices.keys(),
                                key=lambda x: self.params.response_time[x])

        total_available_power = sum(self.devices[d].P_rated for d in sorted_devices)

        if pulse_power > total_available_power:
            print(f"警告：脉冲功率{pulse_power}kW超过总可用功率{total_available_power}kW")
            pulse_power = total_available_power

        # 分配策略：快速响应设备优先
        remaining_power = pulse_power
        for device_name in sorted_devices:
            device = self.devices[device_name]
            if remaining_power <= 0:
                self.duty_cycles[device_name] = 0
            else:
                # 设备能提供的最大功率
                max_power = min(device.P_rated, remaining_power)
                # 计算需要的占空比
                duty_cycle = max_power / device.P_rated if device.P_rated > 0 else 0
                self.duty_cycles[device_name] = min(duty_cycle, 1.0)
                remaining_power -= max_power

        return self.duty_cycles

    def apply_pwm(self, pulse_power, pulse_duration, control_period=0.1):
        """应用PWM控制"""
        duty_cycles = self.calculate_duty_cycles(pulse_power, pulse_duration)
        n_steps = int(pulse_duration / control_period)

        power_outputs = {name: [] for name in self.devices.keys()}

        for step in range(n_steps):
            for device_name, device in self.devices.items():
                duty = duty_cycles[device_name]
                # PWM调制：在控制周期内按占空比开关
                if step < int(duty * n_steps):
                    power = device.P_rated * duty
                    if pulse_power > 0:
                        actual_power = device.discharge(power, control_period)
                    else:
                        actual_power = device.charge(-power, control_period)
                    power_outputs[device_name].append(actual_power)
                else:
                    device.idle_loss(control_period)
                    power_outputs[device_name].append(0)

        return power_outputs


# ==================== 4. 上层MPC经济调度层 ====================
class UpperMPC:
    """上层经济调度层"""

    def __init__(self, devices, params, prediction_horizon=24):
        self.devices = devices
        self.params = params
        self.horizon = prediction_horizon
        self.energy_ref = {}  # 能量参考轨迹

    def economic_dispatch(self, load_forecast, price_forecast):
        """经济调度优化"""
        # 简化的经济调度：在低价时充电，高价时放电
        schedule = {name: np.zeros(self.horizon) for name in self.devices.keys()}

        for t in range(self.horizon):
            # 判断电价时段
            if price_forecast[t] < 0.3:  # 低谷电价
                # 分配充电任务（优先CAES和BESS）
                for name in ['CAES', 'BESS', 'FESS', 'SC', 'SMES']:
                    if name in self.devices:
                        schedule[name][t] = -self.devices[name].P_rated * 0.8
            elif price_forecast[t] > 0.8:  # 高峰电价
                # 分配放电任务
                for name in ['SMES', 'SC', 'FESS', 'BESS', 'CAES']:
                    if name in self.devices:
                        schedule[name][t] = self.devices[name].P_rated * 0.8

        # 计算能量参考轨迹
        for name, device in self.devices.items():
            E_ref = np.zeros(self.horizon)
            E_current = device.energy_stored
            for t in range(self.horizon):
                E_current += schedule[name][t] / 3600  # 假设1小时时间步长
                E_ref[t] = max(min(E_current, device.E_capacity * device.SOC_max),
                               device.E_capacity * device.SOC_min)
            self.energy_ref[name] = E_ref

        return schedule, self.energy_ref


# ==================== 5. 下层MPC实时平衡层 ====================
class LowerMPC:
    """下层实时平衡层"""

    def __init__(self, devices, params, control_horizon=10):
        self.devices = devices
        self.params = params
        self.horizon = control_horizon
        self.pwm = PWMModulator(devices, params)

    def real_time_balance(self, power_imbalance, duration, energy_ref):
        """实时功率平衡"""
        # 使用PWM调制分配脉冲
        power_outputs = self.pwm.apply_pwm(power_imbalance, duration)

        # 计算与参考轨迹的偏差
        deviations = {}
        for name, device in self.devices.items():
            if name in energy_ref:
                current_energy = device.energy_stored
                ref_energy = energy_ref[name][0] if len(energy_ref[name]) > 0 else current_energy
                deviations[name] = abs(current_energy - ref_energy)

        return power_outputs, deviations


# ==================== 6. 主控制系统 ====================
class HybridEnergySystem:
    """混合储能系统主控制器"""

    def __init__(self):
        self.params = StorageParameters()

        # 初始化储能设备（根据任务需求配置）
        # 假设总功率配置：13MW发电，需要平滑7MW脉冲
        self.devices = {
            'BESS': StorageDevice('BESS', self.params,
                                  P_rated=2000,  # 2MW
                                  E_capacity=500),  # 500kWh
            'SC': StorageDevice('SC', self.params,
                                P_rated=1000,  # 1MW
                                E_capacity=50),  # 50kWh
            'FESS': StorageDevice('FESS', self.params,
                                  P_rated=1500,  # 1.5MW
                                  E_capacity=100),  # 100kWh
            'SMES': StorageDevice('SMES', self.params,
                                  P_rated=2000,  # 2MW
                                  E_capacity=20),  # 20kWh
            'CAES': StorageDevice('CAES', self.params,
                                  P_rated=3000,  # 3MW
                                  E_capacity=1000)  # 1000kWh
        }

        self.total_power = sum(d.P_rated for d in self.devices.values())  # 9.5MW

        # 初始化控制器
        self.upper_mpc = UpperMPC(self.devices, self.params)
        self.lower_mpc = LowerMPC(self.devices, self.params)

        # 记录数据
        self.history = {
            'time': [],
            'load': [],
            'generation': [],
            'imbalance': [],
            'total_storage_power': [],
            'SOC': {name: [] for name in self.devices.keys()},
            'power': {name: [] for name in self.devices.keys()},
            'cost': {name: [] for name in self.devices.keys()}
        }

    def simulate_pulse(self, base_load=10000, pulse_peak=20000,
                       pulse_duration=10, total_time=100):
        """模拟脉冲场景"""
        # 电厂发电功率：13MW恒定
        generation = 13000  # kW

        # 时间设置
        time_steps = np.arange(0, total_time, 0.1)  # 0.1秒步长
        pulse_start = 20  # 第20秒开始脉冲
        pulse_end = pulse_start + pulse_duration

        # 初始化记录
        for t in time_steps:
            # 计算负荷
            if pulse_start <= t < pulse_end:
                load = pulse_peak  # 脉冲期间20MW
            else:
                load = base_load  # 基本负荷10MW

            # 计算功率不平衡
            imbalance = load - generation  # 正表示缺电，需要储能放电

            # 上层经济调度（每小时更新一次）
            if t % 3600 < 0.1:  # 每小时
                # 简化的价格预测
                hour = int(t / 3600) % 24
                price_forecast = [0.21 if 0 <= h < 6 or 11 <= h < 13 else
                                  1.12 if 14 <= h < 22 else 0.62 for h in range(24)]
                load_forecast = [base_load] * 24
                schedule, energy_ref = self.upper_mpc.economic_dispatch(
                    load_forecast, price_forecast)

            # 下层实时平衡
            if abs(imbalance) > 100:  # 不平衡超过100kW时触发
                power_outputs, deviations = self.lower_mpc.real_time_balance(
                    imbalance, 0.1, energy_ref)
            else:
                power_outputs = {name: [0] for name in self.devices.keys()}

            # 计算总储能功率
            total_storage_power = sum(p[0] if len(p) > 0 else 0 for p in power_outputs.values())

            # 记录数据
            self.history['time'].append(t)
            self.history['load'].append(load)
            self.history['generation'].append(generation)
            self.history['imbalance'].append(imbalance)
            self.history['total_storage_power'].append(total_storage_power)

            for name, device in self.devices.items():
                self.history['SOC'][name].append(device.SOC)
                self.history['power'][name].append(
                    power_outputs[name][0] if len(power_outputs[name]) > 0 else 0)
                self.history['cost'][name].append(device.cost)

        return self.history

    def calculate_performance(self):
        """计算性能指标"""
        # 脉冲平滑度
        imbalance = np.array(self.history['imbalance'])
        storage_power = np.array(self.history['total_storage_power'])
        smoothing_error = np.sqrt(np.mean((imbalance + storage_power) ** 2))

        # 成本统计
        total_cost = sum(self.history['cost'][name][-1] for name in self.devices.keys())

        # SOC变化范围
        soc_ranges = {}
        for name in self.devices.keys():
            soc_values = self.history['SOC'][name]
            soc_ranges[name] = {
                'min': min(soc_values),
                'max': max(soc_values),
                'avg': np.mean(soc_values)
            }

        # 设备利用率
        utilization = {}
        for name in self.devices.keys():
            power_values = np.abs(self.history['power'][name])
            utilization[name] = np.mean(power_values) / self.devices[name].P_rated * 100

        return {
            'smoothing_error': smoothing_error,
            'total_cost': total_cost,
            'soc_ranges': soc_ranges,
            'utilization': utilization
        }

    def plot_results(self):
        """绘制结果图表"""
        fig = plt.figure(figsize=(16, 12))

        # 1. 功率平衡图
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.history['time'], self.history['load'], 'r-', label='用户负荷', linewidth=2)
        ax1.plot(self.history['time'], self.history['generation'], 'b-',
                 label='电厂发电', linewidth=2)
        ax1.plot(self.history['time'], self.history['total_storage_power'], 'g-',
                 label='储能总功率', linewidth=2)
        ax1.fill_between(self.history['time'], 0, self.history['imbalance'],
                         alpha=0.3, label='功率不平衡')
        ax1.set_xlabel('时间 (秒)')
        ax1.set_ylabel('功率 (kW)')
        ax1.set_title('系统功率平衡')
        ax1.legend()
        ax1.grid(True)

        # 2. 各储能设备功率
        ax2 = plt.subplot(3, 2, 2)
        for name in self.devices.keys():
            ax2.plot(self.history['time'], self.history['power'][name],
                     label=name, linewidth=1.5)
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('功率 (kW)')
        ax2.set_title('各储能设备功率输出')
        ax2.legend()
        ax2.grid(True)

        # 3. SOC变化
        ax3 = plt.subplot(3, 2, 3)
        for name in self.devices.keys():
            ax3.plot(self.history['time'], self.history['SOC'][name],
                     label=name, linewidth=1.5)
        ax3.set_xlabel('时间 (秒)')
        ax3.set_ylabel('SOC')
        ax3.set_title('各储能设备SOC变化')
        ax3.legend()
        ax3.grid(True)

        # 4. 成本累计
        ax4 = plt.subplot(3, 2, 4)
        for name in self.devices.keys():
            ax4.plot(self.history['time'], self.history['cost'][name],
                     label=name, linewidth=1.5)
        ax4.set_xlabel('时间 (秒)')
        ax4.set_ylabel('成本 (元)')
        ax4.set_title('各储能设备成本累计')
        ax4.legend()
        ax4.grid(True)

        # 5. 功率分配占比饼图
        ax5 = plt.subplot(3, 2, 5)
        total_energy = {}
        for name in self.devices.keys():
            total_energy[name] = np.sum(np.abs(self.history['power'][name]))

        if sum(total_energy.values()) > 0:
            labels = list(total_energy.keys())
            sizes = list(total_energy.values())
            ax5.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax5.set_title('各储能设备能量分配占比')
        else:
            ax5.text(0.5, 0.5, '无能量交换', ha='center', va='center')
            ax5.set_title('各储能设备能量分配占比')

        # 6. 脉冲平滑效果放大图
        ax6 = plt.subplot(3, 2, 6)
        pulse_start, pulse_end = 20, 30
        idx = [i for i, t in enumerate(self.history['time'])
               if pulse_start <= t <= pulse_end]

        if idx:
            time_pulse = [self.history['time'][i] for i in idx]
            load_pulse = [self.history['load'][i] for i in idx]
            net_pulse = [self.history['generation'][i] +
                         self.history['total_storage_power'][i] for i in idx]

            ax6.plot(time_pulse, load_pulse, 'r-', label='原始负荷', linewidth=2)
            ax6.plot(time_pulse, net_pulse, 'g--', label='平滑后负荷', linewidth=2)
            ax6.fill_between(time_pulse, load_pulse, net_pulse, alpha=0.3,
                             label='平滑量')
            ax6.set_xlabel('时间 (秒)')
            ax6.set_ylabel('功率 (kW)')
            ax6.set_title('脉冲平滑效果（20-30秒放大）')
            ax6.legend()
            ax6.grid(True)

        plt.tight_layout()
        plt.show()

    def print_report(self):
        """打印详细报告"""
        print("=" * 60)
        print("混合储能系统仿真报告")
        print("=" * 60)

        print("\n1. 系统配置:")
        print("-" * 40)
        total_power = sum(d.P_rated for d in self.devices.values())
        total_energy = sum(d.E_capacity for d in self.devices.values())
        print(f"总功率容量: {total_power / 1000:.2f} MW")
        print(f"总能量容量: {total_energy / 1000:.2f} MWh")

        print("\n2. 各设备配置:")
        print("-" * 40)
        for name, device in self.devices.items():
            print(f"{name}: {device.P_rated / 1000:.2f} MW, {device.E_capacity / 1000:.2f} MWh")

        print("\n3. 性能指标:")
        print("-" * 40)
        perf = self.calculate_performance()
        print(f"脉冲平滑误差: {perf['smoothing_error']:.2f} kW")
        print(f"总运行成本: {perf['total_cost']:.2f} 元")

        print("\n4. 设备利用率 (%):")
        print("-" * 40)
        for name, util in perf['utilization'].items():
            print(f"{name}: {util:.2f}%")

        print("\n5. SOC变化范围:")
        print("-" * 40)
        for name, soc_range in perf['soc_ranges'].items():
            print(f"{name}: {soc_range['min']:.3f} - {soc_range['max']:.3f} (平均: {soc_range['avg']:.3f})")

        print("\n6. 脉冲消除策略:")
        print("-" * 40)
        print("响应优先级: SMES(1ms) → SC(10ms) → FESS(100ms) → BESS(1s) → CAES(10s)")
        print("分配原则:")
        print("  - 高频脉冲 (<1s): SMES + SC 主承担")
        print("  - 中频脉冲 (1-10s): FESS + BESS 辅助")
        print("  - 低频脉冲 (>10s): CAES + BESS 主承担")
        print("  - PWM调制: 根据响应速度分配占空比")

        print("\n7. 经济性分析:")
        print("-" * 40)
        print(f"电价策略: 谷电{self.params.electricity_price['valley']}元/kWh充电")
        print(f"          峰电{self.params.electricity_price['peak']}元/kWh放电")
        print(f"天然气价: {self.params.gas_price}元/立方米 (用于CAES)")

        # 计算投资回收期
        total_investment = 0
        for name, device in self.devices.items():
            capex_power = self.params.capex[name]['power'] * device.P_rated
            capex_energy = self.params.capex[name]['energy'] * device.E_capacity * 1000
            total_investment += capex_power + capex_energy

        daily_profit = -perf['total_cost']  # 负成本表示收益
        if daily_profit > 0:
            payback_years = total_investment / (daily_profit * 365)
            print(f"\n总投资: {total_investment / 1e6:.2f} 百万元")
            print(f"日收益: {daily_profit:.2f} 元")
            print(f"投资回收期: {payback_years:.1f} 年")
        else:
            print("当前配置下系统无经济收益")


# ==================== 7. 主执行程序 ====================
if __name__ == "__main__":
    # 创建混合储能系统
    hes = HybridEnergySystem()

    # 模拟脉冲场景
    print("开始仿真...")
    history = hes.simulate_pulse(
        base_load=10000,  # 10MW基本负荷
        pulse_peak=20000,  # 20MW脉冲峰值
        pulse_duration=10,  # 10秒持续时间
        total_time=100  # 100秒仿真时间
    )

    # 打印报告
    hes.print_report()

    # 绘制图表
    print("\n生成图表...")
    hes.plot_results()

    # 保存数据到CSV
    df_data = pd.DataFrame({
        'time': history['time'],
        'load_kW': history['load'],
        'generation_kW': history['generation'],
        'imbalance_kW': history['imbalance'],
        'total_storage_kW': history['total_storage_power']
    })

    for name in hes.devices.keys():
        df_data[f'{name}_SOC'] = history['SOC'][name]
        df_data[f'{name}_power_kW'] = history['power'][name]
        df_data[f'{name}_cost_yuan'] = history['cost'][name]

    df_data.to_csv('hybrid_energy_system_results.csv', index=False, encoding='utf-8-sig')
    print("数据已保存到: hybrid_energy_system_results.csv")
