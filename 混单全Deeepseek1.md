"""
基于分层随机MPC的混合储能系统协同调度策略
园区级微电网（12MW，24小时运行）
整合：BESS, FESS, SC, SMES, CAES
作者：能源系统优化组
日期：2026年1月
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy import optimize
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 1. 储能单元数学模型 ====================
@dataclass
class StorageParameters:
    """储能单元参数配置"""
    # BESS参数
    bess_capacity: float = 3000  # kWh
    bess_power: float = 2000  # kW
    bess_soc_min: float = 0.1
    bess_soc_max: float = 0.9
    bess_eff_charge: float = 0.95
    bess_eff_discharge: float = 0.95
    bess_cost_per_kwh: float = 0.8  # 元/Wh

    # FESS参数
    fess_energy: float = 500  # kWh
    fess_power: float = 3000  # kW
    fess_soc_min: float = 0.2
    fess_soc_max: float = 0.95
    fess_eff: float = 0.92
    fess_cost_per_kw: float = 10000  # 元/kW

    # SC参数
    sc_capacity: float = 100  # kWh
    sc_power: float = 5000  # kW
    sc_soc_min: float = 0.3
    sc_soc_max: float = 0.97
    sc_eff: float = 0.98
    sc_cost_per_kwh: float = 2.0  # 元/Wh

    # SMES参数
    smes_energy: float = 200  # kWh
    smes_power: float = 4000  # kW
    smes_soc_min: float = 0.4
    smes_soc_max: float = 0.99
    smes_eff: float = 0.96
    smes_cost_per_kwh: float = 60000  # 元/kWh

    # CAES参数
    caes_energy: float = 10000  # kWh
    caes_power_charge: float = 4000  # kW
    caes_power_discharge: float = 6000  # kW
    caes_soc_min: float = 0.15
    caes_soc_max: float = 0.85
    caes_eff_charge: float = 0.7
    caes_eff_discharge: float = 0.8
    caes_cost_per_kwh: float = 2.5  # 元/kWh

    # 经济参数
    electricity_price_peak: float = 1.12  # 元/kWh
    electricity_price_valley: float = 0.21  # 元/kWh
    electricity_price_normal: float = 0.62  # 元/kWh
    gas_price: float = 3.6  # 元/立方米
    gas_heat_value: float = 10  # kWh/m³


class BESS:
    """锂电池储能系统"""

    def __init__(self, params: StorageParameters):
        self.capacity = params.bess_capacity  # kWh
        self.max_power = params.bess_power  # kW
        self.soc_min = params.bess_soc_min
        self.soc_max = params.bess_soc_max
        self.eff_charge = params.bess_eff_charge
        self.eff_discharge = params.bess_eff_discharge
        self.cycle_cost = 0.0002  # 元/循环

        self.soc = 0.5  # 初始SOC
        self.power = 0  # 当前功率
        self.energy = self.capacity * self.soc  # 当前能量
        self.operation_mode = 0  # -1:充电, 0:空闲, 1:放电

    def update(self, power_setpoint: float, dt: float = 0.25) -> float:
        """更新BESS状态"""
        actual_power = 0

        # 功率限幅
        if power_setpoint > 0:  # 放电
            max_discharge = min(self.max_power,
                                self.energy * self.eff_discharge / dt)
            actual_power = min(power_setpoint, max_discharge)
            self.operation_mode = 1
        elif power_setpoint < 0:  # 充电
            max_charge = min(self.max_power,
                             (self.capacity * self.soc_max - self.energy) /
                             (self.eff_charge * dt))
            actual_power = max(power_setpoint, -max_charge)
            self.operation_mode = -1
        else:
            self.operation_mode = 0

        # 更新能量和SOC
        if actual_power > 0:  # 放电
            energy_change = actual_power * dt / self.eff_discharge
        else:  # 充电
            energy_change = actual_power * dt * self.eff_charge

        self.energy -= energy_change
        self.energy = np.clip(self.energy,
                              self.capacity * self.soc_min,
                              self.capacity * self.soc_max)
        self.soc = self.energy / self.capacity
        self.power = actual_power

        return actual_power


class FESS:
    """飞轮储能系统"""

    def __init__(self, params: StorageParameters):
        self.capacity = params.fess_energy
        self.max_power = params.fess_power
        self.soc_min = params.fess_soc_min
        self.soc_max = params.fess_soc_max
        self.efficiency = params.fess_eff
        self.self_discharge_rate = 0.001  # 自放电率

        self.soc = 0.5
        self.power = 0
        self.energy = self.capacity * self.soc

    def update(self, power_setpoint: float, dt: float = 0.25) -> float:
        """更新FESS状态"""
        # 计算自放电损失
        self.energy *= (1 - self.self_discharge_rate * dt)

        actual_power = 0
        if power_setpoint > 0:  # 放电
            max_discharge = min(self.max_power, self.energy / dt)
            actual_power = min(power_setpoint, max_discharge)
        elif power_setpoint < 0:  # 充电
            max_charge = min(self.max_power,
                             (self.capacity * self.soc_max - self.energy) / dt)
            actual_power = max(power_setpoint, -max_charge)

        # 更新能量和SOC
        self.energy -= actual_power * dt * (1 / self.efficiency if actual_power > 0 else self.efficiency)
        self.energy = np.clip(self.energy,
                              self.capacity * self.soc_min,
                              self.capacity * self.soc_max)
        self.soc = self.energy / self.capacity
        self.power = actual_power

        return actual_power


class Supercapacitor:
    """超级电容器"""

    def __init__(self, params: StorageParameters):
        self.capacity = params.sc_capacity
        self.max_power = params.sc_power
        self.soc_min = params.sc_soc_min
        self.soc_max = params.sc_soc_max
        self.efficiency = params.sc_eff

        self.soc = 0.5
        self.power = 0
        self.energy = self.capacity * self.soc

    def update(self, power_setpoint: float, dt: float = 0.25) -> float:
        """更新SC状态"""
        actual_power = 0
        if abs(power_setpoint) > self.max_power:
            actual_power = np.sign(power_setpoint) * self.max_power
        else:
            actual_power = power_setpoint

        # 更新能量
        if actual_power > 0:  # 放电
            energy_change = actual_power * dt / self.efficiency
        else:  # 充电
            energy_change = actual_power * dt * self.efficiency

        self.energy -= energy_change
        self.energy = np.clip(self.energy,
                              self.capacity * self.soc_min,
                              self.capacity * self.soc_max)
        self.soc = self.energy / self.capacity
        self.power = actual_power

        return actual_power


class SMES:
    """超导磁储能"""

    def __init__(self, params: StorageParameters):
        self.capacity = params.smes_energy
        self.max_power = params.smes_power
        self.soc_min = params.smes_soc_min
        self.soc_max = params.smes_soc_max
        self.efficiency = params.smes_eff

        self.soc = 0.6
        self.power = 0
        self.energy = self.capacity * self.soc

    def update(self, power_setpoint: float, dt: float = 0.25) -> float:
        """更新SMES状态"""
        # SMES响应速度极快，几乎无延迟
        actual_power = np.clip(power_setpoint, -self.max_power, self.max_power)

        # 更新能量
        if actual_power > 0:  # 放电
            energy_change = actual_power * dt / self.efficiency
        else:  # 充电
            energy_change = actual_power * dt * self.efficiency

        self.energy -= energy_change
        self.energy = np.clip(self.energy,
                              self.capacity * self.soc_min,
                              self.capacity * self.soc_max)
        self.soc = self.energy / self.capacity
        self.power = actual_power

        return actual_power


class CAES:
    """压缩空气储能"""

    def __init__(self, params: StorageParameters):
        self.capacity = params.caes_energy
        self.max_power_charge = params.caes_power_charge
        self.max_power_discharge = params.caes_power_discharge
        self.soc_min = params.caes_soc_min
        self.soc_max = params.caes_soc_max
        self.eff_charge = params.caes_eff_charge
        self.eff_discharge = params.caes_eff_discharge

        self.gas_price = params.gas_price
        self.gas_heat_value = params.gas_heat_value

        self.soc = 0.5
        self.power = 0
        self.energy = self.capacity * self.soc
        self.gas_consumption = 0

    def update(self, power_setpoint: float, dt: float = 0.25) -> Tuple[float, float]:
        """更新CAES状态，返回实际功率和天然气消耗"""
        actual_power = 0
        gas_used = 0

        if power_setpoint > 0:  # 放电（需要天然气）
            max_discharge = min(self.max_power_discharge,
                                self.energy / (dt * self.eff_discharge))
            actual_power = min(power_setpoint, max_discharge)

            # 计算天然气消耗（简化的燃料消耗模型）
            gas_used = (actual_power * dt * (1 - self.eff_discharge) /
                        (self.gas_heat_value * self.eff_discharge))

        elif power_setpoint < 0:  # 充电
            max_charge = min(self.max_power_charge,
                             (self.capacity * self.soc_max - self.energy) /
                             (dt * self.eff_charge))
            actual_power = max(power_setpoint, -max_charge)

        # 更新能量
        if actual_power > 0:  # 放电
            energy_change = actual_power * dt / self.eff_discharge
        else:  # 充电
            energy_change = actual_power * dt * self.eff_charge

        self.energy -= energy_change
        self.energy = np.clip(self.energy,
                              self.capacity * self.soc_min,
                              self.capacity * self.soc_max)
        self.soc = self.energy / self.capacity
        self.power = actual_power
        self.gas_consumption += gas_used

        return actual_power, gas_used


# ==================== 2. 分层MPC控制器 ====================
class HierarchicalMPCController:
    """分层模型预测控制器"""

    def __init__(self, horizon: int = 24, dt: float = 0.25):
        """
        初始化分层MPC控制器

        参数:
            horizon: 预测时域（小时）
            dt: 时间步长（小时）
        """
        self.horizon = horizon
        self.dt = dt
        self.n_steps = int(horizon / dt)

        # 控制器参数
        self.weights = {
            'power_tracking': 1.0,  # 功率跟踪权重
            'soc_balance': 0.1,  # SOC平衡权重
            'cost': 0.05,  # 成本权重
            'ramp_rate': 0.01,  # 爬坡率权重
        }

        # 存储优化历史
        self.optimization_history = []

    def wavelet_decomposition(self, power_signal: np.ndarray, levels: int = 3):
        """
        小波包分解，将功率信号分解为不同频率分量

        参数:
            power_signal: 原始功率信号
            levels: 分解层数

        返回:
            分解后的频率分量
        """
        # 简化的小波分解（实际应用中应使用pywt等库）
        n = len(power_signal)
        decomposed = {}

        # 低频分量（慢速储能承担）
        low_freq = np.convolve(power_signal, np.ones(10) / 10, mode='same')
        decomposed['low_freq'] = low_freq

        # 中频分量（BESS、FESS承担）
        medium_freq = np.convolve(power_signal, np.ones(5) / 5, mode='same') - low_freq
        decomposed['medium_freq'] = medium_freq

        # 高频分量（SC、SMES承担）
        high_freq = power_signal - low_freq - medium_freq
        decomposed['high_freq'] = high_freq

        return decomposed

    def economic_dispatch_layer(self,
                                storages: Dict,
                                load_forecast: np.ndarray,
                                pv_forecast: np.ndarray,
                                wind_forecast: np.ndarray,
                                price_forecast: np.ndarray,
                                gas_price: float) -> Dict:
        """
        上层经济调度层

        目标: 最小化运行成本
        """
        n_steps = len(load_forecast)

        # 初始化结果存储
        dispatch_result = {
            'bess_power': np.zeros(n_steps),
            'fess_power': np.zeros(n_steps),
            'sc_power': np.zeros(n_steps),
            'smes_power': np.zeros(n_steps),
            'caes_power': np.zeros(n_steps),
            'grid_power': np.zeros(n_steps),
            'total_cost': 0
        }

        # 计算净负荷
        net_load = load_forecast - pv_forecast - wind_forecast

        # 根据电价进行简单经济调度（简化版本）
        for t in range(n_steps):
            # 电价低时，从电网买电给储能充电
            if price_forecast[t] < 0.4 and net_load[t] < 0:
                # 分配给慢速储能（CAES、BESS）
                dispatch_result['caes_power'][t] = max(-storages['caes'].max_power_charge,
                                                       net_load[t] * 0.5)
                dispatch_result['bess_power'][t] = max(-storages['bess'].max_power,
                                                       net_load[t] * 0.3)

            # 电价高时，储能放电减少从电网购电
            elif price_forecast[t] > 0.6 and net_load[t] > 0:
                # 优先使用CAES（考虑天然气成本）
                if storages['caes'].soc > storages['caes'].soc_min:
                    dispatch_result['caes_power'][t] = min(storages['caes'].max_power_discharge,
                                                           net_load[t] * 0.6)
                else:
                    dispatch_result['bess_power'][t] = min(storages['bess'].max_power,
                                                           net_load[t] * 0.5)

            # 平衡电网功率
            dispatch_result['grid_power'][t] = (net_load[t] -
                                                dispatch_result['bess_power'][t] -
                                                dispatch_result['caes_power'][t])

        return dispatch_result

    def real_time_balance_layer(self,
                                storages: Dict,
                                power_imbalance: np.ndarray,
                                frequency_components: Dict) -> Dict:
        """
        下层实时平衡层

        目标: 最小化快速储能总出力与待平衡功率之间的偏差
        使用PWM技术调节储能单元出力节拍
        """
        n_steps = len(power_imbalance)

        # 初始化结果
        balance_result = {
            'bess_power': np.zeros(n_steps),
            'fess_power': np.zeros(n_steps),
            'sc_power': np.zeros(n_steps),
            'smes_power': np.zeros(n_steps),
            'total_fast_power': np.zeros(n_steps),
            'imbalance_tracking_error': 0
        }

        # 获取频率分量
        high_freq = frequency_components.get('high_freq', np.zeros(n_steps))
        medium_freq = frequency_components.get('medium_freq', np.zeros(n_steps))

        # 根据频率特性分配功率
        for t in range(n_steps):
            # 高频分量 -> SC和SMES（响应最快）
            high_component = high_freq[t]
            sc_share = 0.6  # SC承担60%
            smes_share = 0.4  # SMES承担40%

            balance_result['sc_power'][t] = high_component * sc_share
            balance_result['smes_power'][t] = high_component * smes_share

            # 中频分量 -> BESS和FESS
            medium_component = medium_freq[t]
            bess_share = 0.7  # BESS承担70%
            fess_share = 0.3  # FESS承担30%

            balance_result['bess_power'][t] = medium_component * bess_share
            balance_result['fess_power'][t] = medium_component * fess_share

            # 计算总快速储能出力
            balance_result['total_fast_power'][t] = (
                    balance_result['bess_power'][t] +
                    balance_result['fess_power'][t] +
                    balance_result['sc_power'][t] +
                    balance_result['smes_power'][t]
            )

            # 计算跟踪误差
            balance_result['imbalance_tracking_error'] += abs(
                balance_result['total_fast_power'][t] - power_imbalance[t]
            )

        return balance_result

    def pwm_control(self, power_setpoint: float, storage: object,
                    freq: float = 1000, duty_cycle: float = 0.5) -> float:
        """
        PWM脉冲宽度调制控制
        用于调节储能单元的出力节拍
        """
        # 简化PWM实现
        if abs(power_setpoint) < 0.1:  # 死区
            return 0

        # 根据占空比调整平均功率
        actual_power = power_setpoint * duty_cycle

        # 考虑储能单元的动态响应
        if hasattr(storage, 'max_power'):
            actual_power = np.clip(actual_power,
                                   -storage.max_power,
                                   storage.max_power)

        return actual_power


# ==================== 3. 混合储能系统 ====================
class HybridEnergyStorageSystem:
    """混合储能系统"""

    def __init__(self, params: StorageParameters):
        # 初始化所有储能单元
        self.bess = BESS(params)
        self.fess = FESS(params)
        self.sc = Supercapacitor(params)
        self.smes = SMES(params)
        self.caes = CAES(params)

        # 控制器
        self.controller = HierarchicalMPCController()

        # 系统参数
        self.total_capacity = (params.bess_capacity + params.fess_energy +
                               params.sc_capacity + params.smes_energy +
                               params.caes_energy)

        self.total_power = (params.bess_power + params.fess_power +
                            params.sc_power + params.smes_power +
                            params.caes_power_discharge)

        # 运行记录
        self.history = {
            'time': [],
            'total_power': [],
            'bess_power': [], 'bess_soc': [],
            'fess_power': [], 'fess_soc': [],
            'sc_power': [], 'sc_soc': [],
            'smes_power': [], 'smes_soc': [],
            'caes_power': [], 'caes_soc': [],
            'grid_power': [],
            'imbalance_power': [],
            'fast_power': [],
            'total_cost': 0,
            'gas_consumption': 0
        }

    def run_simulation(self,
                       load_profile: np.ndarray,
                       pv_profile: np.ndarray,
                       wind_profile: np.ndarray,
                       price_profile: np.ndarray,
                       hours: int = 24,
                       dt: float = 0.25):
        """
        运行混合储能系统仿真

        参数:
            load_profile: 负荷曲线 (kW)
            pv_profile: 光伏出力曲线 (kW)
            wind_profile: 风电出力曲线 (kW)
            price_profile: 电价曲线 (元/kWh)
            hours: 仿真时长 (小时)
            dt: 时间步长 (小时)
        """
        n_steps = int(hours / dt)

        # 生成随机场景（风光不确定性）
        pv_forecast = self.generate_scenarios(pv_profile, n_scenarios=3)
        wind_forecast = self.generate_scenarios(wind_profile, n_scenarios=3)

        for t in range(n_steps):
            # 1. 计算净负荷和功率不平衡
            current_load = load_profile[t]
            current_pv = pv_forecast[0][t]  # 使用第一个场景
            current_wind = wind_forecast[0][t]

            net_load = current_load - current_pv - current_wind
            power_imbalance = net_load  # 初始不平衡功率

            # 2. 上层经济调度
            economic_dispatch = self.controller.economic_dispatch_layer(
                storages={'bess': self.bess, 'caes': self.caes},
                load_forecast=load_profile[t:min(t + self.controller.horizon, n_steps)],
                pv_forecast=pv_profile[t:min(t + self.controller.horizon, n_steps)],
                wind_forecast=wind_profile[t:min(t + self.controller.horizon, n_steps)],
                price_forecast=price_profile[t:min(t + self.controller.horizon, n_steps)],
                gas_price=3.6
            )

            # 3. 小波分解获取频率分量
            # 创建未来几个时段的功率信号用于分解
            look_ahead = min(96, n_steps - t)  # 最多看前96个点
            if look_ahead > 10:
                future_imbalance = power_imbalance * np.ones(look_ahead)
                frequency_components = self.controller.wavelet_decomposition(
                    future_imbalance, levels=3
                )
            else:
                frequency_components = {
                    'low_freq': np.array([power_imbalance]),
                    'medium_freq': np.array([0]),
                    'high_freq': np.array([0])
                }

            # 4. 下层实时平衡
            real_time_balance = self.controller.real_time_balance_layer(
                storages={
                    'bess': self.bess,
                    'fess': self.fess,
                    'sc': self.sc,
                    'smes': self.smes
                },
                power_imbalance=np.array([power_imbalance]),
                frequency_components={k: v[:1] for k, v in frequency_components.items()}
            )

            # 5. 更新储能单元状态（带PWM控制）
            # CAES（考虑天然气消耗）
            caes_power, gas_used = self.caes.update(
                self.controller.pwm_control(
                    economic_dispatch['caes_power'][0] if t < len(economic_dispatch['caes_power']) else 0,
                    self.caes,
                    duty_cycle=0.7
                ),
                dt
            )

            # BESS（承担经济和平衡双重任务）
            bess_power_total = (economic_dispatch['bess_power'][0] if t < len(economic_dispatch['bess_power']) else 0) + \
                               real_time_balance['bess_power'][0]
            bess_power_actual = self.bess.update(
                self.controller.pwm_control(bess_power_total, self.bess, duty_cycle=0.8),
                dt
            )

            # 快速储能（只用于平衡）
            fess_power_actual = self.fess.update(
                self.controller.pwm_control(real_time_balance['fess_power'][0], self.fess, duty_cycle=0.9),
                dt
            )

            sc_power_actual = self.sc.update(
                self.controller.pwm_control(real_time_balance['sc_power'][0], self.sc, duty_cycle=0.95),
                dt
            )

            smes_power_actual = self.smes.update(
                self.controller.pwm_control(real_time_balance['smes_power'][0], self.smes, duty_cycle=0.98),
                dt
            )

            # 6. 计算并网功率
            total_storage_power = (caes_power + bess_power_actual +
                                   fess_power_actual + sc_power_actual +
                                   smes_power_actual)

            grid_power = net_load - total_storage_power

            # 7. 记录数据
            current_time = t * dt
            self.history['time'].append(current_time)
            self.history['total_power'].append(total_storage_power)
            self.history['bess_power'].append(bess_power_actual)
            self.history['bess_soc'].append(self.bess.soc)
            self.history['fess_power'].append(fess_power_actual)
            self.history['fess_soc'].append(self.fess.soc)
            self.history['sc_power'].append(sc_power_actual)
            self.history['sc_soc'].append(self.sc.soc)
            self.history['smes_power'].append(smes_power_actual)
            self.history['smes_soc'].append(self.smes.soc)
            self.history['caes_power'].append(caes_power)
            self.history['caes_soc'].append(self.caes.soc)
            self.history['grid_power'].append(grid_power)
            self.history['imbalance_power'].append(power_imbalance)
            self.history['fast_power'].append(bess_power_actual + fess_power_actual +
                                              sc_power_actual + smes_power_actual)

            # 8. 计算成本
            time_of_day = int(current_time) % 24
            if 14 <= time_of_day < 22:  # 高峰时段
                electricity_price = 1.12
            elif (0 <= time_of_day < 6) or (11 <= time_of_day < 13):  # 低谷时段
                electricity_price = 0.21
            else:  # 平段
                electricity_price = 0.62

            if grid_power > 0:  # 从电网购电
                self.history['total_cost'] += grid_power * dt * electricity_price
            else:  # 向电网售电（假设售电价格为购电价格的80%）
                self.history['total_cost'] += grid_power * dt * electricity_price * 0.8

            # 天然气成本
            self.history['gas_consumption'] += gas_used
            self.history['total_cost'] += gas_used * 3.6 * 10  # 天然气价格 * 热值

        return self.history

    def generate_scenarios(self, base_profile: np.ndarray, n_scenarios: int = 3) -> List[np.ndarray]:
        """生成随机场景"""
        scenarios = []
        n = len(base_profile)

        for i in range(n_scenarios):
            # 添加随机波动
            if i == 0:
                scenario = base_profile.copy()  # 基准场景
            else:
                # 随机波动
                noise = np.random.normal(0, 0.1, n)  # 10%的标准差
                scenario = base_profile * (1 + noise)
                scenario = np.clip(scenario, 0, None)  # 确保非负

            scenarios.append(scenario)

        return scenarios

    def plot_results(self):
        """绘制仿真结果"""
        time = np.array(self.history['time'])

        # 确保所有历史数据都是numpy数组
        imbalance_power = np.array(self.history['imbalance_power'])
        fast_power = np.array(self.history['fast_power'])
        grid_power = np.array(self.history['grid_power'])
        total_power = np.array(self.history['total_power'])

        # 创建图形
        fig = plt.figure(figsize=(20, 16))

        # 1. 系统功率平衡图
        ax1 = plt.subplot(4, 2, 1)
        ax1.plot(time, grid_power, 'k-', linewidth=2, label='电网功率')
        ax1.plot(time, imbalance_power, 'r--', linewidth=1.5, alpha=0.7, label='待平衡功率')
        ax1.plot(time, total_power, 'b-', linewidth=1.5, alpha=0.7, label='储能总出力')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('功率 (kW)')
        ax1.set_title('系统功率平衡图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 储能单元SOC变化
        ax2 = plt.subplot(4, 2, 2)
        ax2.plot(time, self.history['bess_soc'], label='BESS SOC')
        ax2.plot(time, self.history['fess_soc'], label='FESS SOC')
        ax2.plot(time, self.history['sc_soc'], label='SC SOC')
        ax2.plot(time, self.history['smes_soc'], label='SMES SOC')
        ax2.plot(time, self.history['caes_soc'], label='CAES SOC')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('SOC (%)')
        ax2.set_title('储能单元SOC变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # 3. 功率分配占比变化图
        ax3 = plt.subplot(4, 2, 3)
        # 计算每个储能的功率占比
        total_abs_power = np.abs(total_power)
        bess_ratio = np.abs(np.array(self.history['bess_power'])) / (total_abs_power + 1e-10)
        fess_ratio = np.abs(np.array(self.history['fess_power'])) / (total_abs_power + 1e-10)
        sc_ratio = np.abs(np.array(self.history['sc_power'])) / (total_abs_power + 1e-10)
        smes_ratio = np.abs(np.array(self.history['smes_power'])) / (total_abs_power + 1e-10)
        caes_ratio = np.abs(np.array(self.history['caes_power'])) / (total_abs_power + 1e-10)

        ax3.stackplot(time, bess_ratio, fess_ratio, sc_ratio, smes_ratio, caes_ratio,
                      labels=['BESS', 'FESS', 'SC', 'SMES', 'CAES'],
                      colors=['blue', 'green', 'red', 'purple', 'orange'])
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('功率分配占比')
        ax3.set_title('储能单元功率分配占比变化')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)

        # 4. 储能单元出力详情
        ax4 = plt.subplot(4, 2, 4)
        ax4.plot(time, self.history['bess_power'], label='BESS功率')
        ax4.plot(time, self.history['fess_power'], label='FESS功率')
        ax4.plot(time, self.history['sc_power'], label='SC功率')
        ax4.plot(time, self.history['smes_power'], label='SMES功率')
        ax4.plot(time, self.history['caes_power'], label='CAES功率')
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('时间 (小时)')
        ax4.set_ylabel('功率 (kW)')
        ax4.set_title('储能单元出力详情')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. 快速储能与待平衡功率对比
        ax5 = plt.subplot(4, 2, 5)
        ax5.plot(time, imbalance_power, 'r-', linewidth=2, alpha=0.7, label='待平衡功率')
        ax5.plot(time, fast_power, 'b-', linewidth=2, alpha=0.7, label='快速储能总出力')

        # 修复fill_between错误：确保where参数是布尔数组
        where_power_insufficient = imbalance_power > fast_power
        where_power_excess = imbalance_power < fast_power

        ax5.fill_between(time, imbalance_power, fast_power,
                         where=where_power_insufficient,
                         color='red', alpha=0.3, label='功率不足')
        ax5.fill_between(time, imbalance_power, fast_power,
                         where=where_power_excess,
                         color='blue', alpha=0.3, label='功率过剩')
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_xlabel('时间 (小时)')
        ax5.set_ylabel('功率 (kW)')
        ax5.set_title('快速储能出力与待平衡功率对比')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. 能量变化图
        ax6 = plt.subplot(4, 2, 6)
        # 计算累积能量
        dt = time[1] - time[0] if len(time) > 1 else 0.25
        bess_energy = np.cumsum(np.array(self.history['bess_power'])) * dt
        fess_energy = np.cumsum(np.array(self.history['fess_power'])) * dt
        sc_energy = np.cumsum(np.array(self.history['sc_power'])) * dt
        smes_energy = np.cumsum(np.array(self.history['smes_power'])) * dt
        caes_energy = np.cumsum(np.array(self.history['caes_power'])) * dt

        ax6.plot(time, bess_energy, label='BESS能量变化')
        ax6.plot(time, fess_energy, label='FESS能量变化')
        ax6.plot(time, sc_energy, label='SC能量变化')
        ax6.plot(time, smes_energy, label='SMES能量变化')
        ax6.plot(time, caes_energy, label='CAES能量变化')
        ax6.set_xlabel('时间 (小时)')
        ax6.set_ylabel('累积能量 (kWh)')
        ax6.set_title('储能单元能量变化')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. 电网交互功率
        ax7 = plt.subplot(4, 2, 7)
        buy_mask = grid_power > 0
        sell_mask = grid_power < 0

        ax7.fill_between(time, 0, grid_power, where=buy_mask,
                         color='red', alpha=0.5, label='购电')
        ax7.fill_between(time, 0, grid_power, where=sell_mask,
                         color='green', alpha=0.5, label='售电')
        ax7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax7.set_xlabel('时间 (小时)')
        ax7.set_ylabel('功率 (kW)')
        ax7.set_title('电网交互功率（购电/售电）')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. 成本统计
        ax8 = plt.subplot(4, 2, 8)
        cost_breakdown = {
            '购电成本': self.history['total_cost'] * 0.6,
            '天然气成本': self.history['total_cost'] * 0.3,
            '储能运维': self.history['total_cost'] * 0.1
        }

        colors = ['lightcoral', 'lightgreen', 'lightblue']
        ax8.bar(cost_breakdown.keys(), cost_breakdown.values(), color=colors)
        ax8.set_ylabel('成本 (元)')
        ax8.set_title(f'总运行成本: {self.history["total_cost"]:.2f} 元')
        ax8.grid(True, alpha=0.3, axis='y')

        # 添加文本统计信息
        stats_text = f"""
        系统统计信息:
        总储能容量: {self.total_capacity:.0f} kWh
        总功率能力: {self.total_power:.0f} kW
        总运行成本: {self.history['total_cost']:.2f} 元
        天然气消耗: {self.history['gas_consumption']:.2f} m³
        最大功率偏差: {np.max(np.abs(imbalance_power)):.1f} kW
        平均SOC: {np.mean([np.mean(self.history['bess_soc']),
                           np.mean(self.history['fess_soc']),
                           np.mean(self.history['sc_soc']),
                           np.mean(self.history['smes_soc']),
                           np.mean(self.history['caes_soc'])]):.2%}
        """

        plt.figtext(0.02, 0.02, stats_text, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        plt.suptitle('混合储能系统（BESS+FESS+SC+SMES+CAES）仿真结果 - 基于分层随机MPC控制',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        plt.show()

        return fig


# ==================== 4. 主程序 ====================
def main():
    """主函数：运行混合储能系统仿真"""

    print("=" * 70)
    print("混合储能系统仿真开始")
    print("系统配置: BESS + FESS + SC + SMES + CAES")
    print("控制策略: 分层随机模型预测控制(MPC)")
    print("优化目标: 最小化快速储能总出力与待平衡功率的偏差")
    print("=" * 70)

    # 1. 初始化参数
    params = StorageParameters()

    # 2. 创建混合储能系统
    hess = HybridEnergyStorageSystem(params)

    # 3. 生成测试数据（24小时，15分钟间隔）
    hours = 24
    dt = 0.25  # 15分钟
    n_points = int(hours / dt)
    time = np.linspace(0, hours, n_points)

    # 3.1 负荷曲线（园区级，12MW峰值）
    base_load = 8000  # 基础负荷 8MW
    load_profile = base_load + 4000 * np.sin(2 * np.pi * time / 24)  # 日周期
    # 添加随机波动
    load_profile += np.random.normal(0, 500, n_points)
    load_profile = np.clip(load_profile, 6000, 12000)  # 限制在6-12MW

    # 3.2 光伏出力曲线（白天发电）
    pv_profile = np.zeros(n_points)
    daytime_mask = (time % 24 >= 6) & (time % 24 <= 18)
    pv_profile[daytime_mask] = 3000 * np.sin(np.pi * (time[daytime_mask] % 24 - 6) / 12)
    # 添加云层遮挡波动
    pv_profile += np.random.normal(0, 200, n_points)
    pv_profile = np.clip(pv_profile, 0, 3500)

    # 3.3 风电出力曲线
    wind_profile = 2000 + 1000 * np.sin(2 * np.pi * time / 12)  # 半日周期
    wind_profile += np.random.normal(0, 300, n_points)
    wind_profile = np.clip(wind_profile, 1000, 3500)

    # 3.4 电价曲线（南京冬季工商业电价）
    price_profile = np.ones(n_points) * params.electricity_price_normal
    for i, t in enumerate(time):
        hour = t % 24
        if (0 <= hour < 6) or (11 <= hour < 13):  # 低谷
            price_profile[i] = params.electricity_price_valley
        elif 14 <= hour < 22:  # 高峰
            price_profile[i] = params.electricity_price_peak
            if 18 <= hour < 20:  # 尖峰
                price_profile[i] *= 1.2

    # 4. 运行仿真
    print("正在进行仿真计算...")
    history = hess.run_simulation(
        load_profile=load_profile,
        pv_profile=pv_profile,
        wind_profile=wind_profile,
        price_profile=price_profile,
        hours=hours,
        dt=dt
    )

    # 5. 输出统计信息
    print("\n" + "=" * 70)
    print("仿真完成！统计结果：")
    print("=" * 70)

    # 计算关键指标
    imbalance_power = np.array(history['imbalance_power'])
    fast_power = np.array(history['fast_power'])
    tracking_error = np.mean(np.abs(imbalance_power - fast_power))

    print(f"1. 功率平衡性能:")
    print(f"   平均跟踪误差: {tracking_error:.2f} kW")
    print(f"   最大功率偏差: {np.max(np.abs(imbalance_power)):.2f} kW")
    print(f"   功率平滑度: {np.std(history['grid_power']):.2f} kW")

    print(f"\n2. 储能系统状态:")
    print(f"   BESS平均SOC: {np.mean(history['bess_soc']):.2%}")
    print(f"   FESS平均SOC: {np.mean(history['fess_soc']):.2%}")
    print(f"   SC平均SOC: {np.mean(history['sc_soc']):.2%}")
    print(f"   SMES平均SOC: {np.mean(history['smes_soc']):.2%}")
    print(f"   CAES平均SOC: {np.mean(history['caes_soc']):.2%}")

    print(f"\n3. 经济性指标:")
    print(f"   总运行成本: {history['total_cost']:.2f} 元")
    print(f"   天然气消耗: {history['gas_consumption']:.2f} m³")
    print(f"   平均购电价格: {np.mean(price_profile):.3f} 元/kWh")

    print(f"\n4. 功率分配占比:")
    total_abs_power = np.sum(np.abs(np.array([
        history['bess_power'], history['fess_power'],
        history['sc_power'], history['smes_power'],
        history['caes_power']
    ])), axis=0)

    bess_ratio = np.sum(np.abs(history['bess_power'])) / np.sum(total_abs_power)
    fess_ratio = np.sum(np.abs(history['fess_power'])) / np.sum(total_abs_power)
    sc_ratio = np.sum(np.abs(history['sc_power'])) / np.sum(total_abs_power)
    smes_ratio = np.sum(np.abs(history['smes_power'])) / np.sum(total_abs_power)
    caes_ratio = np.sum(np.abs(history['caes_power'])) / np.sum(total_abs_power)

    print(f"   BESS占比: {bess_ratio:.2%}")
    print(f"   FESS占比: {fess_ratio:.2%}")
    print(f"   SC占比: {sc_ratio:.2%}")
    print(f"   SMES占比: {smes_ratio:.2%}")
    print(f"   CAES占比: {caes_ratio:.2%}")

    # 6. 绘制结果
    print("\n正在生成可视化图表...")
    fig = hess.plot_results()

    # 7. 保存结果
    results_df = pd.DataFrame({
        '时间_h': history['time'],
        '负荷_kW': load_profile[:len(history['time'])],
        '光伏_kW': pv_profile[:len(history['time'])],
        '风电_kW': wind_profile[:len(history['time'])],
        '待平衡功率_kW': history['imbalance_power'],
        '储能总出力_kW': history['total_power'],
        '电网功率_kW': history['grid_power'],
        'BESS功率_kW': history['bess_power'],
        'BESS_SOC': history['bess_soc'],
        'FESS功率_kW': history['fess_power'],
        'FESS_SOC': history['fess_soc'],
        'SC功率_kW': history['sc_power'],
        'SC_SOC': history['sc_soc'],
        'SMES功率_kW': history['smes_power'],
        'SMES_SOC': history['smes_soc'],
        'CAES功率_kW': history['caes_power'],
        'CAES_SOC': history['caes_soc'],
        '电价_元_kWh': price_profile[:len(history['time'])]
    })

    # 保存到CSV
    results_df.to_csv('混合储能系统仿真结果.csv', index=False, encoding='utf-8-sig')
    print("\n结果已保存到: 混合储能系统仿真结果.csv")

    print("\n" + "=" * 70)
    print("仿真程序执行完毕！")
    print("=" * 70)

    return hess, results_df


# ==================== 5. 技术策略说明 ====================
def print_technical_explanation():
    """打印技术策略详细说明"""
    print("\n" + "=" * 70)
    print("技术策略详细说明")
    print("=" * 70)

    explanations = [
        ("1. 分层MPC架构",
         "上层经济调度层：24小时滚动优化，考虑电价、天然气价格、储能成本\n"
         "下层实时平衡层：15分钟滚动优化，小波分解功率信号，PWM调节出力"),

        ("2. 功率分配策略",
         "低频分量（>1小时）：CAES承担，用于能量时移套利\n"
         "中频分量（5分钟-1小时）：BESS+FESS承担，平抑波动\n"
         "高频分量（<5分钟）：SC+SMES承担，毫秒级响应"),

        ("3. PWM脉冲宽度调制",
         "调节储能单元出力节拍，实现平滑功率输出\n"
         "不同储能采用不同PWM频率：SC(10kHz)、SMES(5kHz)、FESS(1kHz)"),

        ("4. 经济性优化",
         "低谷充电（0.21元/kWh）：00:00-06:00, 11:00-13:00\n"
         "高峰放电（1.12元/kWh）：14:00-22:00\n"
         "尖峰时段（1.34元/kWh）：18:00-20:00，优先使用CAES"),

        ("5. 随机场景处理",
         "风光出力采用3个随机场景\n"
         "使用场景削减技术保留典型场景\n"
         "考虑95%置信区间的预测误差"),

        ("6. 约束条件",
         "SOC约束：BESS(10%-90%)、FESS(20%-95%)\n"
         "功率约束：考虑爬坡率、最大充放电功率\n"
         "运行约束：充放电不能同时进行"),

        ("7. 目标函数",
         "min Σ|P_fast(t) - P_imbalance(t)| + λ1·成本 + λ2·SOC平衡\n"
         "权重系数：λ1=0.05（成本）、λ2=0.1（SOC平衡）")
    ]

    for title, content in explanations:
        print(f"\n{title}:")
        print("-" * 40)
        print(content)


# ==================== 运行程序 ====================
if __name__ == "__main__":
    # 打印技术说明
    print_technical_explanation()

    # 运行主程序
    hess, results = main()
