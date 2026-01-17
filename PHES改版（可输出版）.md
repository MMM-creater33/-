import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PHESystem:
    """
    抽水蓄能 (PHES) 系统模型
    对应论文公式 (1) - (6)
    """

    def __init__(self, capacity_m3=1000000, min_volume_ratio=0.1, pump_power_max=250000, gen_power_max=200000, eta_pump=0.85, eta_gen=0.9):
        """
        初始化参数
        :param capacity_m3: 上游水库总库容 (m^3)
        :param min_volume_ratio: 最小有效库容占比 (对应死水位)
        :param pump_power_max: 水泵最大功率 (kW)
        :param gen_power_max: 发电机最大功率 (kW)
        :param eta_pump: 水泵效率 (包含电机和泵)
        :param eta_gen: 发电效率 (包含水轮机和发电机)
        """
        # 物理参数 (论文符号对应)
        self.V_up_max = capacity_m3             # V_up,max
        self.V_up_min = capacity_m3 * min_volume_ratio # V_up,min
        self.P_pump_max = pump_power_max        # P_pump,max
        self.P_gen_max = gen_power_max          # P_gen,max
        self.eta_pump = eta_pump                # eta_pump
        self.eta_gen = eta_gen                  # eta_gen

        # 水力常数 (简化假设水头 h 恒定，密度 rho 和 g 为常数)
        self.rho = 1000         # 水的密度 (kg/m^3)
        self.g = 9.81           # 重力加速度 (m/s^2)
        self.h = 200            # 有效水头 (m) - 假设为定值，实际中会随水位微变

        # 运行状态初始化
        self.V_current = (self.V_up_max + self.V_up_min) / 2  # 初始水量 (m^3)
        selfSOC = (self.V_current - self.V_up_min) / (self.V_up_max - self.V_up_min) # 初始 SOC

    def step(self, price, dt_hours):
        """
        单步仿真逻辑 (简单的基于价格的控制策略)
        """
        # 简单的控制策略：低价电时抽水，高价电时发电
        P_pump = 0
        P_gen = 0
        mu_pump = 0
        mu_gen = 0

        # 论文公式 (5): 互斥约束
        # 同一时间只能有一种模式 (简化处理，实际需优化器求解)
        if price < 50:  # 电价低 -> 抽水 (充电)
            # 检查是否达到上限 (公式 2)
            if self.V_current < self.V_up_max:
                # 尝试以最大功率抽水 (受限于公式 3)
                P_pump = self.P_pump_max
                mu_pump = 1
        elif price > 80: # 电价高 -> 发电 (放电)
            # 检查是否达到下限 (公式 2)
            if self.V_current > self.V_up_min:
                # 尝试以最大功率发电 (受限于公式 4)
                P_gen = self.P_gen_max
                mu_gen = 1

        # 论文公式 (1): 水量更新方程
        # 注意单位转换：功率(kW) * 时间(h) = 能量(kWh) -> 转换为焦耳或直接利用 m^3/s 逻辑
        # 简化计算：利用能量守恒直接计算体积变化
        # 抽水量 (m^3) = (电能输入 * 效率) / (rho * g * h)
        # 发电量 (m^3) = (电能输出) / (rho * g * h * 效率)

        # 计算体积变化率 (m^3/h)
        # 注意：P(kW) = 1000 * J/s -> 转换为 m^3/h 需要 * 3600 / (rho * g * h)
        factor = 3600 / (self.rho * self.g * self.h) # 转换系数

        delta_V_pump = P_pump * self.eta_pump * factor * dt_hours
        delta_V_gen = P_gen * dt_hours / (self.eta_gen * factor) # 这里反向计算体积消耗

        # 更新水量
        V_new = self.V_current + delta_V_pump - delta_V_gen

        # 库容硬约束 (论文公式 2)
        V_new = max(self.V_up_min, min(self.V_up_max, V_new))

        # 更新状态
        self.V_current = V_new
        self.SOC = (self.V_current - self.V_up_min) / (self.V_up_max - self.V_up_min)

        return {
            "P_pump": P_pump,
            "P_gen": P_gen,
            "Volume": self.V_current,
            "SOC": self.SOC
        }

# --- 仿真运行 ---

# 1. 初始化系统
phes = PHESystem(
    capacity_m3=1000000,   # 100万方
    min_volume_ratio=0.2,  # 死库容20%
    pump_power_max=200000, # 200MW
    gen_power_max=180000   # 180MW
)

# 2. 模拟输入 (24小时电价)
hours = 24
dt = 1 # 小时
time = np.arange(hours)
prices = np.random.uniform(20, 100, hours) # 模拟随机电价

# 3. 运行仿真
results = []
for t in range(hours):
    # 简单的启停逻辑，实际应用建议使用Pyomo等优化求解器
    res = phes.step(prices[t], dt)
    results.append(res)

# 4. 数据整理
df = pd.DataFrame(results)
df['Time'] = time
df['Price'] = prices

# --- 结果可视化 ---

# 1. 功率图 (Pump vs Gen)
plt.figure(figsize=(12, 5))
plt.subplot(2, 1, 1)
plt.plot(df['Time'], df['P_pump'], label='Pump Power (kW)', color='b', linestyle='--')
plt.plot(df['Time'], df['P_gen'], label='Gen Power (kW)', color='r')
plt.title('PHES Pumping and Generating Power')
plt.ylabel('Power (kW)')
plt.legend()
plt.grid(True)

# 2. 能量图 (Volume)
plt.subplot(2, 1, 2)
plt.plot(df['Time'], df['Volume'], label='Water Volume (m^3)', color='c', linewidth=2)
plt.axhline(y=phes.V_up_min, color='k', linestyle='--', alpha=0.5, label='Min Limit')
plt.axhline(y=phes.V_up_max, color='k', linestyle='--', alpha=0.5, label='Max Limit')
plt.title('Reservoir Water Volume')
plt.xlabel('Time (Hour)')
plt.ylabel('Volume (m^3)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 3. SOC 图
plt.figure(figsize=(6, 3))
plt.plot(df['Time'], df['SOC'], label='SOC', color='g', marker='o')
plt.title('State of Charge (SOC)')
plt.xlabel('Time (Hour)')
plt.ylabel('SOC (0-1)')
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()

# --- 输出数据表 ---
print("### 仿真数据详情 ###")
print(df[['Time', 'Price', 'P_pump', 'P_gen', 'Volume', 'SOC']])
