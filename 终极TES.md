import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

class ThermalEnergyStorage:
    def __init__(self, capacity_kwh=1000, max_charge_power_kW=200, max_discharge_power_kW=200, self_consumption_rate=0.001):
        """
        初始化热储能系统 (基于热量 kWh 计算)
        """
        self.H_max = capacity_kwh          # 最大储热量 (kWh)
        self.H_min = 0.1 * capacity_kwh    # 最小储热量限制 (10%)
        self.P_charge_max = max_charge_power_kW   # 最大充热功率 (kW)
        self.P_discharge_max = max_discharge_power_kW # 最大放热功率 (kW)

        # 效率参数
        self.eta_charge = 0.95  # 充热效率 (电->热, 电加热器)
        self.eta_discharge = 0.85 # 放热效率 (热->电/工质, 汽轮机/换热器)

        self.self_loss = self_consumption_rate # 自散热系数 (每小时自然流失的比例)

        self.H_current = capacity_kwh * 0.5 # 初始储热量 (50% SOC)

    def step(self, input_power_kW, demand_load_kW, dt=1):
        """
        单步仿真逻辑 (核心)
        input_power_kW: 可用的输入电功率 (正数)
        demand_load_kW: 外部需要的热/电功率 (正数)
        dt: 时间步长 (小时)
        """
        # --- 1. 计算理论充放热功率 ---
        P_charge_theoretical = input_power_kW * self.eta_charge # 理论充入功率 (考虑加热效率)

        # --- 2. 考虑自散热损失 ---
        # 热量流失是基于当前储热量的比例
        P_self_loss_kW = self.H_current * self.self_loss
        # 注意：自散热通常直接从储热罐中扣除，不经过功率转换

        # --- 3. 计算实际充放电 (受限于物理边界) ---
        # 充热限制：受限于功率上限和剩余空间
        P_charge_actual = min(P_charge_theoretical, self.P_charge_max, (self.H_max - self.H_current) / dt)

        # 放热限制：受限于功率上限和可用热量
        # 注意：这里 demand_load_kW 是外部需求，系统能提供的最大功率不能超过 demand_load_kW
        P_discharge_theoretical = demand_load_kW / self.eta_discharge # 系统需要拿出多少热量来满足需求
        P_discharge_actual = min(P_discharge_theoretical, self.P_discharge_max, demand_load_kW, self.H_current / dt)

        # --- 4. 更新储热量 (核心状态方程) ---
        # H(t) = H(t-1) - 自损 - 实际放出 + 实际充入
        H_next = (self.H_current
                  - P_self_loss_kW * dt
                  - P_discharge_actual * self.eta_discharge # 放出的热量 (注意效率已在上面计算，这里扣减储热)
                  + P_charge_actual * dt) # 充入的热量

        # 强制限制在物理边界内 (防止计算误差溢出)
        H_next = max(self.H_min, min(self.H_max, H_next))

        # --- 5. 计算关键指标 ---
        SOC = H_next / self.H_max # SOC 计算

        # 记录端电压 (如果是热泵/电加热器，输入电压可能恒定，如果是热电联产可能变动)
        # 热储能通常没有 "端电压" 概念，除非特指电加热器入口。
        # 这里我们假设电加热器工作电压恒定，或者该值代表热侧出口温度相关的等效值。
        # 如果不需要，可以忽略 voltage 这一项。
        voltage = 800 if P_charge_actual > 0 else 0 # 假设工作电压

        # 记录 C-Rate (对于热储能，C-Rate 通常指 充放电功率 / 容量)
        # 例如 1C 表示 1 小时把 1000kWh 充满，对应功率 1000kW
        C_Rate_charge = P_charge_actual / self.H_max
        C_Rate_discharge = P_discharge_actual / self.H_max

        # --- 6. 更新状态 ---
        self.H_current = H_next

        return {
            "SOC": SOC,
            "H_current": H_next,          # 当前能量 (kWh)
            "P_charge": P_charge_actual,  # 实际充热功率 (kW)
            "P_discharge": P_discharge_actual, # 实际放热功率 (kW)
            "P_loss": P_self_loss_kW,     # 自散热功率 (kW)
            "C_Rate_c": C_Rate_charge,
            "C_Rate_d": C_Rate_discharge,
            "Voltage": voltage            # 模拟值
        }

# --- 2. 仿真设置与运行 ---
if __name__ == "__main__":
    # 1. 初始化系统 (1000kWh 容量, 200kW 功率)
    tes = ThermalEnergyStorage(capacity_kwh=1000, max_charge_power_kW=200, max_discharge_power_kW=200)

    # 2. 模拟数据生成 (24小时)
    hours = 24
    data_log = []

    print(f"{'Time':<5} {'SOC':<8} {'Energy(kWh)':<12} {'P_Charge(kW)':<15} {'P_Dis(kW)':<15} {'C_Rate(c/d)':<15}")

    for t in range(hours):
        # 模拟输入：光伏多余功率 (0-250kW) 用于充电
        solar_surplus = random.uniform(0, 250)
        # 模拟需求：热负荷需求 (0-250kW)
        heat_demand = random.uniform(0, 250)

        # 执行一步仿真
        result = tes.step(input_power_kW=solar_surplus, demand_load_kW=heat_demand, dt=1)

        # 记录数据
        data_log.append({
            "Time": t,
            "SOC": result['SOC'],
            "Energy": result['H_current'],
            "Charge_Power": result['P_charge'],
            "Discharge_Power": result['P_discharge'],
            "C_Rate_c": result['C_Rate_c'],
            "C_Rate_d": result['C_Rate_d']
        })

        # 实时打印数据 (直接的数据产出)
        print(f"{t:<5} {result['SOC']:.4f}     {result['H_current']:.2f}          {result['P_charge']:.2f}             {result['P_discharge']:.2f}             {result['C_Rate_c']:.2f}/{result['C_Rate_d']:.2f}")

    # 转换为 DataFrame 用于表格展示
    df = pd.DataFrame(data_log)
    print("\n--- 详细数据表格 (Pandas DataFrame) ---")
    print(df.to_string(index=False))

    # --- 3. 绘图 ---
    plt.figure(figsize=(12, 8))

    # 子图 1: SOC 变化
    plt.subplot(2, 1, 1)
    plt.plot(df['Time'], df['SOC'], marker='o', color='b', label='SOC')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Max SOC (100%)')
    plt.title('Thermal Storage SOC (State of Charge)')
    plt.ylabel('SOC (p.u.)')
    plt.xlabel('Time (Hour)')
    plt.legend()
    plt.grid(True)

    # 子图 2: 功率曲线
    plt.subplot(2, 1, 2)
    plt.plot(df['Time'], df['Charge_Power'], marker='^', color='g', label='Charge Power (kW)')
    plt.plot(df['Time'], df['Discharge_Power'], marker='v', color='r', label='Discharge Power (kW)')
    plt.title('Thermal Storage Charge/Discharge Power')
    plt.ylabel('Power (kW)')
    plt.xlabel('Time (Hour)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
