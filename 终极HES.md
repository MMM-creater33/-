import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class HydrogenStorageSystem:
    """
    氢储能 (HES) 系统模型
    对应论文公式 (7) - (16)
    """

    def __init__(self, H2_capacity_kg=100, LHV_H2=33.3, eta_elect=0.8, eta_FC=0.5):
        """
        初始化氢储能系统参数
        :param H2_capacity_kg: 储氢罐总容量 (kg)
        :param LHV_H2: 氢气低热值 (kWh/kg)，论文中符号 LHV_H2
        :param eta_elect: 电解槽效率 (电-氢), 论文符号 eta_eb
        :param eta_FC: 燃料电池效率 (氢-电), 论文符号 eta_f,s
        """
        # 物理容量
        self.M_H2_max = H2_capacity_kg  # 最大氢气质量 (kg)
        self.M_H2_min = 0.1 * H2_capacity_kg  # 最小氢气质量 (kg) (预留10%死区)

        # 功率限制 (kW)
        self.P_elect_max = 100.0  # 电解槽最大功率
        self.P_elect_min = 20.0   # 电解槽最小稳定功率
        self.P_FC_max = 80.0      # 燃料电池最大功率
        self.P_FC_min = 10.0      # 燃料电池最小稳定功率

        # 核心参数
        self.LHV_H2 = LHV_H2      # 氢气低热值
        self.eta_elect = eta_elect # 电解槽效率
        self.eta_FC = eta_FC      # 燃料电池发电效率

        # 状态变量初始化
        self.M_H2 = self.M_H2_max / 2  # 初始氢气质量 (kg)，设为一半
        self.soc = self.M_H2 / self.M_H2_max # 初始 SOC

    def step(self, P_charge_demand, P_discharge_request, dt=1.0):
        """
        模拟一个时间步长的运行
        :param P_charge_demand: 可用的充电功率 (如多余光伏功率), kW
        :param P_discharge_request: 负荷需求的放电功率, kW
        :param dt: 时间步长 (小时)
        :return: 当前步长的各项数据字典
        """
        # --- 1. 电解槽运行逻辑 (充电侧) ---
        # 论文约束: mu_eb(t) * P_eb,min <= P_eb(t) <= mu_eb(t) * P_eb,max
        # 简化逻辑：如果有余电且储氢罐未满，则充电
        P_elect_out = 0.0
        m_electrolyzer = 0.0 # 产氢速率 (kg/h)

        if P_charge_demand > 0 and self.M_H2 < self.M_H2_max:
            # 取可用功率和最大功率的最小值
            P_elect_use = min(P_charge_demand, self.P_elect_max)
            # 论文公式 (8): 产氢速率
            m_electrolyzer = (P_elect_use * self.eta_elect) / self.LHV_H2
            P_elect_out = P_elect_use

        # --- 2. 燃料电池运行逻辑 (放电侧) ---
        # 论文约束: mu_fs(t) * P_fs,min <= P_fs(t) <= mu_fs(t) * P_fs,max
        # 简化逻辑：如果有负荷需求且储氢罐有氢，则放电
        P_FC_out = 0.0
        m_FC = 0.0 # 耗氢速率 (kg/h)

        if P_discharge_request > 0 and self.M_H2 > self.M_H2_min:
            # 论文公式 (9): 耗氢速率 -> 功率
            # m_FC = P_FC / (eta_FC * LHV_H2) -> P_FC = m_FC * eta_FC * LHV_H2
            # 这里反向推导最大可发功率
            max_possible_power = (self.M_H2 - self.M_H2_min) * self.eta_FC * self.LHV_H2 / dt
            requested_power = min(P_discharge_request, self.P_FC_max, max_possible_power)
            m_FC = requested_power / (self.eta_FC * self.LHV_H2)
            P_FC_out = requested_power

        # --- 3. 论文公式 (7): 更新储氢罐质量 ---
        # M_H2(t) = M_H2(t-Δt) + (m_eb - m_fs - m_h_load)Δt
        # 这里假设没有直接供氢负荷 (m_h_load = 0)
        delta_M = (m_electrolyzer - m_FC) * dt
        self.M_H2 = np.clip(self.M_H2 + delta_M, self.M_H2_min, self.M_H2_max)

        # --- 4. 更新 SOC ---
        self.soc = (self.M_H2 - self.M_H2_min) / (self.M_H2_max - self.M_H2_min)

        # --- 5. 计算系统总输出功率 ---
        # 净功率 = 放电功率 - 充电功率 (充电为负)
        P_net = P_FC_out - P_elect_out

        return {
            'Time': dt,
            'P_Electrolyzer (kW)': P_elect_out,
            'P_FuelCell (kW)': P_FC_out,
            'P_Net (kW)': P_net,
            'M_H2 (kg)': self.M_H2,
            'SOC': self.soc,
            'Delta_M (kg)': delta_M
        }

def run_simulation():
    # --- 1. 初始化 ---

    hes = HydrogenStorageSystem() # 氢储能模型

    # --- 2. 生成模拟输入数据 (24小时) ---
    # 模拟逻辑：白天有光伏(充电)，晚上有负荷(放电)
    hours = 24
    time = np.arange(0, hours, 1)
    P_charge_signal = np.zeros(hours)
    P_discharge_signal = np.zeros(hours)

    # 白天光伏充电 (8:00 - 16:00)
    P_charge_signal[8:16] = np.linspace(0, 100, 8) # 功率线性增加
    # 晚上负荷放电 (18:00 - 22:00)
    P_discharge_signal[18:22] = np.linspace(80, 50, 4)

    # --- 3. 数据存储列表 ---
    results = []

    for t in range(hours):
        # 获取输入信号
        P_in = P_charge_signal[t]
        P_load = P_discharge_signal[t]

        # 执行一步仿真
        # 注意：这里为了展示，强制让充电和放电互斥
        if P_in > 0:
            data = hes.step(P_in, 0, dt=1.0)
        else:
            data = hes.step(0, P_load, dt=1.0)

        results.append(data)

    # --- 4. 数据处理与输出 ---
    df = pd.DataFrame(results)

    print("=== 氢储能系统运行报告 ===")
    print(df.describe())
    print("\n")

    # --- 5. 绘图 ---
    plt.figure(figsize=(12, 8))

    # 子图 1: 功率曲线
    plt.subplot(3, 1, 1)
    plt.plot(df['P_Electrolyzer (kW)'], label='Electrolyzer (Charge)', color='tab:blue', marker='o')
    plt.plot(df['P_FuelCell (kW)'], label='Fuel Cell (Discharge)', color='tab:red', marker='x')
    plt.title('Power Profile (kW)')
    plt.ylabel('Power (kW)')
    plt.legend()
    plt.grid(True)

    # 子图 2: SOC 变化
    plt.subplot(3, 1, 2)
    plt.plot(df['SOC'], label='SOC', color='tab:green', marker='o')
    plt.axhline(y=0.2, color='r', linestyle='--', label='Min SOC (20%)')
    plt.axhline(y=1.0, color='g', linestyle='--', label='Max SOC (100%)')
    plt.title('State of Charge (SOC)')
    plt.ylabel('SOC')
    plt.legend()
    plt.grid(True)

    # 子图 3: 能量（质量）变化
    plt.subplot(3, 1, 3)
    plt.plot(df['M_H2 (kg)'], label='Stored Hydrogen Mass', color='tab:purple', marker='o')
    plt.title('Energy Storage (Hydrogen Mass in kg)')
    plt.ylabel('Mass (kg)')
    plt.xlabel('Time (Hour)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
