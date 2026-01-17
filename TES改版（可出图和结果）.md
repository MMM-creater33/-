import matplotlib.pyplot as plt
import numpy as np
import random

class TESystem:         #定义一个T的类
    def __init__(self, capacity_kwh, max_charge_kW, max_discharge_kW, self_consumption_rate=0.005):  #构造函数，初始化器。初始化参数
        """
        初始化热储能系统参数
        :param capacity_kwh: 总容量 (kWh)
        :param max_charge_kW: 最大充热功率 (kW)
        :param max_discharge_kW: 最大放热功率 (kW)
        :param self_consumption_rate: 自散热系数 sigma (每小时流失的比例)
        """
        self.H_max = capacity_kwh
        self.H_min = 0.1 * capacity_kwh  # 假设最低保留 10% 的热量，对应公式 (21)
        self.P_charge_max = max_charge_kW
        self.P_discharge_max = max_discharge_kW

        # 效率参数 \定义效率 (对应公式 18, 19)
        self.eta_e2h = 0.95  # 电转热效率 (电加热器)
        self.eta_h2e = 0.85  # 热转电效率 (汽轮机，此处仅用于计算放热量，实际发电受热约束)

        # 系统初始状态
        self.H_current = capacity_kwh * 0.5  # 初始储热量 (kWh)
        self.sigma = self_consumption_rate   # 自散热率
        self.history_H = []      # 保存储热量历史数据    #创建一个空列表
        self.history_P_net = []  # 保存净充放电功率历史
        self.history_price = []  # 保存电价历史


#仿真逻辑（计算系统在每个时间点发生了什么）
    def step(self, electric_price, heat_load_demand_kW, time_step_hour=1):   #定义单步运行函数
        """
        模拟一个时间步长的运行 (对应公式 17, 20)
        :param electric_price: 当前时刻电价 (用于策略判断，元/kWh)
        :param heat_load_demand_kW: 当前时刻外部热负荷需求 (kW)
        :param time_step_hour: 时间步长 (小时)
        """
        # --- 策略逻辑 ---
        # 简单策略：低价电时充电（满足自身损耗和部分负荷），高价电且有需求时放电
        # 实际优化中，这里会替换成由公式 (22-24) 构建的约束求解器

        P_heater_input = 0.0  # 电加热器输入功率 (kW)
        P_turbine_output = 0.0 # 汽轮机发电功率 (kW)

        # 1. 决策充放电 (简单的启发式策略)
        # 如果有热负荷需求 或者 储能快满了，尝试充电
        if electric_price < 0.5 or self.H_current > 0.9 * self.H_max:     #如果电价便宜（小于0.5）或储热罐快空了，就启动电加热器充电
            # 充电模式 (公式 18)
            P_heater_input = self.P_charge_max
        else:
            # 放电模式 (公式 19, 20)
            if heat_load_demand_kW > 0 and self.H_current > self.H_min:
                # 假设放热功率足以覆盖负荷
                P_turbine_output = min(heat_load_demand_kW / self.eta_h2e, self.P_discharge_max)   #否则启动汽轮机放电满足热负荷

        # --- 核心计算 ---

        # 计算实际热功率流动
        P_charge_actual = P_heater_input * self.eta_e2h          # 实际存进去的热功率 (kW)  #输入的电功率不能完全变为热能，要乘电转热效率才是实际的
        P_discharge_actual = P_turbine_output * self.eta_h2e    # 实际取出来的热功率 (kW)

        # 计算自散热损失 (对应公式 17 中的 sigma 部分)
        loss = self.H_current * self.sigma   #模拟储热罐无论用不用，热量都会随时间自然散失一部分

        # 更新储热量 (核心状态方程 - 公式 17)
        H_next = self.H_current * (1 - self.sigma) + (P_charge_actual - P_discharge_actual) * time_step_hour

        # 约束处理 (对应公式 21)
        if H_next > self.H_max:
            H_next = self.H_max
        elif H_next < self.H_min:
            H_next = self.H_min

        # 计算净功率变化 (用于绘图)
        P_net = P_charge_actual - P_discharge_actual

        # 更新状态
        self.H_current = H_next
        self.history_H.append(self.H_current)
        self.history_P_net.append(P_net)
        self.history_price.append(electric_price)

#可视化与主程序

def run_simulation():
    # 1. 初始化模型 (假设容量 10 MWh = 10000 kWh)
    tes = TESystem(capacity_kwh=10000, max_charge_kW=2000, max_discharge_kW=2000)  #实例化，造出一个具体的储热罐

    # 2. 构建模拟场景 (24小时)
    hours = 24
    time_steps = range(hours)

    # 模拟电价 (低谷-高峰)
    prices = [0.3]*6 + [0.8]*12 + [0.3]*6

    # 模拟热负荷需求 (随机波动)
    heat_loads = [random.uniform(500, 1000) for _ in range(hours)]

    # 3. 运行仿真
    for i in time_steps:
        tes.step(prices[i], heat_loads[i])

    # 4. 数据处理与绘图
    plt.figure(figsize=(12, 8))

#SOC图（储热量变化）
    # 绘制储热量 (SOC 图) - 对应公式 17 的结果
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, tes.history_H, marker='o', color='tab:red', label='Stored Heat (kWh)')
    plt.axhline(y=tes.H_max, color='r', linestyle='--', alpha=0.5, label='Max Capacity')
    plt.axhline(y=tes.H_min, color='g', linestyle='--', alpha=0.5, label='Min Limit')
    plt.title('Thermal Energy Storage (TES) Level Over Time')
    plt.xlabel('Time (Hour)')
    plt.ylabel('Energy (kWh)')
    plt.legend()
    plt.grid(True)

#净功率图
    # 绘制净功率 (充放电功率) - 对应公式 18, 19
    plt.subplot(3, 1, 2)
    plt.bar(time_steps, tes.history_P_net, color='tab:blue', label='Net Power (kW)')
    plt.title('Charging(+)/Discharging(-) Power')
    plt.xlabel('Time (Hour)')
    plt.ylabel('Power (kW)')
    plt.grid(True)

#电价走势图
    # 绘制电价 (作为参考)
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, prices, marker='x', color='gray', label='Electric Price')
    plt.title('Electricity Price (Reference)')
    plt.xlabel('Time (Hour)')
    plt.ylabel('Price ($/kWh)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()       #弹出窗口

    # 5. 输出结果摘要
    print("=== TES Simulation Report ===")
    print(f"Final Storage Level: {tes.history_H[-1]:.2f} kWh")
    print(f"Max Energy: {tes.H_max} kWh")
    print(f"Min Energy Limit: {tes.H_min} kWh")
    print(f"Total Cycles: Approx. {np.ptp(tes.history_H) / tes.H_max:.2f}")

if __name__ == "__main__":
    run_simulation()
