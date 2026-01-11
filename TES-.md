import datetime
from meteostat import Point, Hourly
import yaml
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import timedelta

#STEPS 1 & 2
# Set up Meteostat location and date range
def fetch_outdoor_temp(location, start, end):
    data = Hourly(location, start, end).fetch()
    return data['temp']

# Load and process COP data from YAML file
def load_cop_data(yaml_path, condenser_temp=60):
    with open(yaml_path, 'r') as file:
        cop_data = yaml.safe_load(file)['heat_pump_cop_data']
    outdoor_temps = [entry['outdoor_temp_C'] for entry in cop_data]
    cop_values = [entry['COP_noisy'] for entry in cop_data]
    delta_temps = [condenser_temp - t for t in outdoor_temps]
    inverse_delta_temps = [1 / dt if dt != 0 else 0 for dt in delta_temps]
    return inverse_delta_temps, cop_values

# Perform linear regression on COP data and plot results
def plot_cop_vs_inverse_delta_t(inverse_delta_temps, cop_values):
    slope, intercept, _, _, _ = linregress(inverse_delta_temps, cop_values)
    print(f"Fitted values: a = {intercept}, b = {slope}")

    plt.scatter(inverse_delta_temps, cop_values, label='Observed COP values', color='blue')
    plt.plot(inverse_delta_temps, [intercept + slope * x for x in inverse_delta_temps], color='red', label=f'Fitted Line: COP = {intercept:.2f} + {slope:.2f}/ΔT')
    plt.xlabel('1/ΔT (1/(Condenser Temp - Outdoor Temp))')
    plt.ylabel('COP')
    plt.title('COP vs 1/ΔT with Fitted Linear Regression Line')
    plt.legend()
    plt.show()

# Load building inputs for heat load calculation
def load_building_inputs(yaml_path):
    with open(yaml_path, 'r') as file:
        inputs = yaml.safe_load(file)['building_properties']
    return inputs['wall_area']['value'], inputs['wall_U_value']['value'], inputs['roof_area']['value'], inputs['roof_U_value']['value'], inputs['indoor_setpoint_temperature_K']['value']

# Calculate and plot heat load over time based on outdoor temperature
def plot_heat_load(location, start, end, Aw, Uw, Ar, Ur, Tin):
    outdoor_temps = fetch_outdoor_temp(location, start, end)
    timestamps = outdoor_temps.index
    heat_loads = [(Aw * Uw * (Tamb + 273.15 - Tin) + Ar * Ur * (Tamb + 273.15 - Tin)) for Tamb in outdoor_temps]

    plt.plot(timestamps, heat_loads)
    plt.xlabel('Time')
    plt.ylabel('Heat Load (W)')
    plt.title('Heat Load Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print(heat_loads)


