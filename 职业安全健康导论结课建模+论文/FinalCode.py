import numpy as np
import itertools
import matplotlib.pyplot as plt

# 数据定义
work_hours_data = {
    "average_weekly_days": [6.4],
    "average_daily_hours": [9.8],
    "working_7_days_percentage": [61.6],
    "working_8_to_10_hours_percentage": [55.1],
    "working_above_10_hours_percentage": [36.7]
}

order_quantity_data = {
    "orders_11_to_20_percentage": [40.7],
    "orders_below_10_percentage": [33.3],
    "orders_1_to_30_percentage": [18.9]
}

time_pressure_data = {
    "on_time_delivery_percentage": [93],
    "slight_delay_percentage": [5],
    "severe_delay_percentage": [2]
}

emotional_labor_data = {
    "high_emotional_labor_percentage": [36.0],
    "general_emotional_labor_percentage": [29.3],
    "low_emotional_labor_percentage": [25.7]
}

class WorkHours:
    def __init__(self, data):
        self.intensity = data

class OrderQuantity:
    def __init__(self, data):
        self.intensity = data

class TimePressure:
    def __init__(self, data):
        self.intensity = data

class EmotionalLabor:
    def __init__(self, data):
        self.intensity = data

# 固定值
fixed_values = {
    'WorkHours': work_hours_data["working_7_days_percentage"][0],
    'OrderQuantity': order_quantity_data["orders_11_to_20_percentage"][0],
    'TimePressure': time_pressure_data["slight_delay_percentage"][0],
    'EmotionalLabor': emotional_labor_data["high_emotional_labor_percentage"][0]
}

# 定义权重
weights = [0.25, 0.25, 0.25, 0.25]

def generate_combinations(varying_category):
    if varying_category == 'WorkHours':
        combinations = itertools.product(
            work_hours_data["working_7_days_percentage"] + work_hours_data["working_8_to_10_hours_percentage"] +
            work_hours_data["working_above_10_hours_percentage"],
            [fixed_values['OrderQuantity']],
            [fixed_values['TimePressure']],
            [fixed_values['EmotionalLabor']]
        )
    elif varying_category == 'OrderQuantity':
        combinations = itertools.product(
            [fixed_values['WorkHours']],
            order_quantity_data["orders_11_to_20_percentage"] + order_quantity_data["orders_below_10_percentage"] +
            order_quantity_data["orders_1_to_30_percentage"],
            [fixed_values['TimePressure']],
            [fixed_values['EmotionalLabor']]
        )
    elif varying_category == 'TimePressure':
        combinations = itertools.product(
            [fixed_values['WorkHours']],
            [fixed_values['OrderQuantity']],
            time_pressure_data["slight_delay_percentage"] + time_pressure_data["severe_delay_percentage"],
            [fixed_values['EmotionalLabor']]
        )
    elif varying_category == 'EmotionalLabor':
        combinations = itertools.product(
            [fixed_values['WorkHours']],
            [fixed_values['OrderQuantity']],
            [fixed_values['TimePressure']],
            emotional_labor_data["high_emotional_labor_percentage"] + emotional_labor_data["general_emotional_labor_percentage"] +
            emotional_labor_data["low_emotional_labor_percentage"]
        )
    return list(combinations)

def visualization(combinations):
    intensity_results = []
    for combo in combinations:
        work_hours = WorkHours(combo[0])
        order_quantity = OrderQuantity(combo[1])
        time_pressure = TimePressure(combo[2])
        emotional_labor = EmotionalLabor(combo[3])

        intensity_vector = np.array([work_hours.intensity, order_quantity.intensity,
                                     time_pressure.intensity, emotional_labor.intensity])

        final_intensity = np.sum(intensity_vector * np.array(weights))
        intensity_results.append((combo, final_intensity))

    # 按强度排序结果
    sorted_results = sorted(intensity_results, key=lambda x: x[1])

    # 可视化排序后的结果
    combinations_indices = range(len(sorted_results))
    intensities = [result[1] for result in sorted_results]

    plt.figure(figsize=(12, 6))
    plt.plot(combinations_indices, intensities, marker='o')
    plt.title('Labor Intensity for Different Combinations')
    plt.xlabel('Combination Index')
    plt.ylabel('Labor Intensity')
    plt.grid(True)
    plt.show()

# 生成仅工作时间变化的组合
combinations = generate_combinations('WorkHours')
visualization(combinations)

# 生成仅订单数量变化的组合
combinations = generate_combinations('OrderQuantity')
visualization(combinations)

# 生成仅时间压力变化的组合
combinations = generate_combinations('TimePressure')
visualization(combinations)

# 生成仅情绪劳动变化的组合
combinations = generate_combinations('EmotionalLabor')
visualization(combinations)