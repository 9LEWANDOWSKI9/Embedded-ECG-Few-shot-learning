import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Arduino配置
reference_voltage = 5.0  # 参考电压
adc_resolution = 1024  # ADC分辨率

# 读取CSV文件并转换数据，并将结果写入新的CSV文件
with open('ECG.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过标题行
    with open('converted_ECG.csv', 'w', newline='') as new_csvfile:
        writer = csv.writer(new_csvfile)
        writer.writerow(['Timestamp', 'Original Voltage'])  # 写入标题行
        for row in reader:
            # 将心电压数据从字符串转换为整数
            analog_value = int(row[1])

            # 将模拟值转换为电压值
            voltage = (analog_value * reference_voltage) / adc_resolution

            # 将转换后的原始电压值写入新的CSV文件
            writer.writerow([row[0], voltage])

print("转换完成，并已将结果保存到 converted_data.csv 文件中。")

# import pandas as pd
# from datetime import datetime
# df = pd.read_csv('converted_ECG.csv')
# df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# df['Timestamp'] = df['Timestamp'].astype(int) // 10**9
# # df.drop(columns=['Timestamp'], inplace=True)
# df.to_csv('converted_ECG.csv', index=False)

# 读取转换后的CSV文件并提取数据
# timestamps = []
# voltages = []
# with open('converted_ECG.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)  # 跳过标题行
#     for row in reader:
#         timestamps.append(float(row[0]))  # 将时间戳转换为浮点数
#         voltages.append(float(row[1]))    # 将电压值转换为浮点数
#
# # 绘制图像
# plt.plot(timestamps, voltages)
# plt.xlabel('Timestamp')
# plt.ylabel('Original Voltage')
# plt.title('Original Voltage vs. Timestamp')
# plt.grid(True)
# plt.show()
from VMD import VMD
alpha = 1000
tau = 0
DC = 0
init = 1
tol = 1e-7
k = 4
csvfile = pd.read_csv('converted_ECG.csv')
reader = csv.reader(csvfile)
next(reader)
data = np.genfromtxt("converted_ECG.csv", delimiter=",")
f = data[:, 1]
print(f)
# T = data[:, 0]
u, u_hat, omega = VMD(f, alpha, tau, k, DC, init, tol)
plt.figure()
plt.plot(u.T)
plt.show()
