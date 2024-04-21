import serial
import csv
import time

# 打开串口连接
ser = serial.Serial('COM26', 115200)  # 替换 'COM1' 为 Arduino 的串口端口

# 打开 CSV 文件准备写入
with open('ECG.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # 写入 CSV 文件的标题行
    csvwriter.writerow(['Timestamp', 'ECG'])

    # 从串口接收数据并写入 CSV 文件
    while True:
        line = ser.readline().decode().strip()  # 读取并解码一行数据
        if line:  # 如果接收到数据
            # 获取当前时间戳
            timestamp = time.time()
            # 将时间戳和心电信号值写入 CSV 文件
            csvwriter.writerow([timestamp, line])

# 关闭串口连接
ser.close()
