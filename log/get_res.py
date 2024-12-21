import os, time, json, sys, threading, gc
import argparse
current_file_path = os.path.abspath(__file__)
pwd = os.path.dirname(current_file_path)
os.chdir(pwd)
sys.path.append(pwd)

print("Current file path:", pwd)

log_files = []
last_lines = []

for file_name in os.listdir(pwd):
    if file_name.endswith('.log'):
        log_files.append(file_name)
        with open(os.path.join(pwd, file_name), 'r') as file:
            lines = file.readlines()
            if lines:
                last_lines.append(lines[-1].strip())
            else:
                last_lines.append('')

# 打印结果以验证
print("Log files:", log_files)
print("Last lines:", last_lines)

dataset = []
# patch_len = []
# stride = []
for file_name in log_files:
    # 以_分割文件名
    parts = file_name.split('_')
    dataset.append(parts[1][:-4])
    # patch_len.append(parts[5])
    # stride.append(parts[6][:-4])

print("Dataset:", dataset)
# print("Patch length:", patch_len)

mse = []
mae = []
# rse = []
for line in last_lines:
    print(line)
    parts = line.split(',')
    mse.append(parts[0].strip()[4:])
    mae.append(parts[1].strip()[4:])
    # rse.append(parts[2].strip()[4:])
print("MSE:", mse)
print("MAE:", mae)
# print("RSE:", rse)

# 保存结果到csv文件
import pandas as pd
df = pd.DataFrame({
    'Dataset': dataset,
    # 'Patch length': patch_len,
    # 'Stride': stride,
    'MSE': mse,
    'MAE': mae,
    # 'RSE': rse
})
df.to_csv('result.csv', index=False)
print("Result saved to result.csv")
