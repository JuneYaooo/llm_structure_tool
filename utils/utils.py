import pandas as pd
import json
import datetime,time
import shutil
import os
import re
import subprocess
from sklearn.metrics import classification_report
from pynvml import (nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlShutdown)

def read_excel_file(file_path):
    df = pd.read_excel(file_path)
    return df

def save_to_excel(df, file_path):
    df.to_excel(file_path, index=False)

def get_available_gpu(threshold=20000):
    # Initialize NVML
    nvmlInit()

    # Get the number of GPU devices
    device_count = nvmlDeviceGetCount()

    # Find GPU devices with available memory greater than the threshold
    available_gpus = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        free_memory_mb = info.free / 1024 / 1024

        if free_memory_mb > threshold:
            available_gpus.append(i)

    # Shutdown NVML
    nvmlShutdown()

    return available_gpus


def is_numeric(value):
    try:
        float(value)  # 尝试将值转换为浮点数
        return True  # 如果转换成功，则表示值可以转换为数字
    except (ValueError, TypeError):
        return False  # 如果转换失败或者值的类型不是字符串或数字，则表示值不是数字


def accuracy_cal(list1, list2):
    count = 0
    for i in range(len(list1)):
        l = list1[i]
        r = list2[i]
        l = float(l) if is_numeric(l) else str(l)
        # print('l',is_numeric(l),l)
        r = float(r) if is_numeric(r) else str(r)
        # print('r',is_numeric(r),r)
        if l == r:
            count += 1
    accuracy = count / len(list1)
    return accuracy

def evaluate_model(pred_path, gold_path):
    output_path = 'data/evaluate_scores.xlsx'

    # 读取金标签数据excel
    gold_data = pd.read_excel(gold_path, sheet_name=None)
    
    # 读取预测结果excel
    pred_data = pd.read_excel(pred_path, sheet_name=None)
    
    # 创建输出结果集excel
    # writer = pd.ExcelWriter(output_path)
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

    # 定义结果集DataFrame
    result_df = pd.DataFrame(columns=['sheet', 'field', 'precision', 'recall', 'f1-score', 'accuracy', 'support'])
    
    # 遍历金标签数据excel的所有sheet
    for sheet_name in gold_data.keys():
        # 获取金标签数据和预测结果的对应sheet
        gold_sheet = gold_data[sheet_name]
        pred_sheet = pred_data.get(sheet_name, None)
        
        # 如果预测结果中没有该sheet，输出提示信息
        if pred_sheet is None:
            print(f"Warning: Sheet '{sheet_name}' not found in prediction file.")
            continue
        
        # 获取金标签数据和预测结果的所有字段
        gold_columns = sorted(gold_sheet.columns.tolist())
        pred_columns = sorted(pred_sheet.columns.tolist())
        
        # 确保金标签数据和预测结果中的字段一致
        if set(gold_columns) != set(pred_columns):
            print(f"Warning: Columns mismatch in sheet '{sheet_name}'.")
            continue
        
        # 提取金标签数据和预测结果中的标签数据
        gold_labels = gold_sheet.fillna(value='未提及')
        pred_labels = pred_sheet.fillna(value='未提及')
         # 判断预测结果中是否存在空值
        if pred_labels.isnull().any().any():
            print(f"Warning: Missing values detected in sheet '{sheet_name}'.")
        
        # 将预测结果中的数据类型转换为与金标签数据excel中相应的字段相同的数据类型
        for col in gold_columns:
            if gold_labels[col].dtype != pred_labels[col].dtype:
                pred_labels[col] = pred_labels[col].fillna('').astype(str)
                gold_labels[col] = gold_labels[col].fillna('').astype(str)
        max_lengths = {col: pred_labels[col].apply(lambda x: str(x)).str.len().max() for col in pred_labels.columns}
        pred_labels = pred_labels.fillna('').astype(str)
        gold_labels = gold_labels.fillna('').astype(str)
         # 计算准确率、F1分数、精确率和召回率
        for col in gold_columns:
                report = classification_report(gold_labels[col], pred_labels[col], output_dict=True, zero_division=0)
                accuracy = accuracy_cal(list(gold_labels[col]), list(pred_labels[col]))  #report['accuracy']
                new_row = pd.DataFrame({'sheet': sheet_name, 'field': col, 
                        'precision': report['weighted avg']['precision'], 
                        'recall': report['weighted avg']['recall'], 
                        'f1-score': report['weighted avg']['f1-score'],
                        'accuracy': accuracy, 
                        'support': report['weighted avg']['support']}, index=[0])
                result_df = pd.concat([result_df, new_row], axis=0, ignore_index=True)
    # 将结果写入输出结果集excel中的新sheet
    result_df.to_excel(writer, sheet_name='result', index=False)
    workbook = writer.book
    worksheet = writer.sheets['result']
    percent_format = workbook.add_format({'num_format': '0%'})
    worksheet.set_column('C:F', None, percent_format)
    worksheet.conditional_format('C2:F500', {'type': 'data_bar',
                                        'bar_color': '#FFA500'})
    # 保存输出结果集excel
    writer.close()
    print(f"Comparison complete. Results saved to '{output_path}'.")
    return output_path


def stop_train_process():
    process = subprocess.Popen('ps -ef | grep train_bash.py', shell=True, stdout=subprocess.PIPE)
    output, _ = process.communicate()
    process.kill()
    n = 0
    # 解析输出以获取进程ID
    print('output',output)
    try:
        lines = output.decode().split('\n')
        for line in lines:
            if 'train_bash.py' in line:
                parts = line.split()
                pid = parts[1]
                # 杀死进程
                subprocess.call(['kill', '-9', pid])
                n+=1
    except Exception as e:
        print('error!!',e)

    return f'停止了{n//2}个进程'
