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
import json
from config.common_config import *

model_loaded = False
project_change = False
last_project_name = ''

def read_excel_file(file_path):
    df = pd.read_excel(file_path)
    return df

def save_to_excel(df, file_path):
    df.to_excel(file_path, index=False)

def process_data(config_path, training_data_path):
    medical_logic = pd.read_excel(config_path)
    source = medical_logic['文本来源'].dropna().unique().tolist()
    source_table = list(set([s.split('.')[0] for s in source if isinstance(s, str) and '.' in s]+medical_logic['文本类型'].dropna().unique().tolist()))
    source_table = [st for st in source_table if st != '']
    log = []

    log.append(f'配置文件文本来源表： {source_table}')

    source_table_dict = {}
    tables = {}

    for i, st in enumerate(source_table):
        try:
            tables[f'table{i}'] = pd.read_excel(training_data_path, sheet_name=st)
            source_table_dict[f'table{i}'] = st
        except Exception as e:
            log.append(f'尝试读取sheet{st}发生错误!!{e}')

    flipped_source_table_dict = {v: k for k, v in source_table_dict.items()}

    all_data = []

    for i, row in medical_logic.iterrows():
        if pd.isnull(row['文本来源']) or row['文本来源'] == '':
            log.append(f"文本来源为空，跳过字段：{row['字段名']}")
            continue

        table_name, text_field = row['文本来源'].split('.')[:2]

        if table_name not in source_table_dict.values():
            log.append(f'训练数据没有这个表，跳过表：{table_name}')
            continue

        if pd.notnull(row['属性提取']):
            table = tables[flipped_source_table_dict[row['文本类型']]]
        else:
            table = tables[flipped_source_table_dict[table_name]]
        field = row['字段名']

        if field not in table.columns.tolist():
            log.append(f'训练数据表{table_name}中没有这个字段，跳过字段：{field}')
            continue
        
        special_requirement = row['特殊要求'] if ('特殊要求' in row and pd.notnull(row['特殊要求'])) else ''

        for j, row2 in table.iterrows():
            if pd.isnull(row2[field]) or row2[field] == '':
                # 遇到为空的跳过
                log.append(f'{field}字段存在空行，跳过空行')
                continue

            if pd.notnull(row['属性提取']):
                arr_table, arr_field = row['属性提取'].split('.')[:2]
                arr = str(row2[arr_field])

                if pd.notnull(row['属性前缀']):
                    arr_prefix = row['属性前缀']
            else:
                arr = ''
                arr_prefix = ''

            if row['值域类型'] == '单选':
                row['字段名'] = row['字段名'].replace('大小1', '大小')
                context = str(row2[text_field]).replace('_x000D_', '').replace('\n', '')
                field_range = row['值域'].split(',') if row['值域'] is not None else []
                if '未提及' not in field_range:
                    field_range.append('未提及')
                answer = '未提及' if pd.isnull(row2[field]) else str(row2[field])
                instruction = f"##结构化任务##根据下文中信息，判断{arr}{arr_prefix}{row['字段名']}是什么？{'请在值域【'+row['值域']+'】中选择1个。' if row['值域'] is not None else '值域不固定。'}{special_requirement}"
                all_data.append({"instruction": instruction, "input": context, "output": answer})
            elif row['值域类型'] == '多选':
                row['字段名'] = row['字段名'].replace('大小1', '大小')
                context= str(row2[text_field]).replace('x000D', '').replace('\n', '')
                field_range = row['值域'].split(',') if row['值域'] is not None else []
                answer = str(row2[field]).replace('，', ',') if pd.notnull(row2[field]) else ''
                instruction = f"##结构化任务##根据下文中信息，判断{arr}{row['字段名']}是什么？{'请从值域【'+row['值域']+'】中选出文中提到的所有内容。' if row['值域'] is not None else '值域不固定。'}{special_requirement}"
                all_data.append({"instruction": instruction, "input": context, "output": answer})
            elif row['值域类型'] == '提取':
                row['字段名'] = row['字段名'].replace('大小1', '大小')
                context = str(row2[text_field]).replace('_x000D_', '').replace('\n', '')
                answer = str(row2[field]) if pd.notnull(row2[field]) else ''
                instruction = f"##结构化任务##根据下文中信息，判断{arr}{row['字段名']}是什么？请提取文中对应的值。{special_requirement}"
                all_data.append({"instruction": instruction, "input": context, "output": answer})

    log = '\n'.join(log)  # 使用换行符拼接log内容
    return all_data, log



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

def train_model(model_name, project_name, config_path, training_data_path):
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    all_data,log = process_data(config_path, training_data_path)
    log_file_path = f'data/logs/{now_str}.log'  # 定义log文件路径
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # 创建存储log的文件夹
    current_directory = os.getcwd()
    print("当前目录是:", current_directory)
    model_file_name = llm_model_dict[model_name]['name']
    model_path = llm_model_dict[model_name]['model_path']
    template = llm_model_dict[model_name]['template']
    lora_target = llm_model_dict[model_name]['lora_target']
    per_device_train_batch_size = llm_model_dict[model_name]['per_device_train_batch_size']

    folders_to_check = ["data", "checkpoints", "logs"]
    for folder in folders_to_check:
        folder_path = os.path.join(current_directory, "finetune", model_file_name,folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"已创建文件夹：{folder_path}")
        else:
            print(f"文件夹已存在：{folder_path}")

    with open(log_file_path, 'w', encoding="utf-8") as f:
        f.write(log)  # 将log内容写入文件

    # 遍历文件夹中的持久化数据集文件
    persistent_folder_path = "data/persistent"
    output_data = []
    if not os.path.exists(persistent_folder_path):
        os.makedirs(persistent_folder_path)
    for file_name in os.listdir(persistent_folder_path):
        if file_name.endswith('.json') and project_name not in file_name:
            file_path = os.path.join(persistent_folder_path, file_name)
            print('持久化数据集文件 file_path',file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                output_data+=file_data
    all_data += output_data
    with open(f"data/{project_name}_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4,  ensure_ascii=False)
    shutil.copyfile(f"data/{project_name}_dataset.json", f"finetune/{model_file_name}/data/{project_name}_dataset.json")
    
    available_gpus = get_available_gpu(threshold=20000)
    print('available_gpus[0]',available_gpus[0])
    try:
        # 读取JSON文件
        with open(f'{model_name}/data/dataset_info.json', 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {} 
    # 添加内容到JSON数据
    absolute_path = os.path.join(current_directory, f"finetune/{model_file_name}/data/{project_name}_dataset.json")
    print(' absolute_path', absolute_path)
    new_data = {
        project_name: {"file_name": absolute_path}
    }
    data.update(new_data)

    # 保存更新后的JSON文件
    with open(f'finetune/{model_file_name}/data/dataset_info.json', 'w') as file:
        json.dump(data, file, indent=4)
    content = f'''CUDA_VISIBLE_DEVICES={available_gpus[0]} python {current_directory}/train_bash.py --stage sft   --model_name_or_path {model_path}   --do_train     --dataset {project_name}    --template {template}     --finetuning_type lora     --lora_target {lora_target}     --output_dir checkpoints/{project_name}   --overwrite_output_dir  --overwrite_cache     --per_device_train_batch_size {per_device_train_batch_size}     --gradient_accumulation_steps 4     --lr_scheduler_type cosine     --logging_steps 10     --save_steps 1000     --learning_rate 5e-5     --num_train_epochs 3.0     --plot_loss     --fp16
    '''
    sh_file_name = f'finetune/{model_file_name}/train_{project_name}.sh'

    with open(sh_file_name , 'w') as file:
        file.write(content)

    # 设置文件可执行权限
    os.chmod(sh_file_name , 0o755)

    subprocess.Popen(f"""cd {current_directory}/finetune/{model_file_name} && . {conda_env_file} && conda activate llm_structure && nohup sh train_{project_name}.sh > ./logs/train_{now_str}.log 2>&1 &""", shell=True) # 换conda环境
    print('finish')
    # model.train(training_data_path)
    return f'{model_name} on training'+'\n\ndata process recording:\n'+log

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

def infer_model(model_name, project_name, config_path, test_data_path):
        log = []

        log.append(f'开始预测：')

        print('test==',model_name, config_path, test_data_path)
        current_directory = os.getcwd()
        print("当前目录是:", current_directory)
        model = load_model(model_name, project_name)
        if model == 'no enough GPU, please try it later!':
            return 'no enough GPU, please try it later!'

        medical_logic = pd.read_excel(config_path)
        source = medical_logic['文本来源'].unique().tolist()
        print('source',source)
        
        source_table = list(set([s.split('.')[0] for s in source if isinstance(s, str) and '.' in s and pd.notnull(s)]))
        while '' in source_table:
            source_table.remove('')
        log.append(f'source_table：{source_table}')

        source_table_dict = {}
        tables = {}
        for i, st in enumerate(source_table):
            try:
                tables[f'table{i}'] = pd.read_excel(test_data_path, sheet_name=st)
                source_table_dict[f'table{i}'] = st
            except Exception as e:
                print('error!! ', e)
                log.append(f'尝试读取sheet{st}发生错误!!')
        flipped_source_table_dict = {v: k for k, v in source_table_dict.items()}


        start_time = time.time()
        def get_result(model, project_name,row,context,field,arr,special_requirement):
            prompt = ''
            if row['值域类型'] == '多选':
                prompt = f"""##结构化任务##根据下文中信息，判断{arr}{row['字段名']}是什么？请在值域【{row['值域']}】中选择提到的所有内容。{special_requirement}"""
            elif row['值域类型'] == '单选':
                prompt = f"##结构化任务##根据下文中信息，判断{arr}{row['字段名']}是什么？请在值域【{row['值域']}】中选择1个。{special_requirement}"
            elif row['值域类型'] == '提取':
                row['字段名'] = row['字段名'].replace('大小1','大小')
                prompt = f"""##结构化任务##根据下文中信息，判断{arr}{row['字段名']}是什么？请提取文中对应的值。{special_requirement}"""
            else:
                return ''
            res , (prompt_length, response_length) =  model.chat(context,[],prompt) 
            return res


        from collections import defaultdict

        # create a dictionary to store rows with the same '属性提取'
        attr_dict = defaultdict(list)

        for i, row in medical_logic.iterrows():
            if row['文本来源'] =='' or pd.isnull(row['文本来源']):
                print('文本来源为空，跳过字段：',row['字段名'] )
                continue
            table_name, text_field = row['文本来源'].split('.')[0], row['文本来源'].split('.')[1]
            # 如果数据没有这个表，则跳过
            if table_name not in list(source_table_dict.values()):
                log.append(f'跳过表，{table_name}')
                continue
            table = tables[flipped_source_table_dict[table_name]]
            field = row['字段名'] 
            special_requirement = row['特殊要求'] if ('特殊要求' in row and pd.notnull(row['特殊要求'])) else ''
            if pd.notnull(row['属性提取']):
                arr_table,arr_field = row['属性提取'].split('.')[0],row['属性提取'].split('.')[1]
                arr_store_table = row['文本类型']
                if pd.notnull(row['属性前缀']):
                    arr_prefix = row['属性前缀']
                else:
                    arr_prefix = ''

                arr_row = medical_logic.loc[(medical_logic['字段名'] == arr_field)&(medical_logic['文本类型'] == arr_table)].iloc[0]
                arr_table = tables[flipped_source_table_dict[arr_table]]
                for k, row2 in arr_table.iterrows():
                    context = str(row2[text_field]).replace('_x000D_', '').replace('\n', '')
                    arr_content = get_result(model, project_name,arr_row,context,arr_field,'',special_requirement)
                    arr_list = arr_content.split(',')
                    for arr in arr_list:
                        print('arr',arr)
                        if arr and arr !='未提及'and arr !='其他':
                            result = get_result(model, project_name,row,context,field,arr+arr_prefix,special_requirement)
                            print('arr result',arr+arr_prefix,field,result)
                            # add the row to the dictionary with the same '属性提取'
                            attr_dict[arr_store_table].append({'content':context,arr_store_table:arr,arr_field:arr,field:result})
            else:
                for j, row2 in table.iterrows():
                    arr = '' if table_name != '检查记录-心电' else '心电图'
                    context = str(row2[text_field]).replace('_x000D_', '').replace('\n', '')
                    result = get_result(model, project_name,row,context,field, arr,special_requirement)
                    table.loc[j, field] = result 
                    continue
                pass
            tables[table_name] = table


        with pd.ExcelWriter(r'./data/result.xlsx') as writer:
            for st in source_table:
                if st in flipped_source_table_dict:
                    print('flipped_source_table_dict[st]', flipped_source_table_dict[st])
                    table_name = flipped_source_table_dict[st]
                    table = tables[table_name]
                    table.to_excel(writer, sheet_name=st, index=False)
            for attr, rows in attr_dict.items():
                # create a new table for rows with the same '属性提取'
                new_table = pd.DataFrame(rows)
                print(attr,'new_table',new_table)
                new_table = new_table.groupby(["content",attr], as_index=False).agg('first')
                new_table = new_table.drop(attr, axis=1)
                # save the new table to a file with the name of the '属性提取'
                new_table.to_excel(writer, sheet_name=attr, index=False)

        end_time = time.time()
        print(f'cost {round(end_time-start_time,2)} seconds')
         
        return 'data/result.xlsx','success'


def load_model(model_name, project_name):
    global model_loaded, model, tokenizer, project_change, last_project_name
    current_directory = os.getcwd()
    model_file_name = llm_model_dict[model_name]['name']
    model_path = llm_model_dict[model_name]['model_path']
    template = llm_model_dict[model_name]['template']
    lora_target = llm_model_dict[model_name]['lora_target']
    if project_name != last_project_name:
        project_change = True
    if not model_loaded or project_change:
        available_gpus = get_available_gpu(threshold=11000)
        if len(available_gpus)>0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
            print('available_gpus[0]',available_gpus[0])
        else:
            return 'no enough GPU, please try it later!',''
        try:
            from llmtuner.chat.stream_chat import ChatModel
            args = {
                "model_name_or_path": model_path,
                "template": template,
                "finetuning_type": "lora",
                "lora_target": lora_target,
                "checkpoint_dir": f"{current_directory}/finetune/{model_file_name}/checkpoints/{project_name}",
                "max_length":max_length,
                "do_sample":do_sample,
                "temperature":temperature
            }
            model = ChatModel(args)
            model_loaded = True
            last_project_name = project_name
            project_change = False
            # return model, tokenizer
        except Exception as e:
            print('error!! ', e)
            return e,''
    return model

def query_model(model_name, project_name, field_type, field_name, value_range, special_requirement, query):
    print(f'{model_name} start query')
    model = load_model(model_name, project_name)
    if model == 'no enough GPU, please try it later!':
        return 'no enough GPU, please try it later!'
    prompt = ''
    if field_type == '多选':
        prompt = f"""##结构化任务##根据下文中信息，判断{field_name}是什么？请在值域【{value_range}】中选择提到的所有内容。{special_requirement}"""
    elif field_type == '单选':
        prompt = f"##结构化任务##根据下文中信息，判断{field_name}是什么？请在值域【{value_range}】中选择1个。{special_requirement}"
    elif field_type == '提取':
        prompt = f"""##结构化任务##根据下文中信息，判断{field_name}是什么？请提取文中对应的值。{special_requirement}"""
    else:
        return ''
    print('prompt',prompt)
    print('content',query)
    res , (prompt_length, response_length)=  model.chat(query,[],prompt) 
    print('res',res)
    return res    


def is_numeric(value):
    try:
        float(value)  # 尝试将值转换为浮点数
        return True  # 如果转换成功，则表示值可以转换为数字
    except (ValueError, TypeError):
        return False  # 如果转换失败或者值的类型不是字符串或数字，则表示值不是数字



