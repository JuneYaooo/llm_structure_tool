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
from config.common_config import *
model_loaded = False
lora_change = False
last_lora_name = ''
max_new_tokens = 1500
generation_config = dict(
            temperature=0.001,
            top_k=30,
            top_p=0.85,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.2,
            max_new_tokens=max_new_tokens
            )
def read_excel_file(file_path):
    df = pd.read_excel(file_path)
    return df

def save_to_excel(df, file_path):
    df.to_excel(file_path, index=False)

def process_data(training_data_path):
     # 读取 Excel 文件
    df = pd.read_excel(training_data_path)
    log = []
    log.append(f'开始处理数据')
    
    all_data = []
    # 遍历每一行数据
    for index, row in df.iterrows():
        instruction = row['系统指示']
        question = row['问题']
        answer = row['回答']
        
        # 创建字典并将数据添加到列表中
        data = {"instruction": instruction, "input": question, "output": answer}
        all_data.append(data)

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

def pulse_train_model(model_name, lora_name,  training_data_path):
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    print('now_str',now_str)
    all_data,log = process_data(training_data_path)
    log_file_path = f'data/logs/{now_str}.log'  # 定义log文件路径
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # 创建存储log的文件夹

    with open(log_file_path, 'w', encoding="utf-8") as f:
        f.write(log)  # 将log内容写入文件
    with open(f"data/{lora_name}_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4,  ensure_ascii=False)
    if not os.path.exists('finetune/pulse/data'):
        os.makedirs('finetune/pulse/data')
    if not os.path.exists('finetune/pulse/logs'):
        os.makedirs('finetune/pulse/logs')
    shutil.copyfile(f"data/{lora_name}_dataset.json", f"finetune/pulse/data/{lora_name}_dataset.json")
    
    available_gpus = get_available_gpu(threshold=20000)
    print('available_gpus[0]',available_gpus[0])
    content = f'''python convert_to_conv_data.py --orig_data data/{lora_name}_dataset.json --write_data  data/{lora_name}_dataset_conv.json --dataset_name {lora_name}

CUDA_VISIBLE_DEVICES={available_gpus[0]} torchrun --nproc_per_node 1 finetune.py --model_name_or_path {llm_model_dict[model_name]["local_model_path"]} --use_lora True --use_int8_training --lora_config configs/lora_config_bloom.json --train_file data/{lora_name}_dataset_conv.json --validation_file data/{lora_name}_dataset_conv.json --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 2 --num_train_epochs 2 --model_max_length 100 --save_strategy "steps" --save_total_limit 3 --learning_rate 3e-4 --weight_decay 0.00001 --warmup_ratio 0.05 --lr_scheduler_type "cosine" --logging_steps 10 --evaluation_strategy "steps" --seed 2048 --gradient_checkpointing True --cache_dir cache/{lora_name} --output_dir output/{lora_name}
    '''
    sh_file_name = f'finetune/pulse/train_{lora_name}.sh'

    with open(sh_file_name , 'w') as file:
        file.write(content)

    # 设置文件可执行权限
    os.chmod(sh_file_name , 0o755)
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    print('now_str',now_str)
    subprocess.Popen(f"""cd finetune/pulse && . /home/pai/etc/profile.d/conda.sh && conda activate med_llm && nohup sh train_{lora_name}.sh > ./logs/train_{now_str}.log 2>&1 &""", shell=True)
    print('finish')
    # model.train(training_data_path)
    return f'{model_name} on training'

def extract_content(replacement_text, string):
    pattern_helper = r'Helper:(.*?)</s>' # r'Helper:(.*?)(?=<\/s>)'

    match_helper = re.findall(pattern_helper, string, re.DOTALL)

    if match_helper:
        content = match_helper[-1].strip()
        return content.replace('</s>', '').replace('<\s>', '')
    else:
        replaced_string = re.sub(r'Input:.*?(?=\n)', replacement_text, string, re.DOTALL)
        replaced_string = replaced_string.replace('</s>', '').replace('<\s>', '')
        return  replaced_string

import re

def pred_res(prompt,model,tokenizer):
    model_device = next(model.parameters()).device
    inputs = tokenizer([prompt], return_tensors="pt").input_ids.to(model_device) 
    re_token_ids = model.generate(
        inputs=inputs, 
        do_sample=False,
        temperature=0,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2000 #512,
    )
    response = tokenizer.decode(re_token_ids[0])
    response = extract_content(prompt, response)
    import torch
    torch.cuda.empty_cache()
    return response

def pulse_infer_model(model_name, lora_name, config_path, test_data_path):
    try:
        model, tokenizer = load_model(lora_name)
        if model == 'no enough GPU, please try it later!':
            return 'no enough GPU, please try it later!'

        medical_logic = pd.read_excel(config_path)
        source = medical_logic['文本来源'].unique().tolist()
        source_table = list(set([s.split('.')[0] for s in source if isinstance(s, str) and '.' in s and pd.notnull(s)]))
        while '' in source_table:
            source_table.remove('')
        print('source_table',source_table)

        source_table_dict = {}
        tables = {}
        for i, st in enumerate(source_table):
            source_table_dict[f'table{i}'] = st
            tables[f'table{i}'] = pd.read_excel(test_data_path, sheet_name=st) #data4-raw-0324.xlsx
        flipped_source_table_dict = {v: k for k, v in source_table_dict.items()}


        start_time = time.time()
        def get_result(lora_name,row,context,field,arr):
            prompt = ''
            if row['值域类型'] == '多选':
                prompt = f"""##结构化任务##根据下文中信息，判断{arr}{row['字段名']}是什么？请在值域【{row['值域']}】中选择提到的所有内容。"""
            elif row['值域类型'] == '单选':
                prompt = f"##结构化任务##根据下文中信息，判断{arr}{row['字段名']}是什么？请在值域【{row['值域']}】中选择1个。"
            elif row['值域类型'] == '提取':
                prompt = f"""##结构化任务##根据下文中信息，判断{arr}{row['字段名']}是什么？请提取文中对应的值"""
            else:
                return ''
            print('prompt',prompt)
            print('context',context)
            res = pred_res(prompt,model,tokenizer)
            
            print('res',res)
            return res


        from collections import defaultdict

        # create a dictionary to store rows with the same '属性提取'
        attr_dict = defaultdict(list)

        for i, row in medical_logic.iterrows():
            if row['文本来源'] =='' or pd.isnull(row['文本来源']):
                print('文本来源为空，跳过字段：',row['字段名'] )
                continue
            table_name, text_field = row['文本来源'].split('.')[0], row['文本来源'].split('.')[1]
            table = tables[flipped_source_table_dict[table_name]]
            field = row['字段名']
            if pd.notnull(row['属性提取']):
                arr_table,arr_field = row['属性提取'].split('.')[0],row['属性提取'].split('.')[1]
                if pd.notnull(row['属性前缀']):
                    arr_prefix = row['属性前缀']
                else:
                    arr_prefix = ''

                arr_row = medical_logic.loc[medical_logic['字段名'] == arr_field].iloc[0]
                print('arr_table',arr_table,arr_field)
                arr_table = tables[flipped_source_table_dict[arr_table]]
                for k, row2 in arr_table.iterrows():
                    context = str(row2[text_field]).replace('_x000D_', '').replace('\n', '')
                    arr_content = get_result(lora_name,arr_row,context,arr_field,'')
                    arr_list = arr_content.split(',')
                    for arr in arr_list:
                        print('arr',arr)
                        if arr and arr !='未提及'and arr !='其他':
                            result = get_result(lora_name,row,context,field,arr+arr_prefix)
                            print('arr result',arr+arr_prefix,field,result)
                            # add the row to the dictionary with the same '属性提取'
                            attr_dict[arr_field].append({'content':context,arr_field:arr,field:result})
            else:
                for j, row2 in table.iterrows():
                    arr = '' if table_name != '检查记录-心电' else '心电图'
                    context = str(row2[text_field]).replace('_x000D_', '').replace('\n', '')
                    result = get_result(lora_name,row,context,field, arr)
                    table.loc[j, field] = result 
                    continue
                pass
            tables[table_name] = table


        with pd.ExcelWriter(r'./data/result.xlsx') as writer:
            for st in source_table:
                print('flipped_source_table_dict[st]', flipped_source_table_dict[st])
                table_name = flipped_source_table_dict[st]
                table = tables[table_name]
                table.to_excel(writer, sheet_name=st, index=False)
            for attr, rows in attr_dict.items():
                # create a new table for rows with the same '属性提取'
                new_table = pd.DataFrame(rows)
                new_table = new_table.groupby(["content",attr], as_index=False).agg('first')
                # save the new table to a file with the name of the '属性提取'
                new_table.to_excel(writer, sheet_name=attr, index=False)

        end_time = time.time()
        print(f'cost {round(end_time-start_time,2)} seconds')
         
        return 'data/result.xlsx','success'
    except Exception as e:
        print('error!! ', e)
        return '', e

def load_model(model_name,lora_name):
    global model_loaded, model, tokenizer, lora_change, last_lora_name
    print('lora_name, last_lora_name',lora_name, last_lora_name)
    
    if lora_name != last_lora_name:
        lora_change = True
    available_gpus = get_available_gpu(threshold=15000)
    if not model_loaded or lora_change:
        if len(available_gpus)>0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
            print('available_gpus[0]',available_gpus[0])
        else:
            return 'no enough GPU, please try it later!','',''
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
            load_type = torch.float16
            if '不使用' in lora_name:
                tokenizer = AutoTokenizer.from_pretrained(
                    llm_model_dict[model_name]["local_model_path"],
                )
                model = AutoModelForCausalLM.from_pretrained(
                    llm_model_dict[model_name]["local_model_path"], 
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            else:
                from peft import  PeftModel
                base_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_dict[model_name]["local_model_path"], 
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                model = PeftModel.from_pretrained(base_model, f"finetune/pulse/output/{lora_name}", torch_dtype=torch.float16)
                tokenizer = AutoTokenizer.from_pretrained(llm_model_dict[model_name]["local_model_path"], trust_remote_code=True)
                tokenizer.pad_token_id = 0
                tokenizer.eos_token_id = 2
                model_config = AutoConfig.from_pretrained(llm_model_dict[model_name]["local_model_path"])
                model = AutoModelForCausalLM.from_pretrained(
                f"finetune/pulse/output/{lora_name}", 
                torch_dtype=load_type,
                config=model_config,
                device_map='auto'
                )

            model.eval()
            print("Load model successfully")
            model_loaded = True
            last_lora_name = lora_name
            lora_change = False
        except Exception as e:
            print('error!! ', e)
            return e,'',''
    return model, tokenizer
import re

def remove_prefix_suffix(string):
    pattern = r'^((?:Helper:|:)\s*)(.*?)(</s>)?$'
    string = re.sub(pattern, r'\2', string, flags=re.DOTALL)
    string = remove_starting_symbols(string).replace('</s>', '').replace('<\s>', '')
    return string

def clean_res(value, string):
    pattern = r"{}[:：](.*)".format(re.escape(value))
    match = re.search(pattern, string)
    if match:
        extracted_content = match.group(1).strip()
        if "未提及" in extracted_content:
            return "未提及"
        elif not extracted_content:
            return "未获取有效值"
        else:
            return extracted_content
    else:
        if "未提及" in string:
            return "未提及"
        else:
            return "未获取有效值"

def pulse_query(model_name,lora_name, field_type, field_name, value_range, query):
    print('222 pulse start query')
    model, tokenizer = load_model(model_name,lora_name)
    if model == 'no enough GPU, please try it later!':
        return 'no enough GPU, please try it later!'
    prompt = ''
    if field_type == '多选':
        prompt = f"""</s>User:根据以下医疗报告，从值域中选择{field_name}表述的所有内容，结果输出值域中的内容
值域：【{value_range}】

报告内容：{query}

所需信息答案的输出格式：
"
{field_name}：xxx
"

确保所提取的答案存在于报告中，如所需信息没有提到，则答案为“未提及”。
确保从我给的值域中选择值。
</s>Helper:"""
    elif field_type == '单选':
        prompt = f"""</s>User:根据以下医疗报告，从值域中选择{field_name}表述的一个内容，结果输出值域中的内容
值域：【{value_range}】

报告内容：{query}

所需信息答案的输出格式：
"
{field_name}：xxx
"

确保所提取的答案存在于报告中，如所需信息没有提到，则答案为“未提及”。
确保从我给的值域中选择一个值。
</s>Helper:"""
    elif field_type == '提取':
        prompt = f"""</s>User:根据以下医疗报告，提取以下信息：{field_name}

信息提取要求：
信息的答案请根据理解，找出所需信息的答案。
如果所需信息未提到，则答案为“未提及”。

报告内容：{query}

所需信息答案的输出格式：
"
{field_name}：xxx
"
依照答案格式，仅需输出答案。
确保输出格式按照所给答案格式。
确保所提取的答案存在于报告中，如所需信息没有提到，则答案为“未提及”。
确保没有多余信息输出，我只需要一个信息的提取。
</s>Helper:"""
    else:
        return ''
    print('prompt',prompt)
    res = pred_res(prompt,model,tokenizer)
    print('res1',res)
    res = clean_res(field_name, res)
    print('res2',res)
    return res    
