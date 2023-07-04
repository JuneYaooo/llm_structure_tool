import gradio as gr
import os
from finetune.pulse_utils import  pulse_infer_model, pulse_query, pulse_train_model
import shutil
import time
import datetime
from config.common_config import *
from utils.utils import stop_train_process,evaluate_model

def get_file_modify_time(filename):
    try:
        return datetime.datetime.fromtimestamp(os.stat(filename).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print('Failed to get modification time for {}'.format(filename))
        print(e)
        return 'not available'

def get_model_update_time(model_name, lora_name):
    if 'pulse' in model_name.lower():
        update_time = get_file_modify_time(f"finetune/pulse/output/{lora_name}/adapter_model.bin")
    else:
        update_time = 'not available'
    return update_time

def on_train(model_name, lora_name, training_data_file):
    training_data_path = 'data/'+os.path.basename(training_data_file.name)
    if 'pulse' in model_name.lower():
        msg = pulse_train_model(model_name, lora_name, training_data_path)
    else:
        msg = 'please select one model!'
    return msg

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60,2)
    if hours > 0:
        return f"{hours}时{minutes}分{seconds}秒"
    elif minutes > 0:
        return f"{minutes}分{seconds}秒"
    else:
        return f"{seconds}秒"


def on_test(model_name, select_lora, config_file, test_data_file):
    start_time = time.time()
    config_path = 'data/'+os.path.basename(config_file.name)
    test_data_path = 'data/'+os.path.basename(test_data_file.name)
    if 'pulse' in model_name.lower():
        result_path,info = pulse_infer_model(model_name, select_lora, config_path, test_data_path)
    else:
        result_path,info = '',''
    end_time = time.time()
    cost_time = end_time-start_time
    
    info = '用时：'+format_duration(cost_time)+f"  ({round(cost_time,2)}秒)" if info=='success' else info
    return result_path,info

def on_evaluate(model_name, select_lora, test_result_file, test_label_file):
    test_result_path = 'data/'+os.path.basename(test_result_file.name)
    test_label_path = 'data/'+os.path.basename( test_label_file.name)
    result_path = evaluate_model(test_result_path, test_label_path)
    return result_path

def on_query(model_name,select_lora, field_type, field_name, value_range, query):
    if 'pulse' in model_name.lower():
        res = pulse_query(model_name, select_lora, field_type, field_name, value_range, query) 
    else:
        res = 'not availabel'
    return res

def on_stop(model_name,select_lora):
    res = stop_train_process() 
    return res

def upload_file(file):
    print('file',file)
    if not os.path.exists("data"):
        os.mkdir("data")
    filename = os.path.basename(file.name)
    shutil.move(file.name, "data/" + filename)
    # file_list首位插入新上传的文件
    filedir = "data/" + filename
    return filedir

def change_lora_name_input(model_name,lora_name_en):
    if lora_name_en == "新建Lora":
        return gr.update(visible=True), gr.update(visible=True)
    else:
        file_status = f"已加载{lora_name_en}"
        model_update_time = get_model_update_time(model_name, lora_name_en)
        return gr.update(visible=False), gr.update(visible=False), model_update_time


def add_lora(lora_name_en,lora_list):
    if lora_name_en in lora_list:
        print('名称冲突，不新建')
        return gr.update(visible=True,value=lora_name_en), gr.update(visible=False), gr.update(visible=False), lora_list
    else:
        return gr.update(visible=True, choices=[lora_name_en] + lora_list, value=lora_name_en), gr.update(visible=False), gr.update(visible=False),[lora_name_en] + lora_list

def find_folders(directory):
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders

def get_lora_init_list(model_name):
    if 'pulse' in model_name.lower():
        model_name = 'pulse'
    else:
        model_name = ''
    model_dir = os.path.join(f"finetune", model_name,'output')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    lora_list = find_folders(model_dir)
    return lora_list

llm_model_dict_list = list(llm_model_dict.keys())
lora_init_list = get_lora_init_list(llm_model_dict_list[0])

webui_title = """
# 🎉病历结构化🎉

可以选择案例测试和[使用excel配置文件进行训练-预测-评估](https://zg0f0ipp6j.feishu.cn/wiki/XC16wwvGgiVSNbkzSPUczqFMn6e)
"""

def create_tab():
    # 初始化
    with gr.Blocks() as demo:
        set_lora_list = gr.State(lora_init_list+['不使用'])
        gr.Markdown(webui_title)
        with gr.Row():
            with gr.Column():
                model_name = gr.Radio(llm_model_dict_list, 
                                            label="选择模型",
                                            value= llm_model_dict_list[0] if len(llm_model_dict_list)>0 else '暂无可选模型',
                                            interactive=True)
            with gr.Column():
                select_lora = gr.Dropdown(set_lora_list.value+['新建Lora'],
                                        label= "选择或者新建一个Lora",
                                        value= "不使用", #set_lora_list.value[0] if len(set_lora_list.value) > 0 else '新建Lora',
                                        interactive=True,
                                        visible=False)
                # lora_name_en = gr.Textbox(label="请输入Lora英文名称，中间不能有空格，小写字母，单词间可用下划线分开",
                #                             lines=1,
                #                             interactive=True,
                #                             visible=True)
                # lora_add = gr.Button(value="确认添加Lora", visible=True)
        with gr.Row():
            lastest_model = gr.outputs.Textbox(type="text", label='模型更新时间（请切换模型或项目刷新显示）')
        with gr.Tab("案例测试"):
            with gr.Column():
                gr.Markdown(f"初次加载模型可能比较慢，后续会变快")
                field_type = gr.Radio(['单选','多选','提取'],
                                        label="字段类型",
                                        value='提取',
                                        interactive=True)
                field_name = gr.Textbox(label="字段名",
                                         lines=1,
                                         interactive=True)
                value_range = gr.Textbox(label="请输入值域，以','分隔开（对于提取不必输入值域）",
                                         lines=1,
                                         interactive=True)
                query = gr.Textbox(label="请输入原文",
                                         lines=1,
                                         interactive=True)
                query_button = gr.Button(label="获得结果")
                query_res = gr.outputs.Textbox(type="text", label='')

        # with gr.Tab("训练-预测-评估", visible=False):
        #     gr.Markdown(f"""
        #     Step1:选择一个Lora
        #     Step2:根据任务选择训练 预测或评估，上传对应的参数文件或者数据标准文件，请等待文件上传成功后再开始执行！""")
        #     with gr.Row():
        #         with gr.Column():
        #             gr.Markdown("## 训练")
        #             train_config_file = gr.File(label="上传配置文件", file_types=['.xlsx'])
        #             train_data_file = gr.File(label="上传标注数据文件", file_types=['.xlsx'])
        #             train_button = gr.Button("开始训练", label="训练")
        #             kill_train_button = gr.Button("停止所有训练进程", label="训练")
        #             train_res = gr.outputs.Textbox(type="text", label='')
                    

        #         with gr.Column():
        #             gr.Markdown("## 预测")
        #             test_config_file = gr.File(label="上传配置文件", file_types=['.xlsx'])
        #             test_data_file = gr.File(label="上传测试数据文件", file_types=['.xlsx'])
        #             test_button = gr.Button(label="评估")
        #             test_res = gr.outputs.Textbox(type="text", label='')
        #             download_test = gr.outputs.File(label="下载结果文件")

        #         with gr.Column():
        #             gr.Markdown("## 评估")
        #             test_result_file = gr.File(label="上传测试结果文件", file_types=['.xlsx'])
        #             test_label_file = gr.File(label="上传标准结果文件", file_types=['.xlsx'])
        #             evaluate_button = gr.Button(label="评估")
        #             download_evaluate = gr.outputs.File(label="下载评估结果")

        # select_lora.change(fn=change_lora_name_input,
        #                              inputs=[model_name,select_lora],
        #                              outputs=[lora_name_en, lora_add,lastest_model])
        # lora_add.click(fn=add_lora,
        #                          inputs=[lora_name_en,set_lora_list],
        #                          outputs=[select_lora, lora_name_en, lora_add,set_lora_list])
        model_name.change(fn=get_model_update_time, inputs=[model_name, select_lora], outputs=lastest_model)
        # train_config_file.upload(upload_file,
        #         inputs=train_config_file)
        # train_data_file.upload(upload_file,
        #         inputs=train_data_file)  
        # test_config_file.upload(upload_file,
        #         inputs=test_config_file)
        # test_data_file.upload(upload_file,
        #         inputs=test_data_file)
        # test_result_file.upload(upload_file,
        #         inputs=test_result_file)
        # test_label_file.upload(upload_file,
        #         inputs=test_label_file)
        # train_button.click(on_train, inputs=[model_name, select_lora, train_config_file, train_data_file],outputs=[train_res]) 
        # kill_train_button.click(on_stop, inputs=[model_name],outputs=[train_res]) 
        # test_button.click(on_test,show_progress=True, inputs=[model_name, select_lora, test_config_file, test_data_file], outputs=[download_test,test_res]) 
        # evaluate_button.click(on_evaluate,show_progress=True, inputs=[model_name, select_lora,test_result_file, test_label_file], outputs=[download_evaluate]) 
        query_button.click(on_query,show_progress=True, inputs=[model_name, select_lora, field_type, field_name, value_range, query], outputs=[query_res]) 
    return demo

tab = create_tab() 

if __name__ == "__main__":
    tab.queue(concurrency_count=5).launch(server_name='0.0.0.0',server_port=33366,share=True, inbrowser=True)  #
