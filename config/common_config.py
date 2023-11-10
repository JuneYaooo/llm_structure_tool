
## 这里配置本地可用的开源模型
llm_model_dict = {
"PULSE": {"name": "pulse",
        "model_path": "/path/to/your/model",
        "template":"default",
        "lora_target":"query_key_value",
        "per_device_train_batch_size":2
    },
"InternLM": {"name": "internlm",
        "model_path": "/path/to/your/model",
        "template":"intern",
        "lora_target":"q_proj,v_proj",
        "per_device_train_batch_size":2
    },
"ChatGLM2": {"name": "chatglm2",
        "model_path": "/path/to/your/model",
        "template":"chatglm2",
        "lora_target":"query_key_value",
        "per_device_train_batch_size":4
    },
"ChatGLM3": {"name": "chatglm3",
        "model_path": "/path/to/your/model",
        "template":"chatglm3",
        "lora_target":"query_key_value",
        "per_device_train_batch_size":4
    },
}

# 找到 profile.d/conda.sh 文件的绝对路径，填进来
conda_env_file = '/path-to-your-conda/etc/profile.d/conda.sh'

# 生成参数
max_length=1500
do_sample=False
temperature=0