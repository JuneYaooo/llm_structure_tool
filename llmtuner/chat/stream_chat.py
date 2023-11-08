import torch
from typing import Any, Dict, Generator, List, Optional, Tuple
from threading import Thread
from transformers import GenerationConfig, TextIteratorStreamer

from llmtuner.extras.misc import dispatch_model, get_logits_processor
from llmtuner.extras.template import get_template_and_fix_tokenizer
from llmtuner.tuner.core import get_infer_args, load_model_and_tokenizer
import re

class ChatModel:

    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, self.generating_args = get_infer_args(args)
        self.model, self.tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
        self.tokenizer.padding_side = "left"
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer(data_args.template, self.tokenizer)
        self.system_prompt = data_args.system_prompt

    def process_args(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[Dict[str, Any], int]:
        system = system or self.system_prompt
        prompt, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer, query=query, resp="", history=history, system=system
        )
        prompt_length = len(prompt)
        input_ids = torch.tensor([prompt], device=self.model.device)

        do_sample = input_kwargs.pop("do_sample", None)
        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        num_return_sequences = input_kwargs.pop("num_return_sequences", None)
        repetition_penalty = input_kwargs.pop("repetition_penalty", None)
        max_length = input_kwargs.pop("max_length", None)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)

        generating_args = self.generating_args.to_dict()
        generating_args.update(dict(
            do_sample=do_sample if do_sample is not None else generating_args["do_sample"],
            temperature=temperature or generating_args["temperature"],
            top_p=top_p or generating_args["top_p"],
            top_k=top_k or generating_args["top_k"],
            num_return_sequences=num_return_sequences or 1,
            repetition_penalty=repetition_penalty or generating_args["repetition_penalty"],
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            pad_token_id=self.tokenizer.pad_token_id
        ))

        if isinstance(num_return_sequences, int) and num_return_sequences > 1:
            generating_args["do_sample"] = True

        if max_length:
            generating_args.pop("max_new_tokens", None)
            generating_args["max_length"] = max_length

        if max_new_tokens:
            generating_args.pop("max_length", None)
            generating_args["max_new_tokens"] = max_new_tokens

        gen_kwargs = dict(
            inputs=input_ids,
            generation_config=GenerationConfig(**generating_args),
            logits_processor=get_logits_processor()
        )

        return gen_kwargs, prompt_length

    @torch.inference_mode()
    def chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[List[str], Tuple[int, int]]:
        gen_kwargs, prompt_length = self.process_args(query, history, system, **input_kwargs)
        generate_output = self.model.generate(**gen_kwargs)
        outputs = generate_output.tolist()[0][prompt_length:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response_length = len(outputs)
        response=re.sub(r'Helper: ?', '', response)
        return response, (prompt_length, response_length)
    
    @torch.inference_mode()
    def batch_chat(
        self,
        query: List[str],
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[List[str], Tuple[int, int]]:
        gen_kwargs, prompt_length = self.process_args(query, history, system, **input_kwargs)
        generate_output = self.model.generate(**gen_kwargs)
        response_ids = generate_output[:, prompt_length:]
        response = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        response_length = 0
        for i in range(len(response_ids)):
            eos_index = (response_ids[i] == self.tokenizer.eos_token_id).nonzero()
            response_length += eos_index[0].item() if len(eos_index) else len(response_ids[i])
        return response, (prompt_length, response_length)

    @torch.inference_mode()
    def stream_chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Generator[str, None, None]:
        gen_kwargs, _ = self.process_args(query, history, system, **input_kwargs)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        yield from streamer

if __name__ == "__main__":
        # Define the command-line arguments as a dictionary
        args = {
            "stage": "sft",
            "model_name_or_path": "/mnt/data/smart_health_02/yaoyujun/models/chatglm3-6b",
            "template": "chatglm3",
            "finetuning_type": "lora",
            "lora_target": "query_key_value",
            "checkpoint_dir": "/mnt/data/smart_health_02/yaoyujun/github/LLaMA-Efficient-Tuning/checkpoints/chatglm3-6b_1102"
        }

        # Create an instance of ChatModel with the args dictionary
        chat_model = ChatModel(args)

        # Test the chat function
        query = "图像质量：乙级；检查途径：经体表；检查项目：二维、 M型、多普勒（彩色、 脉冲式、连续式）         心脏M型及二维测值：（单位mm）                      M型左心功能测定         名称        测量值   正常参考值         名称           测量值      主动脉根部内径：    33     20-37        左室舒张末期容量：    120                 左房内径：    46     19-40        左室收缩末期容量：     44          左室舒张期内径：    50     35-56         左室射血分数(%)：     64          左室收缩期内径：    33     20-37       左室短轴缩短率(%)：     35              室间隔厚度：    13     6-11           每搏输出量(ml):      77                左室后壁厚度：    11     6-11            (注:女性左房内径参考值:16-37、左室舒张期内径参考值:32-53、左室收缩期内径参考值:17-34) 一、二维灰阶超声描述：  1.左房增大，余房室腔内径正常范围。大动脉关系、内径正常。  2.室间隔增厚，余室壁厚度正常；静息状态下左室壁各节段收缩活动未见明显异常。  3.主动脉瓣局部增厚，回声增强，开放不受限，余心瓣膜形态、结构、启闭运动未见明显异常。  4.心包腔未见明显异常。二、彩色及频谱多普勒超声描述：  1.房、室间隔水平未见明显分流。  2.二尖瓣未见明显反流。舒张期经二尖瓣口血流:E= 54cm/s， A= 84cm/s，E/A=0.6。  3.主动脉瓣可见轻微反流。  4.三尖瓣可见轻度反流，反流峰值流速约2.0m/s，跨瓣压差16mmHg。  5.肺动脉瓣可见轻微反流。三、组织多普勒检查：   二尖瓣瓣环水平：室间隔侧 E'=  4cm/s，E/E'= 14。                   左室侧壁 E'=  8cm/s，E/E'=  7。"
        response, (prompt_length, response_length) = chat_model.chat(query,[],'##结构化任务##根据下文中信息，判断主动脉根部内径是什么？请提取文中对应的值。')
        print("User:", query)
        print("Response:", response)
        print("Prompt Length:", prompt_length)
        print("Response Length:", response_length)

