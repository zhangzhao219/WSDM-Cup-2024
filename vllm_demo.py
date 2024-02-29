import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from swift.llm import (
    get_vllm_engine,
    get_template,
    inference_vllm,
    get_template,
    TemplateType,
    LoRATM,
    register_model,
)

from swift.utils import seed_everything, get_logger

from typing import Any, Dict
import torch
from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from torch import dtype as Dtype
from transformers.utils.versions import require_version

logger = get_logger()


@register_model(
    "output/solar-10-7b-instruct-v1/v0-20240113-031021/checkpoint-1500-merged",
    "output/solar-10-7b-instruct-v1/v0-20240113-031021/checkpoint-1500-merged",
    LoRATM.llama2,
    TemplateType.llama,
    support_vllm=True,
)
def get_model_tokenizer(
    model_dir: str,
    torch_dtype: Dtype,
    model_kwargs: Dict[str, Any],
    load_model: bool = True,
    **kwargs,
):
    print(model_dir)
    use_flash_attn = kwargs.pop("use_flash_attn", False)
    if use_flash_attn:
        require_version("transformers>=4.34")
        logger.info("Setting use_flash_attention_2: True")
        model_kwargs["use_flash_attention_2"] = True
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_config.pretraining_tp = 1
    model_config.torch_dtype = torch_dtype
    logger.info(f"model_config: {model_config}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = None
    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **model_kwargs,
        )
    return model, tokenizer


model_type = "output/solar-10-7b-instruct-v1/v0-20240113-031021/checkpoint-1500-merged"
llm_engine = get_vllm_engine(model_type)
template_type = TemplateType.llama
template = get_template(template_type, llm_engine.tokenizer)
seed_everything(42)
# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 512

request_list = [{"query": "你好!"}, {"query": "浙江的省会在哪？"}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")

history1 = resp_list[1]["history"]
request_list = [{"query": "这有什么好吃的", "history": history1}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
    print(f"history: {resp['history']}")
