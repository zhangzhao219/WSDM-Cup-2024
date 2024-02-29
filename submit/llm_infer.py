from swift.llm import infer_main

from typing import Any, Dict

from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from torch import dtype as Dtype
from transformers.utils.versions import require_version

from swift.llm import LoRATM, TemplateType, register_model
from swift.utils import get_logger

logger = get_logger()


class CustomModelType:
    SOLAR_10_7B_v1_instruct = "solar-10-7b-instruct-v1"

@register_model(
    CustomModelType.SOLAR_10_7B_v1_instruct,
    "upstage/SOLAR-10.7B-Instruct-v1.0",
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


if __name__ == "__main__":
    result = infer_main()
