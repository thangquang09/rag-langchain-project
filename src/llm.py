from constant import model_name, max_new_tokens

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline


def get_model(
    model_name: str = model_name,
    max_new_tokens: int = max_new_tokens,
    **kwargs
):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
  
    # Nếu không sử dụng multi-platform của bitsanbytes thì sẽ lỗi khi thực hiện quantization trên CPU
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=nf4_config,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print("No quantization because:", e)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # quantization_config=nf4_config,
            low_cpu_mem_usage=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs=kwargs
    )
    return llm

if __name__ == "__main__":
    pass