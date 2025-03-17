import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "vitumafeni/tiny-crypto-sentiment-analysis"


def create_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def create_model(model_name: str = MODEL_NAME) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )


def predict(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 16,
    return_full_text: bool = False,
) -> str:
    encoding = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs=encoding, max_new_tokens=max_new_tokens)
    if outputs.numel() == 0:
        return ""
    prediction = outputs[0]
    if not return_full_text:
        prediction = prediction[encoding.shape[1] :]
    return tokenizer.decode(prediction, skip_special_tokens=True).strip()
