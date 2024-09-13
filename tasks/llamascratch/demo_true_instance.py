from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
import torch

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
# device = "cuda:6"
model_id = "/data0/xiac/hf_models/Llama-3-8B-Instruct"

# Use LlamaTokenizer and LlamaForCausalLM instead of the auto classes
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("model.device:", model.device)

# messages = [
#     {
#         "role": "system",
#         "content": "You are a pirate chatbot who always responds in pirate speak!",
#     },
#     {"role": "user", "content": "Who are you?"},
# ]

# input_ids = tokenizer.apply_chat_template(
#     messages, add_generation_prompt=True, return_tensors="pt"
# ).to(model.device)

prompt = "What is the meaning of life?"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

outputs = model.generate(
    input_ids,
    max_new_tokens=128,
    #    eos_token_id=terminators,
    do_sample=False,
    # temperature=0.01,
    #    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1] :]
print(tokenizer.decode(response, skip_special_tokens=True))
