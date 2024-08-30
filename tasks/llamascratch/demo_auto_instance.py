# 查看cuda版本并输出
# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# # 输出python版本
# import sys
# print(sys.version)

# from vllm import LLM, SamplingParams
# import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# prompts = [
#     "what is computer?",
# ]

# llm = LLM(
#     model="/data0/yfman/hf_models/Llama-3-8B-Instruct",
#     trust_remote_code=True,
#     gpu_memory_utilization=0.9,
# )

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# outputs = llm.generate(prompts, sampling_params)
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# import transformers
# import torch

# model_id = "/data0/yfman/hf_models/Llama-3-8B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

# messages = [
#     {
#         "role": "system",
#         "content": "You are a pirate chatbot who always responds in pirate speak!",
#     },
#     {"role": "user", "content": "Who are you?"},
# ]

# terminators = [
#     pipeline.tokenizer.eos_token_id,
#     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
# ]

# outputs = pipeline(
#     messages,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )
# print(outputs[0]["generated_text"][-1])


# from transformers import AutoTokenizer, LlamaForCausalLM
# import torch

# # Step 1: 设置设备
# device = "cuda:6" if torch.cuda.is_available() else "cpu"

# # Step 2: 初始化 tokenizer 和模型，并将模型移动到指定设备
# tokenizer = AutoTokenizer.from_pretrained("/data0/yfman/hf_models/Llama-3-8B-Instruct", legacy=False)
# model = LlamaForCausalLM.from_pretrained("/data0/yfman/hf_models/Llama-3-8B-Instruct").to(device)

# # Step 3: 编写生成文本的函数
# def generate_text(prompt, max_length=50):
#     # 编码输入文本并移动到指定设备
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
#     input_ids = inputs['input_ids']
#     attention_mask = inputs['attention_mask']

#     # 使用模型生成文本
#     with torch.no_grad():
#         output = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length, do_sample=True)

#     # 解码生成的输出
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

#     return generated_text

# # Step 4: 设置提示文本并生成文本
# prompt_text = "Once upon a time in a land far away"
# generated_text = generate_text(prompt_text, max_length=50)

# # Step 5: 打印生成的文本
# print("Generated Text:", generated_text)

# from transformers import AutoTokenizer, LlamaForCausalLM

# model = LlamaForCausalLM.from_pretrained("/data0/xiac/hf_models/Llama-3-8B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("/data0/xiac/hf_models/Llama-3-8B-Instruct")

# prompt = "Hey, are you conscious? Can you talk to me?"
# inputs = tokenizer(prompt, return_tensors="pt")

# inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id).int()

# # Generate
# generate_ids = model.generate(
#     inputs.input_ids,
#     attention_mask=inputs.attention_mask,
#     max_length=30,
#     pad_token_id=tokenizer.pad_token_id,
# )
# output = tokenizer.batch_decode(
#     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )[0]

# # 打印输出
# print(output)


# from transformers import AutoTokenizer, LlamaForCausalLM

# # Load model and tokenizer
# model = LlamaForCausalLM.from_pretrained("/data0/xiac/hf_models/llama2-7B-hf")
# tokenizer = AutoTokenizer.from_pretrained("/data0/xiac/hf_models/llama2-7B-hf")

# # Ensure the pad_token_id is set correctly
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id

# # Set input prompt
# prompt = "Hey, are you conscious? Can you talk to me?"
# inputs = tokenizer(prompt, return_tensors="pt")

# # Set the attention_mask using the pad_token_id
# inputs['attention_mask'] = inputs['input_ids'].ne(tokenizer.pad_token_id).int()

# # Generate text
# generate_ids = model.generate(
#     inputs.input_ids, 
#     attention_mask=inputs.attention_mask, 
#     max_length=30, 
#     pad_token_id=tokenizer.pad_token_id
# )

# # Decode generated text and print output
# output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(output)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
# device = "cuda:6"
model_id = "/data0/xiac/hf_models/Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("model.device:", model.device)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
