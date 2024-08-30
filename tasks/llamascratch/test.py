from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from llama import Engine
import torch
import os

# very_long_text = "A path from a point approximately 330 metres east of the most south westerly corner of 17 Batherton Close, Widnes and approximately 208 metres east-south-east of the most southerly corner of Unit 3 Foundry Industrial Estate, Victoria Street, Widnes, proceeding in a generally east-north-easterly direction for approximately 28 metres to a point approximately 202 metres east-south-east of the most south-easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street, and approximately 347 metres east of the most south-easterly corner of 17 Batherton Close, then proceeding in a generally northerly direction for approximately 21 metres to a point approximately 210 metres east of the most south-easterly corner of Unit 5 Foundry Industrial Estate, Victoria Street, and approximately 202 metres east-south-east of the most north-easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-north-east direction for approximately 64 metres to a point approximately 282 metres east-south-east of the most easterly corner of Unit 2 Foundry Industrial Estate, Victoria Street, Widnes and approximately 259 metres east of the most southerly corner of Unit 4 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-north-east direction for approximately 350 metres to a point approximately 3 metres west-north-west of the most north westerly corner of the boundary fence of the scrap metal yard on the south side of Cornubia Road, Widnes, and approximately 47 metres west-south-west of the stub end of Cornubia Road be diverted to a 3 metre wide path from a point approximately 183 metres east-south-east of the most easterly corner of Unit 5 Foundry Industrial Estate, Victoria Street and approximately 272 metres east of the most north-easterly corner of 26 Ann Street West, Widnes, then proceeding in a generally north easterly direction for approximately 58 metres to a point approximately 216 metres east-south-east of the most easterly corner of Unit 4 Foundry Industrial Estate, Victoria Street and approximately 221 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally easterly direction for approximately 45 metres to a point approximately 265 metres east-south-east of the most north-easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 265 metres east of the most southerly corner of Unit 5 Foundry Industrial Estate, Victoria Street, then proceeding in a generally east-south-east direction for approximately 102 metres to a point approximately 366 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 463 metres east of the most north easterly corner of 22 Ann Street West, Widnes, then proceeding in a generally north-north-easterly direction for approximately 19 metres to a point approximately 368 metres east-south-east of the most easterly corner of Unit 3 Foundry Industrial Estate, Victoria Street and approximately 512 metres east of the most south easterly corner of 17 Batherton Close, Widnes then proceeding in a generally east-south, easterly direction for approximately 16 metres to a point approximately 420 metres east-south-east of the most southerly corner of Unit 2 Foundry Industrial Estate, Victoria Street and approximately 533 metres east of the most south-easterly corner of 17 Batherton Close, then proceeding in a generally east-north-easterly direction for approximately 240 metres to a point approximately 606 metres east of the most northerly corner of Unit 4 Foundry Industrial Estate, Victoria Street and approximately 23 metres south of the most south westerly corner of the boundary fencing of the scrap metal yard on the south side of Cornubia Road, Widnes, then proceeding in a generally northern direction for approximately 44 metres to a point approximately 3 metres west-north-west of the most north westerly corner of the boundary fence of the scrap metal yard on the south side of Cornubia Road and approximately 47 metres west-south-west of the stub end of Cornubia Road."

very_long_text = "What is the meaning of life?"

# weight_path = os.getenv("HF_MODELS_CACHE")
weight_path = "/data0/xiac/hf_models/Llama-3-8B-Instruct"
if weight_path is None:
    raise ValueError("HF_MODELS_CACHE environment variable is not set")
print("weight_path:", weight_path)
print("Type of weight_path:", type(weight_path))
# tokenizer = LlamaTokenizer.from_pretrained(weight_path)
tokenizer = AutoTokenizer.from_pretrained(weight_path)
# tokenizer = PreTrainedTokenizerFast.from_pretrained(weight_path)
hf_model = LlamaForCausalLM.from_pretrained(weight_path, torch_dtype=torch.bfloat16).to(
    "cuda:5"
)
assert type(hf_model) == LlamaForCausalLM
# prompt = f"Content: {very_long_text}\n\n Summary:"
prompt = very_long_text
prompt_ids = (
    tokenizer(prompt, return_tensors="pt")["input_ids"].view(1, -1).to(hf_model.device)
)
max_new_tokens = 128
gen_config = GenerationConfig(
    max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True
)

output_ids = hf_model.generate(prompt_ids, gen_config).sequences  # type:ignore
# output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
output_text = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(output_ids[0])
)

del tokenizer, hf_model, prompt_ids, gen_config

print("Output text:", output_text)

engine = Engine(weight_path)
llama_output = engine.execute([prompt])[0]
# assert engine.prompt_ids == prompt_ids
print("Llama output:", llama_output)
assert output_text == llama_output
