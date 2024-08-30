from transformers import AutoTokenizer, GenerationConfig
from my_modeling_llama import LlamaForCausalLM
import torch


class Engine:
    def __init__(self, model_path: str) -> None:
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        ).to("cuda:6")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.prompt_ids = None

    def execute(
        self, prompts: list[str], max_new_tokens: int = 128, temperature=0.001
    ) -> list[str]:
        self.prompt_ids = (
            self.tokenizer(prompts[0], return_tensors="pt")["input_ids"]
            .view(1, -1)
            .to(self.model.device)
        )
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True
        )
        output_ids = self.model.generate(self.prompt_ids, gen_config).sequences
        output_text = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(output_ids[0])
        )
        return [output_text]


if __name__ == "__main__":
    engine = Engine("/data0/xiac/hf_models/Llama-3-8B-Instruct")
    llama_output = engine.execute(["What is the meaning of life?"], temperature=0.001)[
        0
    ]
    print(llama_output)
