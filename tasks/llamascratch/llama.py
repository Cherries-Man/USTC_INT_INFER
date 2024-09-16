from transformers import (
    AutoTokenizer,
    GenerationConfig,
    DynamicCache,
    EncoderDecoderCache,
    logging,
)  # , LlamaForCausalLM
from my_modeling_llama import LlamaForCausalLM
from typing import Optional
import torch
import time


class Engine:
    def __init__(self, model_path: str) -> None:
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to("cuda:6")
        print(self.model.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.prompt_ids = None

    def _generate(
        self,
        input_ids: torch.Tensor,
        generation_config: Optional[GenerationConfig] = None,
    ):

        batch_size = input_ids.shape[0]
        device = input_ids.device
        past_key_values = DynamicCache()

        cur_len = 0
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        eos_token_tensor = torch.tensor(
            self.model.generation_config.eos_token_id, device=device, dtype=torch.long
        )

        # 初始化 cache_position
        cache_position = (
            torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        )
        # 记录当前时间
        start_time = time.time()
        while cur_len < generation_config.max_new_tokens:
            # 记录prefill完成时间
            if cur_len == 0:
                prefill_finish_time = time.time()
            attention_mask = torch.ones(
                input_ids.shape[:2], dtype=torch.long, device=input_ids.device
            )
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
            )

            # forward pass to get next token
            outputs = self.model(
                input_ids=model_inputs["input_ids"],
                attention_mask=attention_mask,
                position_ids=model_inputs["position_ids"],
                past_key_values=past_key_values,
                cache_position=cache_position,
                return_dict=True,
            )

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_scores = outputs.logits[:, -1, :].clone()

            next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + eos_token_tensor * (
                1 - unfinished_sequences
            )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # update past_key_values keeping its naming used in model code
            _, cache = self.model._extract_past_from_model_output(outputs)
            past_key_values = cache
            
            cache_position = cache_position[-1:] + 1 # update cache_position, only for the last token
            

            unfinished_sequences = unfinished_sequences & ~torch.isin(
                input_ids[:, -1], eos_token_tensor
            )  # stopping_criteria(input_ids, scores)
            cur_len += 1

            if unfinished_sequences.max() == 0:
                break
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
        # 记录生成完成时间
        decode_finish_time = time.time()
        print(f"prefill time: {prefill_finish_time - start_time:.4f} 秒")
        print(f"decode time:  {decode_finish_time - prefill_finish_time:.4f} 秒")
        print("cur_len: ", cur_len)
        # Convert to legacy cache if needed
        past_key_values = past_key_values.to_legacy_cache()
        return input_ids

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
        # output_ids = self.model.generate(self.prompt_ids, gen_config).sequences
        output_ids = self._generate(self.prompt_ids, gen_config)
        output_text = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(output_ids[0])
        )
        return [output_text]

logger = logging.get_logger(__name__)

if __name__ == "__main__":
    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 100,000 memory events, via the `max_entries` field.
    # torch.cuda.memory._record_memory_history(
    #    max_entries=100000
    # )
    
    engine = Engine("/data0/xiac/hf_models/Llama-3-8B-Instruct")
    llama_output = engine.execute(["What is the meaning of life?"], temperature=0.001)[
        0
    ]
    print(llama_output)
    
    # In this sample, we save the snapshot after running 5 iterations.
    #   - Save as many snapshots as you'd like.
    #   - Snapshots will save last `max_entries` number of memory events
    #     (100,000 in this example).
    # file_prefix = "llama3_8b_instruct"
    # try:
    #    torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    # except Exception as e:
    #    logger.error(f"Failed to capture memory snapshot {e}")

    # # Stop recording memory snapshot history.
    # torch.cuda.memory._record_memory_history(enabled=None)
