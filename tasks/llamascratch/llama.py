from transformers import (
    AutoTokenizer,
    GenerationConfig,
    DynamicCache,
    Cache,
    logging,
    LlamaPreTrainedModel,
    LlamaConfig,
)  # , LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaMLP,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPast,
)

# from my_modeling_llama import (
#     LlamaAttention,
# )
from typing import Optional, Union, Tuple
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import time, math


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    # if attention_mask is not None and attention_mask.dim() == 4:
    #     # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
    #     causal_mask = attention_mask
    # else:
    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(
        -1, 1
    )
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = (
            causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        )
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[
            :, :, :, :mask_length
        ].masked_fill(padding_mask, min_dtype)

    return causal_mask


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MyLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        print(
            f"self.q_proj tensor weight memory usage: {(self.q_proj.weight.element_size() * self.q_proj.weight.numel()) / (1024 ** 2):.2f} MB"
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        print(
            f"self.k_proj tensor weight memory usage: {(self.k_proj.weight.element_size() * self.k_proj.weight.numel()) / (1024 ** 2):.2f} MB"
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        print(
            f"self.v_proj tensor weight memory usage: {(self.v_proj.weight.element_size() * self.v_proj.weight.numel()) / (1024 ** 2):.2f} MB"
        )
        self.o_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )
        print(
            f"self.o_proj tensor weight memory usage: {(self.o_proj.weight.element_size() * self.o_proj.weight.numel()) / (1024 ** 2):.2f} MB"
        )
        # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        print(f"self.rotary_emb tensor weight memory usage: 暂时不考虑")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class MyLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        print(f"Initializing LlamaDecoderLayer {layer_idx}")
        self.hidden_size = config.hidden_size

        self.self_attn = MyLlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        outputs += (present_key_value,)

        return outputs


class MyLlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        print(
            f"Embedding layer weight memory usage: {(self.embed_tokens.weight.element_size() * self.embed_tokens.weight.numel()) / (1024 ** 2):.2f} MB"
        )
        self.layers = nn.ModuleList(
            [
                MyLlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        print(
            f"self.norm tensor weight memory usage: {(self.norm.weight.element_size() * self.norm.weight.numel()) / (1024 ** 2):.2f} MB"
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        print(f"self.norm tensor weight memory usage: 暂时不考虑")
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Cache = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        inputs_embeds = self.embed_tokens(input_ids)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        next_decoder_cache = None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]
            next_decoder_cache = layer_outputs[1]

        hidden_states = self.norm(hidden_states)
        next_cache = next_decoder_cache  # if use_cache else None
        return (
            hidden_states,
            next_cache,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
    ):
        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask


class MyLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MyLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        print(
            f"self.lm_head tensor weight memory usage: {(self.lm_head.weight.element_size() * self.lm_head.weight.numel()) / (1024 ** 2):.2f} MB"
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Cache = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        # 输入到LlamaModel中
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        # hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()  # 将输出重新映射到vocab_size维度
        return (logits, past_key_values)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                input_ids.shape[1] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        return (input_ids.clone(memory_format=torch.contiguous_format), position_ids)


class Engine:
    def __init__(self, model_path: str) -> None:
        self.model = MyLlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to("cuda:5")
        print(self.model.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
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
            model_input_ids, position_ids = self.model.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
            )

            # forward pass to get next token
            logits, past_key_values = self.model(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_scores = logits[:, -1, :].clone()

            next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + eos_token_tensor * (
                1 - unfinished_sequences
            )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # update past_key_values keeping its naming used in model code
            # _, past_key_values = self.model._extract_past_from_model_output(outputs)

            cache_position = (
                cache_position[-1:] + 1
            )  # update cache_position, only for the last token

            unfinished_sequences = unfinished_sequences & ~torch.isin(
                input_ids[:, -1], eos_token_tensor
            )  # stopping_criteria(input_ids, scores)
            cur_len += 1

            if unfinished_sequences.max() == 0:
                break
            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # del outputs
            del logits
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
            self.tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]
            # .view(1, -1)
            .to(self.model.device)
        )
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True
        )
        # output_ids = self.model.generate(self.prompt_ids, gen_config).sequences
        output_ids = self._generate(self.prompt_ids, gen_config)

        # output_text = self.tokenizer.convert_tokens_to_string(
        #     self.tokenizer.convert_ids_to_tokens(output_ids[0])
        # )
        # return [output_text]

        output_text = []
        for output_id in output_ids:
            output_text.append(
                self.tokenizer.decode(output_id, skip_special_tokens=True)
            )
        return output_text


logger = logging.get_logger(__name__)

if __name__ == "__main__":
    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 100,000 memory events, via the `max_entries` field.
    # torch.cuda.memory._record_memory_history(
    #    max_entries=100000
    # )

    engine = Engine("/data0/xiac/hf_models/Llama-3-8B-Instruct")
    llama_output = engine.execute(
        ["What is the meaning of life?", "What is the meaning of life?"], temperature=0.001
    )
    # print(llama_output)
    for output in llama_output:
        print(output)

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
