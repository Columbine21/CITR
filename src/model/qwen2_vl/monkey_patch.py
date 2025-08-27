import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.utils import (
    is_flash_attn_2_available, logging
)
if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func

from .modeling_qwen2_vl import apply_multimodal_rotary_pos_emb, repeat_kv, BaseModelOutputWithPast, Qwen2VLModel, Qwen2VLFlashAttention2
from .varlen_cache import DynamicCacheSplitHead

logger = logging.get_logger(__name__)

def adaptive_qwen2vl_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_pre_act_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    plain_attn: bool = False
)-> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_pre_act_attentions = output_pre_act_attentions if output_pre_act_attentions is not None else False
    if not plain_attn:
        output_attentions = False
        output_pre_act_attentions = False
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False
    
    ## NOTE: adakv for prefill and decode
    num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
    past_key_values = DynamicCacheSplitHead.from_legacy_cache(past_key_values, num_key_value_groups=num_key_value_groups)
    if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions or output_pre_act_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_pre_act_self_attns = () if output_pre_act_attentions else None
    all_casual_importance_score = ()

    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                plain_attn=plain_attn
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_pre_act_attentions=output_pre_act_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                plain_attn=plain_attn
            )
        hidden_states = layer_outputs[0]
        cis_index, cache_index = 1, 1

        if output_attentions:
            all_self_attns += (layer_outputs[1],)
            cis_index += 1
            cache_index += 1

        if output_pre_act_attentions:
            all_pre_act_self_attns += (layer_outputs[2 if output_attentions else 1],)
            cis_index += 1
            cache_index += 1 

        if use_cache:
            next_decoder_cache = layer_outputs[cache_index]
            cis_index += 1

        all_casual_importance_score += (layer_outputs[cis_index],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        pre_act_attentions=all_pre_act_self_attns,
        casual_importance_score=all_casual_importance_score,
    )

def adaptive_qwen2vl_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
):
    bsz, q_len, _ = hidden_states.size()
    assert bsz == 1, "Only support batch size 1 for now"

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # NOTE ========== compute the CIS ==========
    causal_importance_score = self.retaining_head2(self.act(self.retaining_head1(torch.cat((query_states, key_states, value_states), dim=-1))))

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    # repeat k/v heads if n_kv_heads < n_heads
    # key_states = repeat_kv(key_states, self.num_key_value_groups)
    # value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, "num_key_value_groups": self.num_key_value_groups}  # Specific to RoPE models
    ## NOTE: AdaKV将自动处理GQA的情况
    key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx, cache_kwargs=cache_kwargs)

    bsz, num_head, seqlen, headdim = query_states.shape
    head_lens = torch.full((num_head, ), seqlen, device=query_states.device, dtype=torch.int32)
    max_seqlen_q = seqlen
    cu_seqlens_q = torch.cumsum(head_lens, dim=0) - seqlen
    cu_seqlens_q = torch.cat((cu_seqlens_q, torch.sum(head_lens, dim=0, keepdim=True)), dim=0).to(device=query_states.device, dtype=torch.int32)
    query_states = query_states.reshape(-1, 1, headdim)

    cu_seqlens_k = past_key_value.get_cu_seqlens_k(self.layer_idx)
    max_seqlen_k = past_key_value.get_max_seqlen_k(self.layer_idx)
    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout_rate,
        causal=True
    )
    attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value, None, causal_importance_score

origin_qwen2vl_model_forward = Qwen2VLModel.forward
originqwen2vl_flash_attn2_forward = Qwen2VLFlashAttention2.forward

def patch_qwen2vl():
    Qwen2VLModel.forward = adaptive_qwen2vl_model_forward
    Qwen2VLFlashAttention2.forward = adaptive_qwen2vl_flash_attn2_forward

def close_patch_qwen2vl():
    Qwen2VLModel.forward = origin_qwen2vl_model_forward
    Qwen2VLFlashAttention2.forward = originqwen2vl_flash_attn2_forward