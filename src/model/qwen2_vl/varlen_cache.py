import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.utils import (
    is_flash_attn_2_available, logging
)
if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func

# class DynamicCacheSplitHeadFlatten(Cache):
#     """
#     Flattened version of DynamicCacheSplitHead
#     """
#     def __init__(self) ->None:
#         # Token wise List[]  Head wise KV List[torch.Tensor]
#         super().__init__()
#         self.key_cache: List[List[torch.Tensor]] = []
#         self.value_cache: List[List[torch.Tensor]] = []
#         self._seen_tokens = 0
#         ## NOTE: example
#         """
#         >>> the shape of a key cache for a layer is (total_tokens, headdim)
#         >>> total_tokens is the sum of tokens of the all heads
#         >>> Assume that a key cache is like this (2, 4, 3), each is the seqlen of a head,
#         >>> the seq_len_per_head will be torch.Tensor(0, 2, 6, 9)
#         """
#         self.seq_len_per_head = List[torch.Tensor] = []
#         self.max_seqlen_key = 0

#     def __len__(self):
#         return len(self.key_cache)

#     def __iter__(self):
#         for layer_idx in range(len(self)):
#             yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

#     def __getitem__(self, layer_idx: int) -> Tuple[Tuple[torch.Tensor],Tuple[torch.Tensor]]:
#         if layer_idx < len(self):
#             return (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))
#         else:
#             raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

#     def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
#         # NOTE: k, v = [head_num](bs, 1, seqlen, dim)
#         # each layer is a flatten layout like:
#         # [head_0_len + head_1_len + ..., dim]
#         if len(self.key_cache) <= layer_idx:
#             self.key_cache.append(key_states)
#             self.value_cache.append(value_states)
#         else:
#             assert self.key_cache[layer_idx].dim() == 2
#             bs, head, seqlen, dim = key_states.shape
#             # assert bs == 1 and seqlen == 1
#             # NOTE: phase 2. we got [bs, head, seqlen, dim] as k, v input
#             # head_lens = cache_kwargs["head_lens"]
#             # cu_klen = cache_kwargs["cu_klen"]

#             seq_len_per_head = self.seq_len_per_head[layer_idx]
#             klen_sum = torch.sum(seq_len_per_head[layer_idx], keepdim=True)
#             cu_klen = torch.cat((seq_len_per_head[layer_idx], klen_sum), dim=0)

#             # TODO: wrap as a python interface
#             from tiny_api_cuda import update_flatten_view
#             new_key_cache = update_flatten_view(self.key_cache[layer_idx].view(-1,dim), key_states.view(-1, dim), seq_len_per_head, cu_klen)
#             new_value_cache = update_flatten_view(self.value_cache[layer_idx].view(-1,dim), value_states.view(-1, dim), seq_len_per_head, cu_klen)


#             self.key_cache[layer_idx] = new_key_cache
#             self.value_cache[layer_idx] = new_value_cache


#         return self.key_cache[layer_idx], self.value_cache[layer_idx]

#     def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
#         if len(self.key_cache) <= layer_idx:
#             return 0

#         # TODO: return 1 to means has content for now
#         return 1
#         # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

#     def get_max_length(self) -> Optional[int]:
#         return None

#     def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
#         """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
#         legacy_cache = ()
#         for layer_idx in range(len(self)):
#             legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
#         return legacy_cache

#     @classmethod
#     def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCacheEachHead":
#         """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
#         cache = cls()
#         if past_key_values is not None:
#             for layer_idx in range(len(past_key_values)):
#                 key_states, value_states = past_key_values[layer_idx]
#                 cache.update(key_states, value_states, layer_idx)
#         return cache
    


class DynamicCacheSplitHead(Cache):
    """
    demo for illustrate the splited cache update
    This class is slower than DynamicCacheSplitHeadFlatten, due to the frequent tensor copy
    """
    def __init__(self, num_key_value_groups: int = 1) ->None:
        super().__init__()
        #  [Layer[Head]]
        self.key_cache: List[List[torch.Tensor]] = []
        self.value_cache: List[List[torch.Tensor]] = []
        self._seen_tokens = 0
        self.cu_seqlens_k_list: List[torch.Tensor] = []
        self.max_seqlen_k_list: List[int] = []
        self.num_key_value_groups = num_key_value_groups

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),tuple(self.value_cache[layer_idx]))

    def __getitem__(self, indices: int | Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(indices, tuple):
            layer_idx, head_idx = indices
            if layer_idx < len(self):
                return self.key_cache[layer_idx][head_idx], self.value_cache[layer_idx][head_idx]
            else:
                raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        elif isinstance(indices, int):
            layer_idx = indices
            if layer_idx < len(self):
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
            else:
                raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        else:
            raise RuntimeError(f"Unsupported type of indices {type(indices)}")

    def update(
        self,
        key_states: torch.Tensor | List[torch.Tensor],
        value_states: torch.Tensor | List[torch.Tensor],
        layer_idx: int,
        cache_kwargs=None
    ) -> Tuple[Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]]:
        """
        Args:
            key_states: torch.Tensor, shape = (batch, num_head, seq_len, headdim) | List[torch.Tensor, shape = (batch, seqlen, headdim)]
            value_states: torch.Tensor, shape = (batch, num_head, seq_len, headdim) | List[torch.Tensor, shape = (batch, seqlen, headdim)]
        """
        if isinstance(key_states, torch.Tensor):
            if key_states.dim() == 4:
                batch, num_head, seqlen, headdim = key_states.shape
                assert batch == 1, "Only support batch size 1 for now"
                device = key_states.device
                dtype = key_states.dtype
                # transfer key_states and value states to List
                key_states = list(key_states.unbind(dim=1))
                value_states = list(value_states.unbind(dim=1))
            else:
                raise RuntimeError(f"Invalid shape of key_states, shape is {key_states.shape}")
        elif isinstance(key_states, list):
            assert isinstance(key_states[0], torch.Tensor), f"Invalid type of key_states, type is {type(key_states[0])}"
            device = key_states[0].device
            dtype = key_states[0].dtype
            batch, _, headdim = key_states[0].shape
        else:
            raise RuntimeError(f"Invalid type of key_states, type is {type(key_states)}")
        if layer_idx == 0:
            self._seen_tokens += max(map(lambda states: states.shape[-2], key_states))

        # 记录flash_attn_varlen_func需要的数据
        num_key_value_heads = len(key_states)
        # 记录每个head的长度情况，由于GQA的存在，这里要考虑到groups
        seq_len_per_head = torch.tensor([key_states[i].shape[-2] for i in range(num_key_value_heads)], device=device, dtype=torch.int32)
        seq_len_per_head = torch.repeat_interleave(seq_len_per_head, repeats=self.num_key_value_groups, dim=0).to(device=device, dtype=torch.int32)
        if len(self.key_cache)<=layer_idx:
            cu_seq_len = torch.cumsum(seq_len_per_head, dim=0) - seq_len_per_head[0]
            cu_seq_len = torch.cat((cu_seq_len, torch.sum(seq_len_per_head, dim=0, keepdim=True)), dim=0).to(device=device, dtype=torch.int32)
            self.cu_seqlens_k_list.append(cu_seq_len)
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.max_seqlen_k_list.append(seq_len_per_head.max().item())
        else:
            for head_idx in range(0, num_key_value_heads):
                self.key_cache[layer_idx][head_idx] = torch.cat([self.key_cache[layer_idx][head_idx],key_states[head_idx]], dim=-2)
                self.value_cache[layer_idx][head_idx] = torch.cat([self.value_cache[layer_idx][head_idx],value_states[head_idx]], dim=-2)
            cu_seq_len = torch.cumsum(seq_len_per_head, dim=0) - seq_len_per_head[0]
            cu_seq_len = torch.cat((cu_seq_len, torch.sum(seq_len_per_head, dim=0, keepdim=True)), dim=0).to(device=device)
            self.cu_seqlens_k_list[layer_idx] += cu_seq_len
            self.max_seqlen_k_list[layer_idx] += seq_len_per_head.max().item()
        # 为了满足GQA，update时将存储的cache复制到对于组数，并展开
        cur_layer_key_cache = [ts.view(-1, 1, headdim).repeat((self.num_key_value_groups, 1, 1)).view(-1, 1, headdim) for ts in self.key_cache[layer_idx]]
        cur_layer_key_cache = torch.cat(cur_layer_key_cache, dim=0).to(device=device, dtype=dtype)

        cur_layer_value_cache = [ts.view(-1, 1, headdim).repeat((self.num_key_value_groups, 1, 1)).view(-1, 1, headdim) for ts in self.value_cache[layer_idx]]
        cur_layer_value_cache = torch.cat(cur_layer_value_cache, dim=0).to(device=device, dtype=dtype)
        return cur_layer_key_cache, cur_layer_value_cache

    def get_cu_seqlens_k(self, layer_idx: int):
        return self.cu_seqlens_k_list[layer_idx]

    def get_max_seqlen_k(self, layer_idx: int):
        return self.max_seqlen_k_list[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None


    # Tuple[Tuple[Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]],...]
    ## NOTE: no use, I'm not sure whethe it causes a bug
    def to_legacy_cache(self)-> Tuple[Tuple[Tuple[torch.Tensor,...],Tuple[torch.Tensor,...]],...]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((tuple(self.key_cache[layer_idx]), tuple(self.value_cache[layer_idx])),)
        return legacy_cache
    @classmethod
    def from_legacy_cache(cls, past_key_values=None, num_key_value_groups: int = 1):
        cache = cls(num_key_value_groups=num_key_value_groups)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states,value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

if __name__ == "__main__":
    device = "cuda:0"
    dtype = torch.bfloat16
    batch, num_head, headdim = 1, 4, 10
    query_len = 10
    seqlens = [2, 2, 4, 8]
    # 定义正态分布的均值和标准差
    mean = torch.tensor(50.0)  # 均值设置为 50，使随机数更可能集中在 0 - 100 的中间
    std = torch.tensor(20.0)   # 标准差设置为 20，可根据需要调整

    def normal_random(size):
        # 生成-100-100的服从正态分布的随机数
        data = torch.normal(mean=mean, std=std, size=size, device=device, dtype=dtype)
        clamp_data = torch.clamp(data, min=-100, max=100).to(device=device, dtype=dtype)
        return clamp_data
    key_states = [normal_random((batch, seqlen, headdim)) for seqlen in seqlens]
    value_states = [normal_random((batch, seqlen, headdim)) for seqlen in seqlens]
    query_states = normal_random((batch, query_len, headdim))

    past_key_values = [(key_states, value_states)]

    qlen = torch.full((num_head, ), query_len, device=device, dtype=torch.int32)
    max_seqlen_q = qlen.max().item()
    cu_seqlens_q = torch.cumsum(qlen, dim=0) - query_len
    cu_seqlens_q = torch.cat((cu_seqlens_q, torch.sum(qlen, dim=0, keepdim=True)), dim=0).to(dtype=torch.int32)
    query_states = query_states.view(-1, 1, headdim)
    # key_states = [state.view(-1, 1, headdim) for state in key_states]
    # key_states = torch.cat(key_states, dim=0)

    # value_states = [state.view(-1, 1, headdim) for state in value_states]
    # value_states = torch.cat(value_states, dim=0)

    # seqlens_tensor = torch.tensor(seqlens, device=device)
    # max_seqlen_k = seqlens_tensor.max().item()
    # cu_seqlens_k = torch.cumsum(seqlens_tensor, dim=0) - seqlens[0]
    # cu_seqlens_k = torch.cat((cu_seqlens_k, torch.sum(seqlens_tensor, dim=0, keepdim=True)), dim=0).to(device=device, dtype=torch.int32)

    # attn_score = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)

    past_key_values = DynamicCacheSplitHead.from_legacy_cache(past_key_values)
    key_states, value_states = past_key_values.update(key_states, value_states, layer_idx=0)
    cu_seqlens_k = past_key_values.get_cu_seqlens_k(layer_idx=0)
    max_seqlen_k = past_key_values.get_max_seqlen_k(layer_idx=0)
    attn_score = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True)
    print()
