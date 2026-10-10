import math
import struct
import inspect
import time

from .QFConfig import QFConfig
from typing import Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


# RMS归一化层 | RMS Normalization Layer
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * (
            x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)


# 预计算旋转位置编码复数 | Precompute rotary positional encoding complex numbers
def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis


# 应用旋转位置编码 | Apply rotary positional embeddings
def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        """
        调整位置编码形状以匹配输入序列长度 | Adjust the shape of pos_cis to match the sequence length of x.
        """
        seq_len = x.shape[
            1
        ]  # 获取输入序列的实际长度 | Get actual sequence length of input
        assert (
            pos_cis.shape[0] >= seq_len
        ), "位置编码长度必须大于或等于序列长度 | Positional encoding length must be >= sequence length"
        return pos_cis[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, L, 1, D/2]

    def complex_mul(x_real, x_imag, pos_cis_real, pos_cis_imag):
        x_real = x_real.to(torch.float32)
        x_imag = x_imag.to(torch.float32)
        pos_cis_real = pos_cis_real.to(torch.float32)
        pos_cis_imag = pos_cis_imag.to(torch.float32)

        real_part = x_real * pos_cis_real - x_imag * pos_cis_imag
        imag_part = x_real * pos_cis_imag + x_imag * pos_cis_real
        return real_part, imag_part

    # 提取查询和键的实部与虚部 | Extract real and imaginary parts of xq and xk
    xq_real = xq[..., 0::2]
    xq_imag = xq[..., 1::2]
    xk_real = xk[..., 0::2]
    xk_imag = xk[..., 1::2]

    # 调整位置编码形状 | Adjust the shape of pos_cis
    pos_cis = unite_shape(pos_cis, xq_real)
    pos_cis_real = pos_cis[..., 0].unsqueeze(2)  # [1, L, 1, D/2]
    pos_cis_imag = pos_cis[..., 1].unsqueeze(2)  # [1, L, 1, D/2]

    # 手动应用旋转位置编码 | Apply rotary position embeddings manually
    xq_out_real, xq_out_imag = complex_mul(xq_real, xq_imag, pos_cis_real, pos_cis_imag)
    xk_out_real, xk_out_imag = complex_mul(xk_real, xk_imag, pos_cis_real, pos_cis_imag)

    # 将实部虚部重新合并 | Combine real and imaginary parts back into a single tensor
    xq_out = torch.cat([xq_out_real, xq_out_imag], dim=-1)
    xk_out = torch.cat([xk_out_real, xk_out_imag], dim=-1)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# 键值头重复扩展 | Repeat key-value heads for multi-head attention
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# 多头注意力层 | Multi-head attention layer
class Attention(nn.Module):
    def __init__(self, args: QFConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attn
        )
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # 因果掩码 | Causal mask
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_cis: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)
        # KV缓存实现 | KV cache implementation
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=dropout_p, is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores += self.mask[:, :, :seq_len, :seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


# 前馈网络层 | Feed-forward network layer
class FeedForward(nn.Module):
    def __init__(self, config: QFConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * (
                (hidden_dim + config.multiple_of - 1) // config.multiple_of
            )
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# Transformer块 | Transformer block
class QubitFormerBlock(nn.Module):
    def __init__(self, layer_id: int, config: QFConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


# 主模型 | Main model
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = QFConfig

    def __init__(self, params: QFConfig, **kwargs):
        # 将 config 和所有额外参数（如 dtype, device_map）传给父类
        super().__init__(params, **kwargs)
        # 如果仍想保留 self.params，可以同时赋值（指向同一对象）
        self.params = params
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList(
            [QubitFormerBlock(l, params) for l in range(self.n_layers)]
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        # 预计算位置编码 | Precompute positional encodings
        pos_cis = precompute_pos_cis(
            dim=params.dim // params.n_heads, theta=params.rope_theta
        ).float()  # 确保为float32 | Ensure float32 type

        self.register_buffer("pos_cis", pos_cis, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **args,
    ):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get("start_pos", 0)
        h = self.dropout(self.tok_embeddings(input_ids))
        pos_cis = self.pos_cis[start_pos : start_pos + input_ids.size(1)]
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis, past_key_value=past_key_values[l], use_cache=use_cache
            )
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))

        # 直接构造并返回 CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            logits=logits, past_key_values=past_kvs if use_cache else None
        )

    # 推理模式生成函数 | Inference mode generation function
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 如果已有缓存，只取最后一个 token / If cache exists, take only the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 计算 start_pos（当前序列长度）/ Compute start_pos (current sequence length)
        start_pos = 0
        if past_key_values is not None and past_key_values[0] is not None:
            start_pos = past_key_values[0][0].size(1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "start_pos": start_pos,
        }

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        # 用于 beam search 时重新排序每层的缓存 / Reorder cache for each layer during beam search
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_layer = tuple(
                past_state.index_select(0, beam_idx) for past_state in layer_past
            )
            reordered_past += (reordered_layer,)
        return reordered_past
