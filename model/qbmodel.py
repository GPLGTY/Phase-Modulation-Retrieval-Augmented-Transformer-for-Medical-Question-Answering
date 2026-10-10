import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Any, Optional, Tuple, List, Union
from .QFConfig import QFConfig


# RMSNorm 层 / RMSNorm layer
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        inv_norm = torch.rsqrt(variance + self.eps)
        result = x * inv_norm.type_as(x)
        return self.weight * result


# 键值头重复函数 / Key-value head repeat function
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# 增强版动态 NTK 缩放 / Enhanced dynamic NTK scaling
class EnhancedDynamicNTKScaling:
    def __init__(self, max_train_len=1024, alpha=1.0, beta=0.5, gamma=0.1):
        self.max_train_len = max_train_len
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, seq_len, dim, base_theta=1e6):
        # if seq_len <= self.max_train_len:
        #     return base_theta
        ratio = seq_len / self.max_train_len
        scale = (ratio ** (dim / (dim - 2))) * (1 + self.beta * math.log1p(ratio))
        scale = scale + self.gamma * (scale - 1)
        return base_theta * (scale**self.alpha)


# NTK 位置编码预计算 / Precompute NTK position encoding
def precompute_ntk_pos_cis(dim: int, max_seq_len: int, theta: float = 1e6):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float)[: (dim // 2)] / dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


# 量子嵌入层 / Qubit embedding layer
class QubitEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        vocab_size: int,
        temperature: float = 0.9,
        phase_scale_init: float = 1.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, dim * 2)
        self.temperature = 1.1
        self.register_buffer("pi_const", torch.tensor(math.pi))

        self.phase_norm = nn.LayerNorm(dim)
        self.phase_scale = nn.Parameter(torch.tensor(phase_scale_init))

    def forward(self, x):
        x_embed = self.embed(x)
        h = self.proj(x_embed)
        h_real, h_imag = h.chunk(2, dim=-1)

        h_imag = self.phase_norm(h_imag)
        h_imag = torch.tanh(h_imag / self.temperature) * self.phase_scale

        qubit_real = h_real * torch.cos(h_imag)
        return qubit_real, h_imag


# 增强版旋转嵌入 / Enhanced rotary embedding
def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    phase_angles: torch.Tensor,
    head_dim: int,
    seq_len: int,
    dynamic_ntk: Optional[EnhancedDynamicNTKScaling] = None,
) -> torch.Tensor:
    bsz, _, n_heads, _ = x.shape
    dim = head_dim

    if dynamic_ntk is not None:
        scaled_theta = dynamic_ntk(seq_len, dim)
        freqs = 1.0 / (
            scaled_theta
            ** (
                torch.arange(0, dim, 2, device=x.device, dtype=torch.float)[: dim // 2]
                / dim
            )
        )
        t = torch.arange(seq_len, device=x.device, dtype=torch.float)
        dynamic_freqs = torch.outer(t, freqs)
        cos_cur = torch.cos(dynamic_freqs)
        sin_cur = torch.sin(dynamic_freqs)
    else:
        cos_cur = cos[:seq_len]
        sin_cur = sin[:seq_len]

    cos_cur = cos_cur.view(1, seq_len, 1, -1)
    sin_cur = sin_cur.view(1, seq_len, 1, -1)

    if phase_angles.dim() == 3:
        phase_angles = phase_angles.unsqueeze(2)

    phase_angles = phase_angles[..., : dim // 2]
    if phase_angles.size(-1) < dim // 2:
        pad_size = dim // 2 - phase_angles.size(-1)
        phase_angles = F.pad(phase_angles, (0, pad_size), "constant", 0)

    unified_cos = torch.cos(phase_angles) * cos_cur - torch.sin(phase_angles) * sin_cur
    unified_sin = torch.sin(phase_angles) * cos_cur + torch.cos(phase_angles) * sin_cur

    x_ = x.view(bsz, seq_len, n_heads, head_dim // 2, 2)
    x1, x2 = x_[..., 0], x_[..., 1]

    x1_rot = x1 * unified_cos - x2 * unified_sin
    x2_rot = x1 * unified_sin + x2 * unified_cos

    return (
        torch.stack([x1_rot, x2_rot], dim=-1)
        .view(bsz, seq_len, n_heads, head_dim)
        .type_as(x)
    )


# 增强版注意力机制 / Enhanced attention mechanism
class QubitAttention(nn.Module):
    def __init__(self, args: QFConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.q_proj = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

        phase_dim = args.dim // 2
        self.phase_proj = nn.Sequential(
            nn.Linear(args.dim, phase_dim),
            nn.GELU(),
            nn.LayerNorm(phase_dim),
            nn.Linear(phase_dim, self.n_heads * (self.head_dim // 2)),
            nn.Tanh(),
        )

        self.dropout = args.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention") and args.flash_attn
        self.max_seq = args.max_seq_len

        self.dynamic_ntk = EnhancedDynamicNTKScaling(
            max_train_len=args.max_seq_len, alpha=args.ntk_alpha, beta=0.5
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        h_imag: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        seq_start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        bsz, seq_len, _ = x.shape

        phase = self.phase_proj(h_imag)
        phase = phase.view(bsz, seq_len, self.n_heads, self.head_dim // 2)
        phase_k = phase[:, :, : self.n_kv_heads, :]

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q_rot = apply_rotary_emb(
            q, cos, sin, phase, self.head_dim, seq_len, self.dynamic_ntk
        )
        k_rot = apply_rotary_emb(
            k, cos, sin, phase_k, self.head_dim, seq_len, self.dynamic_ntk
        )

        k_rot = repeat_kv(k_rot, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_rot = torch.cat([past_k, k_rot], dim=1)
            v = torch.cat([past_v, v], dim=1)

        kv_cache = (k_rot, v) if use_cache else None

        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)
        v = v.transpose(1, 2)

        use_flash = self.flash and (past_key_value is None or k_rot.size(2) <= 2048)

        if use_flash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                attn_output = F.scaled_dot_product_attention(
                    q_rot,
                    k_rot,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=(past_key_value is None),
                )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, seq_len, -1)
        else:
            total_len = k_rot.size(2)
            attn_weights = torch.matmul(q_rot, k_rot.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )

            if past_key_value is None:
                mask = torch.full(
                    (1, 1, seq_len, seq_len), float("-inf"), device=x.device
                )
                mask = torch.triu(mask, diagonal=1)
            else:
                hist_len = total_len - seq_len
                mask_hist = torch.zeros((1, 1, seq_len, hist_len), device=x.device)
                mask_current = torch.triu(
                    torch.full(
                        (1, 1, seq_len, seq_len), float("-inf"), device=x.device
                    ),
                    diagonal=1,
                )
                mask = torch.cat([mask_hist, mask_current], dim=-1)

            attn_weights += mask
            attn_weights = F.softmax(attn_weights.float(), dim=-1).to(q_rot.dtype)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, seq_len, -1)

        return self.o_proj(attn_output), kv_cache


# 轻量级量子门控 / Lightweight qubit gating
class LiteQubitGate(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, reduction_ratio: int = 8):
        super().__init__()
        self.phase_proj = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.LayerNorm(dim // reduction_ratio),
            nn.Linear(dim // reduction_ratio, 2 * hidden_dim, bias=False),
        )
        nn.init.xavier_uniform_(self.phase_proj[0].weight)
        nn.init.xavier_uniform_(self.phase_proj[-1].weight)

    def forward(self, h_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = self.phase_proj(h_imag)
        gate, scale = gates.chunk(2, dim=-1)
        scale_result = torch.tanh(scale)
        gate_result = torch.sigmoid(gate)
        return gate_result, scale_result


# 增强版前馈网络 / Enhanced feed-forward network
class QubitFeedForward(nn.Module):
    def __init__(self, config: QFConfig):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.qubit_gate = LiteQubitGate(
            dim=config.dim, hidden_dim=hidden_dim, reduction_ratio=8
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, h_imag: torch.Tensor):
        gate, scale = self.qubit_gate(h_imag)
        w1_x = self.w1(x)
        w3_x = self.w3(x)
        h = F.silu(w1_x) * gate + torch.tanh(scale * w3_x)
        return self.dropout(self.w2(h))


# 增强版 Transformer 块 / Enhanced Transformer block
class QubitBlock(nn.Module):
    def __init__(self, layer_id: int, config: QFConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = QubitAttention(config)
        self.ffn = QubitFeedForward(config)

        self.phase_update = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.LayerNorm(config.dim),
        )

        self.phase_gate = nn.Parameter(torch.tensor(0.5))
        self.phase_decay = nn.Parameter(torch.tensor(0.95))
        self.extrapolation_stabilizer = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x,
        cos,
        sin,
        h_imag,
        past_key_value=None,
        use_cache=False,
        phase_scale=None,
        phase_decay=None,
        seq_start_pos=0,
    ):
        # 使用张量 clamp，支持标量张量 / Use tensor clamp, support scalar tensors
        if phase_scale is not None:
            phase_gate_val = self.phase_gate.clamp(0, phase_scale)
        else:
            phase_gate_val = self.phase_gate

        if phase_decay is not None:
            phase_decay_val = self.phase_decay.clamp(phase_decay, 1.0)
        else:
            phase_decay_val = self.phase_decay

        h_imag_norm = F.normalize(h_imag, p=2, dim=-1)
        h_imag_updated = torch.tanh(self.phase_update(h_imag_norm))
        h_imag_updated = h_imag_updated * phase_decay_val

        extrapolation_stab = self.extrapolation_stabilizer * torch.norm(
            h_imag_updated - h_imag, p=2, dim=-1, keepdim=True
        )
        h_imag_updated = h_imag_updated + extrapolation_stab

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            attn_out, past_kv = self.attn(
                self.attention_norm(x),
                cos,
                sin,
                h_imag_updated,
                past_key_value=past_key_value,
                use_cache=use_cache,
                seq_start_pos=seq_start_pos,
            )
        x = x + attn_out

        norm_x = self.ffn_norm(x)
        ffn_out = self.ffn(norm_x, h_imag_updated)
        x = x + ffn_out

        h_imag = phase_gate_val * h_imag_updated + (1 - phase_gate_val) * h_imag

        return x, h_imag, past_kv


# 增强版 QubitFormer 模型（张量化版本，消除 graph break）/ Enhanced QubitFormer model (tensorized version, eliminates graph break)
class QubitFormer(PreTrainedModel, GenerationMixin):
    config_class = QFConfig

    def __init__(self, params: QFConfig, **kwargs):
        super().__init__(params, **kwargs)
        self.params = params
        self.tok_embeddings = QubitEmbed(params.dim, params.vocab_size)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList(
            [QubitBlock(l, params) for l in range(params.n_layers)]
        )
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.embed.weight = self.output.weight

        self.dynamic_ntk = EnhancedDynamicNTKScaling(
            max_train_len=params.max_seq_len, alpha=params.ntk_alpha, beta=0.5
        )

        self.register_buffer("global_step", torch.tensor(0), persistent=True)
        self.extrapolation_prob = 0.1

        # post_init 会设置 all_tied_weights_keys 并走 HF 标准 init_weights/_init_weights；
        # 新版 transformers 在 from_pretrained 收尾会依赖上述属性。
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, QubitBlock):
            nn.init.constant_(module.phase_gate, 0.1)
            nn.init.constant_(module.extrapolation_stabilizer, 0.01)

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        """告知 transformers 勿注入 HF DynamicCache；本模型使用每层 tuple KV。"""
        return False

    def forward(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = kwargs.get("start_pos", 0)

        h_real, h_imag = self.tok_embeddings(input_ids)
        h = self.dropout(h_real)

        seq_len = input_ids.size(1)
        total_seq_len = seq_len
        if past_key_values[0] is not None:
            total_seq_len += (
                past_key_values[0][0].size(1)
                if past_key_values[0][0] is not None
                else 0
            )

        scaled_theta = self.dynamic_ntk(
            total_seq_len, self.params.dim // self.params.n_heads
        )
        max_len_needed = total_seq_len + 128

        cos, sin = precompute_ntk_pos_cis(
            self.params.dim // self.params.n_heads,
            max_len_needed,
            scaled_theta,
        )
        cos = cos.to(h.device)
        sin = sin.to(h.device)

        # 动态相位计算 / Dynamic phase computation
        if self.training:
            current_step = self.global_step.float()
            # 计算标量张量 phase_scale 和 phase_decay / Compute scalar tensors phase_scale and phase_decay
            phase_scale = torch.max(
                torch.tensor(0.3, device=current_step.device),
                0.5 * (1 - torch.log1p(current_step / 10000) / 10),
            )
            phase_decay = torch.max(
                torch.tensor(0.8, device=current_step.device),
                0.95 - current_step / 50000,
            )
            self.global_step += 1
        else:
            # 推理时使用固定值，同样生成为标量张量 / Use fixed values during inference, also generated as scalar tensors
            phase_scale = torch.tensor(0.99, device=h.device)
            phase_decay = torch.tensor(1.0, device=h.device)

        present_kvs = []
        for i, (layer, past) in enumerate(zip(self.layers, past_key_values)):
            h, h_imag, kv = layer(
                h,
                cos,
                sin,
                h_imag,
                past_key_value=past,
                use_cache=use_cache,
                phase_scale=phase_scale,
                phase_decay=phase_decay,
                seq_start_pos=start_pos,
            )
            present_kvs.append(kv)

        norm_h = self.norm(h)
        logits = self.output(norm_h)

        return CausalLMOutputWithPast(
            logits=logits, past_key_values=present_kvs if use_cache else None
        )

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
