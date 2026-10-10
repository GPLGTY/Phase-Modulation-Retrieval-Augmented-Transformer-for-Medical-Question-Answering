from transformers import PretrainedConfig
from typing import List


class QFConfig(PretrainedConfig):
    model_type = "QubitFormer"

    def __init__(
        self,
        dim: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        vocab_size: int = 32000,
        hidden_dim: int = None,
        multiple_of: int = 64,
        norm_eps: float = 1e-5,
        max_seq_len: int = 1024,
        n_rot_groups: int = 8,
        rope_theta: int = 1e6,
        dropout: float = 0.1,
        flash_attn: bool = True,
        quantum_depth: int = 4,
        entanglement_strategy: str = "linear",
        use_parallel_ffn: bool = True,
        use_shared_kv_proj: bool = True,
        phase_scale_init: float = 1.0,
        ntk_alpha: float = 0.8,
        **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.n_rot_groups = n_rot_groups
        self.quantum_depth = quantum_depth
        self.entanglement_strategy = entanglement_strategy
        self.use_parallel_ffn = use_parallel_ffn
        self.use_shared_kv_proj = use_shared_kv_proj
        self.phase_scale_init = phase_scale_init
        self.ntk_alpha = ntk_alpha

        super().__init__(**kwargs)

    # --- Hugging Face transformers 生成接口兼容（DynamicCache 等依赖下列字段名）---
    @property
    def num_hidden_layers(self):
        return self.n_layers

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_key_value_heads(self):
        return self.n_kv_heads

    @property
    def hidden_size(self):
        return self.dim
