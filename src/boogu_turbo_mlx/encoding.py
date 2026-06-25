from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import (
    component_config_path,
    flatten_parameter_shapes as _flatten_parameter_shapes,
    load_indexed_component_weights,
    processor_path,
    read_artifact,
    read_json,
)
from .constants import DEFAULT_MAX_SEQUENCE_LENGTH, OFFICIAL_TEXT_WEIGHT_PREFIX
from .errors import BooguTurboMlxError
from .quantization import (
    apply_quantization_to_model,
    quantization_settings_for_component,
    quantized_linear_paths_for_component,
)
from .reuse_audit import build_t2i_chat_messages

try:  # Keep package importable without the optional MLX runtime.
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - exercised by non-MLX environments.
    mx = None
    nn = None

_MlxModuleBase = object if nn is None else nn.Module


@dataclass(frozen=True)
class InstructionEncoding:
    hidden_states: Any
    attention_mask: Any
    input_ids: Any


@dataclass(frozen=True)
class Qwen3VLTextConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_type: str = "default"
    mrope_interleaved: bool = True
    mrope_section: tuple[int, int, int] = (24, 20, 20)
    pad_token_id: int | None = None

    @classmethod
    def from_mllm_config(cls, payload: Mapping[str, Any]) -> "Qwen3VLTextConfig":
        if payload.get("model_type") != "qwen3_vl":
            raise ValueError("Instruction encoder supports only model_type='qwen3_vl'")
        text_config = payload.get("text_config")
        if not isinstance(text_config, Mapping):
            raise ValueError("Qwen3-VL config must contain a text_config mapping")
        return cls.from_text_config(text_config)

    @classmethod
    def from_text_config(cls, payload: Mapping[str, Any]) -> "Qwen3VLTextConfig":
        rope_payload = payload.get("rope_scaling") or payload.get("rope_parameters") or {}
        if not isinstance(rope_payload, Mapping):
            raise ValueError("Qwen3-VL rope_scaling must be a mapping")

        config = cls(
            vocab_size=int(payload.get("vocab_size", cls.vocab_size)),
            hidden_size=int(payload.get("hidden_size", cls.hidden_size)),
            intermediate_size=int(
                payload.get("intermediate_size", cls.intermediate_size)
            ),
            num_hidden_layers=int(
                payload.get("num_hidden_layers", cls.num_hidden_layers)
            ),
            num_attention_heads=int(
                payload.get("num_attention_heads", cls.num_attention_heads)
            ),
            num_key_value_heads=int(
                payload.get("num_key_value_heads", cls.num_key_value_heads)
            ),
            head_dim=int(payload.get("head_dim", cls.head_dim)),
            hidden_act=str(payload.get("hidden_act", cls.hidden_act)),
            rms_norm_eps=float(payload.get("rms_norm_eps", cls.rms_norm_eps)),
            rope_theta=float(payload.get("rope_theta", cls.rope_theta)),
            max_position_embeddings=int(
                payload.get("max_position_embeddings", cls.max_position_embeddings)
            ),
            attention_bias=bool(payload.get("attention_bias", cls.attention_bias)),
            attention_dropout=float(
                payload.get("attention_dropout", cls.attention_dropout)
            ),
            rope_type=str(rope_payload.get("rope_type", cls.rope_type)),
            mrope_interleaved=bool(
                rope_payload.get("mrope_interleaved", cls.mrope_interleaved)
            ),
            mrope_section=_tuple_ints3(
                rope_payload.get("mrope_section", cls.mrope_section),
                "mrope_section",
            ),
            pad_token_id=(
                None
                if payload.get("pad_token_id") is None
                else int(payload["pad_token_id"])
            ),
        )
        if payload.get("model_type") not in {None, "qwen3_vl_text"}:
            raise ValueError("Instruction encoder supports only qwen3_vl_text")
        config.validate()
        return config

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def to_mllm_config(self) -> dict[str, Any]:
        return {
            "model_type": "qwen3_vl",
            "text_config": {
                "model_type": "qwen3_vl_text",
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "hidden_act": self.hidden_act,
                "rms_norm_eps": self.rms_norm_eps,
                "rope_theta": self.rope_theta,
                "max_position_embeddings": self.max_position_embeddings,
                "attention_bias": self.attention_bias,
                "attention_dropout": self.attention_dropout,
                "pad_token_id": self.pad_token_id,
                "rope_scaling": {
                    "rope_type": self.rope_type,
                    "mrope_interleaved": self.mrope_interleaved,
                    "mrope_section": list(self.mrope_section),
                },
            },
        }

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError("head_dim must be a positive even integer")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.hidden_act != "silu":
            raise ValueError("Instruction encoder supports only hidden_act='silu'")
        if self.attention_bias:
            raise ValueError("Instruction encoder supports only attention_bias=false")
        if self.attention_dropout != 0.0:
            raise ValueError("Instruction encoder supports only attention_dropout=0")
        if self.rope_type != "default":
            raise ValueError("Instruction encoder supports only default RoPE")
        if not self.mrope_interleaved:
            raise ValueError("Instruction encoder requires interleaved M-RoPE")
        if any(section <= 0 for section in self.mrope_section):
            raise ValueError("mrope_section entries must be positive")
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError("sum(mrope_section) must equal head_dim // 2")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")


class Qwen3VLTextRMSNorm(_MlxModuleBase):
    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.weight = mx.ones((hidden_size,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, hidden_states: Any) -> Any:
        input_dtype = hidden_states.dtype
        hidden_float = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_float), axis=-1, keepdims=True)
        hidden_float = hidden_float * mx.rsqrt(variance + self.eps)
        return hidden_float.astype(input_dtype) * self.weight.astype(input_dtype)


class Qwen3VLTextMLP(_MlxModuleBase):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
        _require_mlx()
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, hidden_states: Any) -> Any:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        return self.down_proj(_silu(gate.astype(mx.float32)).astype(gate.dtype) * up)


class Qwen3VLTextAttention(_MlxModuleBase):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
        _require_mlx()
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.kv_dim, bias=False)
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = Qwen3VLTextRMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
        )
        self.k_norm = Qwen3VLTextRMSNorm(
            config.head_dim,
            eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any,
    ) -> Any:
        batch_size = hidden_states.shape[0]
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.reshape(batch_size, -1, self.heads, self.head_dim)
        key = key.reshape(batch_size, -1, self.kv_heads, self.head_dim)
        value = value.reshape(batch_size, -1, self.kv_heads, self.head_dim)

        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = _apply_rotary_pos_emb(query, key, position_embeddings)

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        hidden_states = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            mask=attention_mask,
        )
        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, self.heads * self.head_dim
        )
        return self.o_proj(hidden_states)


class Qwen3VLTextDecoderLayer(_MlxModuleBase):
    def __init__(self, config: Qwen3VLTextConfig) -> None:
        _require_mlx()
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        attention_mask: Any,
    ) -> Any:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3VLInstructionEncoder(_MlxModuleBase):
    def __init__(
        self,
        config: Qwen3VLTextConfig | None = None,
        *,
        tokenizer: Any | None = None,
    ) -> None:
        if nn is not None:
            super().__init__()
        self.config = config or Qwen3VLTextConfig()
        self.config.validate()
        self.tokenizer = tokenizer
        if nn is None:
            return

        self.embed_tokens = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
        )
        self.layers = [
            Qwen3VLTextDecoderLayer(self.config)
            for _ in range(self.config.num_hidden_layers)
        ]
        self.norm = Qwen3VLTextRMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    @classmethod
    def from_pretrained(
        cls,
        artifact_dir: str | Path,
    ) -> "Qwen3VLInstructionEncoder":
        _require_mlx()
        root = Path(artifact_dir).expanduser()
        artifact = read_artifact(root, "instruction encoder")

        config = Qwen3VLTextConfig.from_mllm_config(
            read_json(
                component_config_path(root, artifact, "mllm"),
                "instruction encoder",
            )
        )
        tokenizer = _load_tokenizer(processor_path(root, artifact))
        model = cls(config, tokenizer=tokenizer)
        quantized_paths = quantized_linear_paths_for_component(
            root,
            artifact,
            "mllm",
            key_transform=_strip_official_text_weight_prefix,
        )
        quantization_settings = quantization_settings_for_component(
            artifact,
            "mllm",
            quantized_paths,
        )
        if quantization_settings is not None:
            apply_quantization_to_model(model, quantized_paths, quantization_settings)

        selected_weights = load_indexed_component_weights(
            root=root,
            artifact=artifact,
            component="mllm",
            expected_shapes=_flatten_parameter_shapes(model.parameters()),
            mx_module=mx,
            artifact_label="Instruction encoder",
            weight_label="MLLM",
            key_transform=_strip_official_text_weight_prefix,
        )

        model.load_weights(sorted(selected_weights.items()), strict=True)
        mx.eval(*selected_weights.values())
        return model

    def encode(
        self,
        prompt_or_prompts: str | Sequence[str],
        *,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        truncate: bool = False,
    ) -> InstructionEncoding:
        _require_mlx()
        prompts, _ = _normalize_prompts(prompt_or_prompts)
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.tokenizer is None:
            raise BooguTurboMlxError(
                "Qwen3VLInstructionEncoder.encode requires a tokenizer loaded "
                "from a boogu-turbo-mlx artifact."
            )

        rendered = [
            # The official Turbo T2I pipeline encodes the instruction without an
            # assistant generation prompt (its Qwen3VLProcessor.apply_chat_template
            # call leaves add_generation_prompt at its default of False). Adding the
            # trailing "<|im_start|>assistant\n" would feed three spurious
            # instruction tokens to the diffusion transformer.
            self.tokenizer.apply_chat_template(
                build_t2i_chat_messages(prompt),
                tokenize=False,
                add_generation_prompt=False,
            )
            for prompt in prompts
        ]
        tokenized = self.tokenizer(
            rendered,
            padding="longest",
            max_length=max_sequence_length,
            truncation=truncate,
            add_special_tokens=False,
            return_tensors=None,
        )
        input_ids = mx.array(tokenized["input_ids"], dtype=mx.int32)
        attention_mask = mx.array(tokenized["attention_mask"], dtype=mx.int32)
        hidden_states = self(input_ids, attention_mask)
        mx.eval(hidden_states, attention_mask, input_ids)
        return InstructionEncoding(hidden_states, attention_mask, input_ids)

    def __call__(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
        _require_mlx()
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be shaped [B, T]")
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must be shaped like input_ids")

        hidden_states = self.embed_tokens(input_ids)
        position_ids = build_text_position_ids(attention_mask)
        position_embeddings = build_text_rotary_embeddings(
            self.config,
            position_ids,
            dtype=hidden_states.dtype,
        )
        mask = _causal_padding_attention_mask(attention_mask)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, mask)
        return self.norm(hidden_states)


def encode_instruction(
    prompt: str | Sequence[str],
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    truncate: bool = False,
    *,
    model_path: str | Path | None = None,
    encoder: Qwen3VLInstructionEncoder | None = None,
) -> InstructionEncoding:
    """Encode text prompt(s) into Qwen3-VL hidden states for the Boogu transformer."""

    prompts, single_prompt = _normalize_prompts(prompt)
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if encoder is not None and model_path is not None:
        raise ValueError("provide either encoder or model_path, not both")
    if encoder is None:
        if model_path is None:
            raise ValueError("encode_instruction requires either model_path or encoder")
        encoder = Qwen3VLInstructionEncoder.from_pretrained(model_path)

    return encoder.encode(
        prompts[0] if single_prompt else prompts,
        max_sequence_length=max_sequence_length,
        truncate=truncate,
    )


def build_text_position_ids(attention_mask: Any) -> Any:
    _require_mlx()
    positions = mx.cumsum(attention_mask.astype(mx.int32), axis=1) - 1
    return mx.stack(
        [
            mx.where(
                attention_mask.astype(mx.bool_),
                positions,
                mx.ones_like(positions),
            )
            for _ in range(3)
        ],
        axis=0,
    )


def build_text_rotary_embeddings(
    config: Qwen3VLTextConfig,
    position_ids: Any,
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    _require_mlx()
    if position_ids.ndim == 2:
        position_ids = mx.stack([position_ids, position_ids, position_ids], axis=0)
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("position_ids must be shaped [3, B, T] or [B, T]")

    inv_freq = 1.0 / (
        config.rope_theta
        ** (mx.arange(0, config.head_dim, 2, dtype=mx.float32) / config.head_dim)
    )
    freqs = position_ids[:, :, :, None].astype(mx.float32) * inv_freq[None, None, None, :]
    freqs = _apply_interleaved_mrope(freqs, config.mrope_section)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _apply_interleaved_mrope(freqs: Any, mrope_section: tuple[int, int, int]) -> Any:
    half_dim = freqs.shape[-1]
    replacements = []
    for index in range(half_dim):
        axis = _mrope_axis_for_frequency_index(index, mrope_section)
        replacements.append(freqs[axis, ..., index : index + 1])
    return mx.concatenate(replacements, axis=-1)


def _mrope_axis_for_frequency_index(
    index: int,
    mrope_section: tuple[int, int, int],
) -> int:
    for axis in (1, 2):
        length = mrope_section[axis] * 3
        if index < length and index % 3 == axis:
            return axis
    return 0


def _apply_rotary_pos_emb(
    query: Any,
    key: Any,
    position_embeddings: tuple[Any, Any],
) -> tuple[Any, Any]:
    cos, sin = position_embeddings
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    query_embed = (query * cos) + (_rotate_half(query) * sin)
    key_embed = (key * cos) + (_rotate_half(key) * sin)
    return query_embed, key_embed


def _rotate_half(x: Any) -> Any:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _causal_padding_attention_mask(attention_mask: Any) -> Any:
    _require_mlx()
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be shaped [B, T]")
    seq_len = attention_mask.shape[1]
    query_positions = mx.arange(seq_len)[:, None]
    key_positions = mx.arange(seq_len)[None, :]
    causal = query_positions >= key_positions
    key_padding = attention_mask.astype(mx.bool_)[:, None, None, :]
    return causal[None, None, :, :] & key_padding


def _silu(x: Any) -> Any:
    return x * mx.sigmoid(x)


def _normalize_prompts(prompt_or_prompts: str | Sequence[str]) -> tuple[list[str], bool]:
    if isinstance(prompt_or_prompts, str):
        if not prompt_or_prompts.strip():
            raise ValueError("prompt must be a non-empty string")
        return [prompt_or_prompts], True
    if not isinstance(prompt_or_prompts, Sequence):
        raise ValueError("prompt must be a string or sequence of strings")
    prompts = list(prompt_or_prompts)
    if not prompts:
        raise ValueError("prompt sequence must not be empty")
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt sequence entries must be non-empty strings")
    return prompts, False


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard.
        raise BooguTurboMlxError(
            "Instruction tokenization requires transformers>=4.57. Install "
            "`boogu-turbo-mlx[runtime]`."
        ) from exc
    if not path.exists():
        raise BooguTurboMlxError(f"Missing required processor directory: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise BooguTurboMlxError("Processor tokenizer must define a pad token")
    if not getattr(tokenizer, "chat_template", None):
        raise BooguTurboMlxError("Processor tokenizer must define a chat template")
    return tokenizer


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise BooguTurboMlxError(
            "The Qwen3-VL instruction encoder requires MLX. Install "
            "`boogu-turbo-mlx[runtime]` on an MLX-supported machine."
        )


def _tuple_ints3(value: Iterable[Any], name: str) -> tuple[int, int, int]:
    items = tuple(int(item) for item in value)
    if len(items) != 3:
        raise ValueError(f"{name} must contain three integers")
    return items


def _strip_official_text_weight_prefix(key: str) -> str | None:
    if not key.startswith(OFFICIAL_TEXT_WEIGHT_PREFIX):
        return None
    return key.removeprefix(OFFICIAL_TEXT_WEIGHT_PREFIX)
