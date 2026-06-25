from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifacts import (
    component_config_path,
    flatten_parameters as _flatten_parameters,
    flatten_parameter_shapes as _flatten_parameter_shapes,
    load_indexed_component_weights,
    read_artifact,
    read_json,
)
from .errors import BooguTurboMlxError, ComponentNotImplementedError
from .quantization import (
    apply_quantization_to_model,
    quantization_settings_for_component,
    quantized_linear_paths_for_component,
)

try:  # Keep package importable without the optional MLX runtime.
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - exercised by non-MLX environments.
    mx = None
    nn = None

_MlxModuleBase = object if nn is None else nn.Module


@dataclass(frozen=True)
class BooguImageTransformerConfig:
    patch_size: int = 2
    in_channels: int = 16
    out_channels: int | None = None
    hidden_size: int = 3360
    num_layers: int = 40
    num_double_stream_layers: int = 8
    num_refiner_layers: int = 2
    num_attention_heads: int = 28
    num_kv_heads: int = 7
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    axes_dim_rope: tuple[int, int, int] = (40, 40, 40)
    axes_lens: tuple[int, int, int] = (2048, 1664, 1664)
    timestep_scale: float = 1000.0
    instruction_feat_dim: int = 4096
    num_instruction_feature_layers: int = 1
    reduce_type: str = "mean"
    theta: int = 10000

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BooguImageTransformerConfig":
        instruction_config = dict(payload.get("instruction_feature_configs") or {})
        num_instruction_layers = (
            instruction_config.get("num_instruction_feature_layers")
            or instruction_config.get("num_instruction_feat_layers")
            or payload.get("num_instruction_feature_layers")
            or payload.get("num_instruction_feat_layers")
            or 1
        )

        config = cls(
            patch_size=int(payload.get("patch_size", cls.patch_size)),
            in_channels=int(payload.get("in_channels", cls.in_channels)),
            out_channels=(
                None
                if payload.get("out_channels") is None
                else int(payload["out_channels"])
            ),
            hidden_size=int(payload.get("hidden_size", cls.hidden_size)),
            num_layers=int(payload.get("num_layers", cls.num_layers)),
            num_double_stream_layers=int(
                payload.get("num_double_stream_layers", cls.num_double_stream_layers)
            ),
            num_refiner_layers=int(
                payload.get("num_refiner_layers", cls.num_refiner_layers)
            ),
            num_attention_heads=int(
                payload.get("num_attention_heads", cls.num_attention_heads)
            ),
            num_kv_heads=int(
                payload.get(
                    "num_kv_heads",
                    payload.get("num_key_value_heads", cls.num_kv_heads),
                )
            ),
            multiple_of=int(payload.get("multiple_of", cls.multiple_of)),
            ffn_dim_multiplier=payload.get(
                "ffn_dim_multiplier", cls.ffn_dim_multiplier
            ),
            norm_eps=float(payload.get("norm_eps", cls.norm_eps)),
            axes_dim_rope=_int_tuple3(
                payload.get("axes_dim_rope", cls.axes_dim_rope), "axes_dim_rope"
            ),
            axes_lens=_int_tuple3(
                payload.get("axes_lens", cls.axes_lens), "axes_lens"
            ),
            timestep_scale=float(
                payload.get("timestep_scale", cls.timestep_scale)
            ),
            instruction_feat_dim=int(
                instruction_config.get(
                    "instruction_feat_dim",
                    payload.get("instruction_feat_dim", cls.instruction_feat_dim),
                )
            ),
            num_instruction_feature_layers=int(num_instruction_layers),
            reduce_type=str(
                instruction_config.get(
                    "reduce_type", payload.get("reduce_type", cls.reduce_type)
                )
            ),
        )
        config.validate()
        return config

    @property
    def out_channels_effective(self) -> int:
        return self.in_channels if self.out_channels is None else self.out_channels

    @property
    def num_single_stream_layers(self) -> int:
        return self.num_layers - self.num_double_stream_layers

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def conditioning_dim(self) -> int:
        return min(self.hidden_size, 1024)

    @property
    def patch_dim(self) -> int:
        return self.patch_size * self.patch_size * self.in_channels

    @property
    def output_patch_dim(self) -> int:
        return self.patch_size * self.patch_size * self.out_channels_effective

    @property
    def ffn_inner_dim(self) -> int:
        inner_dim = 4 * self.hidden_size
        if self.ffn_dim_multiplier is not None:
            inner_dim = int(self.ffn_dim_multiplier * inner_dim)
        return self.multiple_of * ((inner_dim + self.multiple_of - 1) // self.multiple_of)

    @property
    def preprocessed_instruction_feat_dim(self) -> int:
        if "cat" in self.reduce_type.lower():
            return self.num_instruction_feature_layers * self.instruction_feat_dim
        if "mean" in self.reduce_type.lower():
            return self.instruction_feat_dim
        raise ValueError(f"Invalid reduce_type: {self.reduce_type}")

    @property
    def num_key_value_heads(self) -> int:
        return self.num_kv_heads

    def to_dict(self) -> dict[str, Any]:
        return {
            "_class_name": "BooguImageTransformer2DModel",
            "patch_size": self.patch_size,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_double_stream_layers": self.num_double_stream_layers,
            "num_refiner_layers": self.num_refiner_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "multiple_of": self.multiple_of,
            "ffn_dim_multiplier": self.ffn_dim_multiplier,
            "norm_eps": self.norm_eps,
            "axes_dim_rope": list(self.axes_dim_rope),
            "axes_lens": list(self.axes_lens),
            "timestep_scale": self.timestep_scale,
            "instruction_feature_configs": {
                "instruction_feat_dim": self.instruction_feat_dim,
                "num_instruction_feature_layers": self.num_instruction_feature_layers,
                "reduce_type": self.reduce_type,
            },
        }

    def validate(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.out_channels is not None and self.out_channels <= 0:
            raise ValueError("out_channels must be positive when provided")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")
        if self.head_dim != sum(self.axes_dim_rope):
            raise ValueError(
                "hidden_size // num_attention_heads must equal sum(axes_dim_rope)"
            )
        if any(dim < 0 or dim % 2 for dim in self.axes_dim_rope):
            raise ValueError("axes_dim_rope entries must be non-negative even integers")
        if len(self.axes_lens) != 3 or any(length <= 0 for length in self.axes_lens):
            raise ValueError("axes_lens must contain three positive integers")
        if self.num_double_stream_layers > self.num_layers:
            raise ValueError("num_double_stream_layers cannot exceed num_layers")
        if self.num_refiner_layers < 0:
            raise ValueError("num_refiner_layers cannot be negative")
        if self.multiple_of <= 0:
            raise ValueError("multiple_of must be positive")
        if self.num_instruction_feature_layers <= 0:
            raise ValueError("num_instruction_feature_layers must be positive")
        if self.preprocessed_instruction_feat_dim <= 0:
            raise ValueError("instruction feature dimension must be positive")


class TimestepEmbedding(_MlxModuleBase):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        _require_mlx()
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def __call__(self, sample: Any) -> Any:
        sample = self.linear_1(sample)
        sample = _silu(sample)
        return self.linear_2(sample)


class LuminaRMSNormZero(_MlxModuleBase):
    def __init__(self, embedding_dim: int, norm_eps: float) -> None:
        _require_mlx()
        super().__init__()
        self.linear = nn.Linear(min(embedding_dim, 1024), 4 * embedding_dim, bias=True)
        self.norm = nn.RMSNorm(embedding_dim, eps=norm_eps)

    def __call__(self, x: Any, emb: Any) -> tuple[Any, Any, Any, Any]:
        emb = self.linear(_silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(emb, 4, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, None, :])
        return x, gate_msa, scale_mlp, gate_mlp


class LuminaLayerNormContinuous(_MlxModuleBase):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_dim: int,
        out_dim: int,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.linear_1 = nn.Linear(conditioning_dim, embedding_dim, bias=True)
        self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=True)
        self.eps = 1e-6

    def __call__(self, x: Any, conditioning_embedding: Any) -> Any:
        emb = self.linear_1(_silu(conditioning_embedding).astype(x.dtype))
        x = _layer_norm_no_affine(x, self.eps)
        x = x * (1 + emb[:, None, :])
        return self.linear_2(x)


class LuminaFeedForward(_MlxModuleBase):
    def __init__(self, config: BooguImageTransformerConfig) -> None:
        _require_mlx()
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.ffn_inner_dim, bias=False)
        self.linear_2 = nn.Linear(config.ffn_inner_dim, config.hidden_size, bias=False)
        self.linear_3 = nn.Linear(config.hidden_size, config.ffn_inner_dim, bias=False)

    def __call__(self, x: Any) -> Any:
        h1 = self.linear_1(x)
        h2 = self.linear_3(x)
        return self.linear_2(_silu(h1.astype(mx.float32)).astype(h1.dtype) * h2)


class Lumina2CombinedTimestepCaptionEmbedding(_MlxModuleBase):
    def __init__(self, config: BooguImageTransformerConfig) -> None:
        _require_mlx()
        super().__init__()
        self.frequency_embedding_size = 256
        self.timestep_scale = config.timestep_scale
        self.timestep_embedder = TimestepEmbedding(
            self.frequency_embedding_size, config.conditioning_dim
        )
        self.caption_embedder = [
            nn.RMSNorm(config.preprocessed_instruction_feat_dim, eps=config.norm_eps),
            nn.Linear(
                config.preprocessed_instruction_feat_dim,
                config.hidden_size,
                bias=True,
            ),
        ]

    def __call__(
        self,
        timestep: Any,
        instruction_hidden_states: Any,
        dtype: Any,
    ) -> tuple[Any, Any]:
        timestep_proj = _timesteps_embedding(
            timestep, self.frequency_embedding_size, scale=self.timestep_scale
        ).astype(dtype)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder[1](
            self.caption_embedder[0](instruction_hidden_states)
        )
        return time_embed, caption_embed


class BooguAttention(_MlxModuleBase):
    def __init__(self, config: BooguImageTransformerConfig) -> None:
        _require_mlx()
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.to_k = nn.Linear(config.hidden_size, config.kv_dim, bias=False)
        self.to_v = nn.Linear(config.hidden_size, config.kv_dim, bias=False)
        self.to_out = [nn.Linear(config.hidden_size, config.hidden_size, bias=False)]
        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-5)

    def __call__(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        attention_mask: Any,
        image_rotary_emb: Any,
    ) -> Any:
        batch_size = hidden_states.shape[0]
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query = query.reshape(batch_size, -1, self.heads, self.head_dim)
        key = key.reshape(batch_size, -1, self.kv_heads, self.head_dim)
        value = value.reshape(batch_size, -1, self.kv_heads, self.head_dim)

        query = self.norm_q(query)
        key = self.norm_k(key)
        if image_rotary_emb is not None:
            query = _apply_rotary_emb(query, image_rotary_emb)
            key = _apply_rotary_emb(key, image_rotary_emb)

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        mask = _attention_mask(attention_mask, batch_size)

        hidden_states = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=mask
        )
        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, self.heads * self.head_dim
        )
        return self.to_out[0](hidden_states)


class BooguDoubleStreamSelfAttentionProcessor(_MlxModuleBase):
    def __init__(self, config: BooguImageTransformerConfig) -> None:
        _require_mlx()
        super().__init__()
        query_dim = config.hidden_size
        kv_dim = config.kv_dim
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads

        self.img_to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.img_to_k = nn.Linear(query_dim, kv_dim, bias=False)
        self.img_to_v = nn.Linear(query_dim, kv_dim, bias=False)
        self.instruct_to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.instruct_to_k = nn.Linear(query_dim, kv_dim, bias=False)
        self.instruct_to_v = nn.Linear(query_dim, kv_dim, bias=False)
        self.instruct_out = nn.Linear(query_dim, query_dim, bias=False)
        self.img_out = nn.Linear(query_dim, query_dim, bias=False)

    def __call__(
        self,
        attn: "BooguJointAttention",
        img_hidden_states: Any,
        instruct_hidden_states: Any,
        joint_attention_mask: Any,
        rotary_emb: Any,
        encoder_seq_lengths: list[int],
        seq_lengths: list[int],
    ) -> Any:
        batch_size = img_hidden_states.shape[0]

        img_q = self.img_to_q(img_hidden_states)
        img_k = self.img_to_k(img_hidden_states)
        img_v = self.img_to_v(img_hidden_states)
        instruct_q = self.instruct_to_q(instruct_hidden_states)
        instruct_k = self.instruct_to_k(instruct_hidden_states)
        instruct_v = self.instruct_to_v(instruct_hidden_states)

        query, key, value = self._concat_instruction_image_features(
            [img_q, img_k, img_v],
            [instruct_q, instruct_k, instruct_v],
            encoder_seq_lengths,
            seq_lengths,
        )

        query = query.reshape(batch_size, -1, attn.heads, attn.head_dim)
        key = key.reshape(batch_size, -1, attn.kv_heads, attn.head_dim)
        value = value.reshape(batch_size, -1, attn.kv_heads, attn.head_dim)

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        mask = _attention_mask(joint_attention_mask, batch_size)

        hidden_states = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=attn.scale, mask=mask
        )
        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(
            batch_size, -1, attn.heads * attn.head_dim
        )

        instruct_hidden_states, img_hidden_states = self._split_instruction_image_features(
            hidden_states, encoder_seq_lengths, seq_lengths
        )
        instruct_projected = self.instruct_out(instruct_hidden_states)
        img_projected = self.img_out(img_hidden_states)
        merged = self._concat_instruction_image_features(
            [img_projected],
            [instruct_projected],
            encoder_seq_lengths,
            seq_lengths,
        )[0]
        return attn.to_out[0](merged)

    def _concat_instruction_image_features(
        self,
        img_hidden_states_list: list[Any],
        instruct_hidden_states_list: list[Any],
        encoder_seq_lengths: list[int],
        seq_lengths: list[int],
    ) -> list[Any]:
        max_seq_len = max(seq_lengths)
        concatenated_list = []
        for img_tensor, instruct_tensor in zip(
            img_hidden_states_list, instruct_hidden_states_list
        ):
            rows = []
            for i, (encoder_seq_len, seq_len) in enumerate(
                zip(encoder_seq_lengths, seq_lengths)
            ):
                row = mx.concatenate(
                    [
                        instruct_tensor[i, :encoder_seq_len],
                        img_tensor[i, : seq_len - encoder_seq_len],
                    ],
                    axis=0,
                )
                rows.append(_pad_tokens(row, max_seq_len))
            concatenated_list.append(mx.stack(rows, axis=0))
        return concatenated_list

    def _split_instruction_image_features(
        self,
        hidden_states: Any,
        encoder_seq_lengths: list[int],
        seq_lengths: list[int],
    ) -> tuple[Any, Any]:
        max_instruct_len = max(encoder_seq_lengths)
        max_img_len = max(
            seq_len - encoder_seq_len
            for seq_len, encoder_seq_len in zip(seq_lengths, encoder_seq_lengths)
        )
        instruct_rows = []
        img_rows = []
        for i, (encoder_seq_len, seq_len) in enumerate(
            zip(encoder_seq_lengths, seq_lengths)
        ):
            instruct_rows.append(_pad_tokens(hidden_states[i, :encoder_seq_len], max_instruct_len))
            img_rows.append(
                _pad_tokens(hidden_states[i, encoder_seq_len:seq_len], max_img_len)
            )
        return mx.stack(instruct_rows, axis=0), mx.stack(img_rows, axis=0)


class BooguJointAttention(_MlxModuleBase):
    def __init__(self, config: BooguImageTransformerConfig) -> None:
        _require_mlx()
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-5)
        self.processor = BooguDoubleStreamSelfAttentionProcessor(config)
        self.to_out = [nn.Linear(config.hidden_size, config.hidden_size, bias=False)]


class BooguImageTransformerBlock(_MlxModuleBase):
    def __init__(self, config: BooguImageTransformerConfig, modulation: bool) -> None:
        _require_mlx()
        super().__init__()
        self.modulation = modulation
        self.attn = BooguAttention(config)
        self.feed_forward = LuminaFeedForward(config)
        self.norm1 = (
            LuminaRMSNormZero(config.hidden_size, config.norm_eps)
            if modulation
            else nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        )
        self.ffn_norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        hidden_states: Any,
        attention_mask: Any,
        image_rotary_emb: Any,
        temb: Any | None = None,
    ) -> Any:
        if self.modulation:
            if temb is None:
                raise ValueError("temb must be provided when modulation is enabled")
            norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(
                hidden_states, temb
            )
            attn_output = self.attn(
                norm_hidden_states,
                norm_hidden_states,
                attention_mask,
                image_rotary_emb,
            )
            hidden_states = hidden_states + mx.tanh(gate_msa[:, None, :]) * self.norm2(
                attn_output
            )
            mlp_output = self.feed_forward(
                self.ffn_norm1(hidden_states) * (1 + scale_mlp[:, None, :])
            )
            return hidden_states + mx.tanh(gate_mlp[:, None, :]) * self.ffn_norm2(
                mlp_output
            )

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn(
            norm_hidden_states,
            norm_hidden_states,
            attention_mask,
            image_rotary_emb,
        )
        hidden_states = hidden_states + self.norm2(attn_output)
        mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
        return hidden_states + self.ffn_norm2(mlp_output)


class BooguImageDoubleStreamTransformerBlock(_MlxModuleBase):
    def __init__(self, config: BooguImageTransformerConfig) -> None:
        _require_mlx()
        super().__init__()
        self.hidden_size = config.hidden_size
        self.img_instruct_attn = BooguJointAttention(config)
        self.img_self_attn = BooguAttention(config)
        self.img_feed_forward = LuminaFeedForward(config)

        self.img_norm1 = LuminaRMSNormZero(config.hidden_size, config.norm_eps)
        self.img_norm2 = LuminaRMSNormZero(config.hidden_size, config.norm_eps)
        self.img_norm3 = LuminaRMSNormZero(config.hidden_size, config.norm_eps)
        self.img_ffn_norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.img_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.img_self_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.img_ffn_norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.instruct_feed_forward = LuminaFeedForward(config)
        self.instruct_norm1 = LuminaRMSNormZero(config.hidden_size, config.norm_eps)
        self.instruct_norm2 = LuminaRMSNormZero(config.hidden_size, config.norm_eps)
        self.instruct_ffn_norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.instruct_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.instruct_ffn_norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)

    def __call__(
        self,
        img_hidden_states: Any,
        instruct_hidden_states: Any,
        img_attention_mask: Any,
        joint_attention_mask: Any,
        image_rotary_emb: Any,
        rotary_emb: Any,
        temb: Any,
        encoder_seq_lengths: list[int],
        seq_lengths: list[int],
    ) -> tuple[Any, Any]:
        img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(
            img_hidden_states, temb
        )
        img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
        img_norm3_out, img_gate_self, _, _ = self.img_norm3(img_hidden_states, temb)

        (
            instruct_norm1_out,
            instruct_gate_msa,
            instruct_scale_mlp,
            instruct_gate_mlp,
        ) = self.instruct_norm1(instruct_hidden_states, temb)
        instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(
            instruct_hidden_states, temb
        )

        joint_attn_out = self.img_instruct_attn.processor(
            self.img_instruct_attn,
            img_norm1_out,
            instruct_norm1_out,
            joint_attention_mask,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        )
        instruct_attn_out, img_attn_out = _split_joint_outputs(
            joint_attn_out,
            encoder_seq_lengths,
            seq_lengths,
            instruct_hidden_states.shape[1],
            img_hidden_states.shape[1],
        )

        img_self_attn_out = self.img_self_attn(
            img_norm3_out,
            img_norm3_out,
            img_attention_mask,
            image_rotary_emb,
        )

        img_hidden_states = img_hidden_states + mx.tanh(
            img_gate_msa[:, None, :]
        ) * self.img_attn_norm(img_attn_out)
        img_hidden_states = img_hidden_states + mx.tanh(
            img_gate_self[:, None, :]
        ) * self.img_self_attn_norm(img_self_attn_out)

        img_mlp_input = (
            1 + img_scale_mlp[:, None, :]
        ) * img_norm2_out + img_shift_mlp[:, None, :]
        img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
        img_hidden_states = img_hidden_states + mx.tanh(
            img_gate_mlp[:, None, :]
        ) * self.img_ffn_norm2(img_mlp_out)

        instruct_hidden_states = instruct_hidden_states + mx.tanh(
            instruct_gate_msa[:, None, :]
        ) * self.instruct_attn_norm(instruct_attn_out)
        instruct_mlp_input = (
            1 + instruct_scale_mlp[:, None, :]
        ) * instruct_norm2_out + instruct_shift_mlp[:, None, :]
        instruct_mlp_out = self.instruct_feed_forward(
            self.instruct_ffn_norm1(instruct_mlp_input)
        )
        instruct_hidden_states = instruct_hidden_states + mx.tanh(
            instruct_gate_mlp[:, None, :]
        ) * self.instruct_ffn_norm2(instruct_mlp_out)

        return img_hidden_states, instruct_hidden_states


class BooguImageTransformer(_MlxModuleBase):
    def __init__(
        self,
        config: BooguImageTransformerConfig | None = None,
    ) -> None:
        if nn is not None:
            super().__init__()
        self.config = config or BooguImageTransformerConfig()
        self.config.validate()
        if nn is None:
            return

        self.x_embedder = nn.Linear(
            self.config.patch_dim, self.config.hidden_size, bias=True
        )
        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(self.config)
        self.noise_refiner = [
            BooguImageTransformerBlock(self.config, modulation=True)
            for _ in range(self.config.num_refiner_layers)
        ]
        self.context_refiner = [
            BooguImageTransformerBlock(self.config, modulation=False)
            for _ in range(self.config.num_refiner_layers)
        ]
        self.double_stream_layers = [
            BooguImageDoubleStreamTransformerBlock(self.config)
            for _ in range(self.config.num_double_stream_layers)
        ]
        self.single_stream_layers = [
            BooguImageTransformerBlock(self.config, modulation=True)
            for _ in range(self.config.num_single_stream_layers)
        ]
        self.norm_out = LuminaLayerNormContinuous(
            self.config.hidden_size,
            self.config.conditioning_dim,
            self.config.output_patch_dim,
        )
        # Loaded for strict checkpoint parity; only the out-of-scope reference-image path uses it.
        self.image_index_embedding = (
            mx.random.normal((5, self.config.hidden_size)) * 0.02
        )

    @classmethod
    def from_pretrained(
        cls,
        artifact_dir: str | Path,
    ) -> "BooguImageTransformer":
        _require_mlx()
        root = Path(artifact_dir).expanduser()
        artifact = read_artifact(root, "transformer")

        transformer_config = read_json(
            component_config_path(root, artifact, "transformer"),
            "transformer",
        )
        config = BooguImageTransformerConfig.from_dict(transformer_config)
        model = cls(config)
        quantized_paths = quantized_linear_paths_for_component(
            root,
            artifact,
            "transformer",
        )
        quantization_settings = quantization_settings_for_component(
            artifact,
            "transformer",
            quantized_paths,
        )
        if quantization_settings is not None:
            apply_quantization_to_model(model, quantized_paths, quantization_settings)

        selected_weights = load_indexed_component_weights(
            root=root,
            artifact=artifact,
            component="transformer",
            expected_shapes=_flatten_parameter_shapes(model.parameters()),
            mx_module=mx,
            artifact_label="Transformer",
            weight_label="transformer",
        )

        model.load_weights(sorted(selected_weights.items()), strict=True)
        mx.eval(*selected_weights.values())
        return model

    def __call__(
        self,
        latents: Any,
        timestep: Any,
        instruction_hidden_states: Any,
        freqs_cis: Iterable[Any] | None,
        instruction_attention_mask: Any,
        ref_image_hidden_states: Any | None = None,
        prepared_inputs: "TransformerPreparedInputs | None" = None,
    ) -> Any:
        _require_mlx()
        if ref_image_hidden_states is not None:
            raise ComponentNotImplementedError(
                "Reference-image transformer branches are planned for a later milestone."
            )
        if latents.ndim != 4:
            raise ValueError("latents must be shaped [B, C, H, W]")
        batch_size, channels, height, width = latents.shape
        if channels != self.config.in_channels:
            raise ValueError(
                f"latents channel count {channels} does not match config in_channels "
                f"{self.config.in_channels}"
            )
        if height % self.config.patch_size or width % self.config.patch_size:
            raise ValueError("latent height and width must be divisible by patch_size")

        instruction_hidden_states = self.preprocess_instruction_hidden_states(
            instruction_hidden_states
        )
        if instruction_hidden_states.shape[0] != batch_size:
            raise ValueError("instruction_hidden_states batch does not match latents")
        if instruction_attention_mask.shape[:2] != instruction_hidden_states.shape[:2]:
            raise ValueError(
                "instruction_attention_mask must be shaped [B, instruction_length]"
            )

        if prepared_inputs is None:
            prepared_inputs = self.prepare_forward_inputs(
                freqs_cis,
                instruction_attention_mask,
                height,
                width,
            )
        self._validate_prepared_inputs(
            prepared_inputs,
            batch_size=batch_size,
            height=height,
            width=width,
            instruction_attention_mask=instruction_attention_mask,
        )
        return self._forward_core(
            latents,
            timestep,
            instruction_hidden_states,
            prepared_inputs,
        )

    def prepare_forward_inputs(
        self,
        freqs_cis: Iterable[Any] | None,
        instruction_attention_mask: Any,
        height: int,
        width: int,
    ) -> "TransformerPreparedInputs":
        _require_mlx()
        if freqs_cis is None:
            freqs_cis = build_transformer_freqs_cis(self.config)
        batch_size = int(instruction_attention_mask.shape[0])
        if height % self.config.patch_size or width % self.config.patch_size:
            raise ValueError("latent height and width must be divisible by patch_size")
        img_len = (height // self.config.patch_size) * (width // self.config.patch_size)
        rope = build_rotary_embeddings_for_latents(
            self.config,
            freqs_cis,
            instruction_attention_mask,
            height,
            width,
        )
        return TransformerPreparedInputs(
            rotary_embeddings=rope,
            context_attention_mask=_attention_mask(
                instruction_attention_mask,
                batch_size,
            ),
            noise_attention_mask=_attention_mask(
                _mask_from_lengths([img_len] * batch_size, img_len),
                batch_size,
            ),
            img_attention_mask=_attention_mask(
                _mask_from_lengths(
                    rope.combined_img_seq_lengths,
                    max(rope.combined_img_seq_lengths),
                ),
                batch_size,
            ),
            joint_attention_mask=_attention_mask(
                _mask_from_lengths(rope.seq_lengths, max(rope.seq_lengths)),
                batch_size,
            ),
            height=int(height),
            width=int(width),
            batch_size=batch_size,
        )

    def _validate_prepared_inputs(
        self,
        prepared_inputs: "TransformerPreparedInputs",
        *,
        batch_size: int,
        height: int,
        width: int,
        instruction_attention_mask: Any,
    ) -> None:
        if prepared_inputs.batch_size != batch_size:
            raise ValueError("prepared transformer inputs batch does not match latents")
        if prepared_inputs.height != height or prepared_inputs.width != width:
            raise ValueError("prepared transformer inputs geometry does not match latents")
        if (
            len(prepared_inputs.rotary_embeddings.encoder_seq_lengths)
            != instruction_attention_mask.shape[0]
        ):
            raise ValueError("prepared transformer inputs do not match instruction batch")

    def _forward_core(
        self,
        latents: Any,
        timestep: Any,
        instruction_hidden_states: Any,
        prepared_inputs: "TransformerPreparedInputs",
    ) -> Any:
        batch_size, _, height, width = latents.shape
        rope = prepared_inputs.rotary_embeddings
        temb, instruction_hidden_states = self.time_caption_embed(
            timestep, instruction_hidden_states, latents.dtype
        )

        img_tokens = patchify_latents(latents, self.config.patch_size)
        img_tokens = self.x_embedder(img_tokens)

        for layer in self.context_refiner:
            instruction_hidden_states = layer(
                instruction_hidden_states,
                prepared_inputs.context_attention_mask,
                rope.context_rotary_emb,
            )

        for layer in self.noise_refiner:
            img_tokens = layer(
                img_tokens,
                prepared_inputs.noise_attention_mask,
                rope.noise_rotary_emb,
                temb,
            )

        instruct_hidden_states = instruction_hidden_states
        img_hidden_states = img_tokens
        if self.double_stream_layers:
            for layer in self.double_stream_layers:
                img_hidden_states, instruct_hidden_states = layer(
                    img_hidden_states,
                    instruct_hidden_states,
                    prepared_inputs.img_attention_mask,
                    prepared_inputs.joint_attention_mask,
                    rope.combined_img_rotary_emb,
                    rope.rotary_emb,
                    temb,
                    rope.encoder_seq_lengths,
                    rope.seq_lengths,
                )

        hidden_states = _fuse_joint_hidden_states(
            instruct_hidden_states,
            img_hidden_states,
            rope.encoder_seq_lengths,
            rope.seq_lengths,
        )

        for layer in self.single_stream_layers:
            hidden_states = layer(
                hidden_states,
                prepared_inputs.joint_attention_mask,
                rope.rotary_emb,
                temb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        img_len = img_tokens.shape[1]
        output_tokens = []
        for i, seq_len in enumerate(rope.seq_lengths):
            output_tokens.append(hidden_states[i, seq_len - img_len : seq_len])
        output_tokens = mx.stack(output_tokens, axis=0)
        return unpatchify_latents(
            output_tokens,
            self.config.patch_size,
            self.config.out_channels_effective,
            height,
            width,
        )

    def preprocess_instruction_hidden_states(self, raw_instruction_hidden_states: Any) -> Any:
        if isinstance(raw_instruction_hidden_states, (list, tuple)):
            if len(raw_instruction_hidden_states) != self.config.num_instruction_feature_layers:
                raise ValueError(
                    "instruction hidden state list length does not match config"
                )
            if "cat" in self.config.reduce_type.lower():
                instruction_hidden_states = mx.concatenate(
                    list(raw_instruction_hidden_states), axis=-1
                )
            elif "mean" in self.config.reduce_type.lower():
                instruction_hidden_states = mx.mean(
                    mx.stack(list(raw_instruction_hidden_states), axis=0), axis=0
                )
            else:
                raise ValueError(f"Invalid reduce_type: {self.config.reduce_type}")
        else:
            instruction_hidden_states = raw_instruction_hidden_states

        if (
            instruction_hidden_states.shape[-1]
            != self.config.preprocessed_instruction_feat_dim
        ):
            raise ValueError(
                "instruction hidden state feature dimension does not match config"
            )
        return instruction_hidden_states


@dataclass(frozen=True)
class TransformerRotaryEmbeddings:
    context_rotary_emb: Any
    ref_img_rotary_emb: Any
    noise_rotary_emb: Any
    rotary_emb: Any
    encoder_seq_lengths: list[int]
    seq_lengths: list[int]
    combined_img_rotary_emb: Any
    combined_img_seq_lengths: list[int]


@dataclass(frozen=True)
class TransformerPreparedInputs:
    rotary_embeddings: TransformerRotaryEmbeddings
    context_attention_mask: Any
    noise_attention_mask: Any
    img_attention_mask: Any
    joint_attention_mask: Any
    height: int
    width: int
    batch_size: int

    def mlx_values(self) -> tuple[Any, ...]:
        rope = self.rotary_embeddings
        return (
            rope.context_rotary_emb,
            rope.ref_img_rotary_emb,
            rope.noise_rotary_emb,
            rope.rotary_emb,
            rope.combined_img_rotary_emb,
            self.context_attention_mask,
            self.noise_attention_mask,
            self.img_attention_mask,
            self.joint_attention_mask,
        )


def build_transformer_freqs_cis(config: BooguImageTransformerConfig) -> tuple[Any, Any, Any]:
    _require_mlx()
    config.validate()
    return tuple(
        _get_1d_rotary_pos_embed(dim, length, theta=config.theta)
        for dim, length in zip(config.axes_dim_rope, config.axes_lens)
    )


def build_rotary_embeddings_for_latents(
    config: BooguImageTransformerConfig,
    freqs_cis: Iterable[Any],
    attention_mask: Any,
    height: int,
    width: int,
) -> TransformerRotaryEmbeddings:
    _require_mlx()
    p = config.patch_size
    batch_size = attention_mask.shape[0]
    encoder_seq_len = attention_mask.shape[1]
    h_tokens = height // p
    w_tokens = width // p
    img_len = h_tokens * w_tokens

    cap_lengths = [int(value) for value in mx.sum(attention_mask, axis=1).tolist()]
    seq_lengths = [cap_len + img_len for cap_len in cap_lengths]
    max_seq_len = max(seq_lengths)

    axis_freqs = tuple(freqs_cis)
    full_rows = []
    context_rows = []
    image_rows = []
    for cap_len, seq_len in zip(cap_lengths, seq_lengths):
        text_positions = mx.repeat(mx.arange(cap_len, dtype=mx.int32)[:, None], 3, axis=1)
        row_ids = mx.repeat(mx.arange(h_tokens, dtype=mx.int32), w_tokens)
        col_ids = mx.tile(mx.arange(w_tokens, dtype=mx.int32), h_tokens)
        image_positions = mx.stack(
            [
                mx.full((img_len,), cap_len, dtype=mx.int32),
                row_ids,
                col_ids,
            ],
            axis=1,
        )
        position_ids = mx.concatenate([text_positions, image_positions], axis=0)
        position_ids = _pad_positions(position_ids, max_seq_len)
        gathered = mx.concatenate(
            [
                mx.take(axis_freqs[axis], position_ids[:, axis], axis=0)
                for axis in range(3)
            ],
            axis=-1,
        )
        full_rows.append(gathered)
        context_rows.append(_pad_tokens(gathered[:cap_len], encoder_seq_len))
        image_rows.append(_pad_tokens(gathered[cap_len:seq_len], img_len))

    full = mx.stack(full_rows, axis=0)
    context = mx.stack(context_rows, axis=0)
    image = mx.stack(image_rows, axis=0)
    empty_ref = mx.zeros((batch_size, 0, full.shape[-1]), dtype=full.dtype)
    return TransformerRotaryEmbeddings(
        context_rotary_emb=context,
        ref_img_rotary_emb=empty_ref,
        noise_rotary_emb=image,
        rotary_emb=full,
        encoder_seq_lengths=cap_lengths,
        seq_lengths=seq_lengths,
        combined_img_rotary_emb=image,
        combined_img_seq_lengths=[img_len] * batch_size,
    )


def patchify_latents(latents: Any, patch_size: int) -> Any:
    _require_mlx()
    batch_size, channels, height, width = latents.shape
    if height % patch_size or width % patch_size:
        raise ValueError("latent height and width must be divisible by patch_size")
    h_tokens = height // patch_size
    w_tokens = width // patch_size
    latents = latents.reshape(
        batch_size, channels, h_tokens, patch_size, w_tokens, patch_size
    )
    latents = latents.transpose(0, 2, 4, 3, 5, 1)
    return latents.reshape(batch_size, h_tokens * w_tokens, patch_size * patch_size * channels)


def unpatchify_latents(
    tokens: Any,
    patch_size: int,
    channels: int,
    height: int,
    width: int,
) -> Any:
    _require_mlx()
    h_tokens = height // patch_size
    w_tokens = width // patch_size
    tokens = tokens.reshape(
        tokens.shape[0], h_tokens, w_tokens, patch_size, patch_size, channels
    )
    tokens = tokens.transpose(0, 5, 1, 3, 2, 4)
    return tokens.reshape(tokens.shape[0], channels, height, width)


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise BooguTurboMlxError(
            "The Boogu transformer runtime requires MLX. Install "
            "`boogu-turbo-mlx[runtime]` on an MLX-supported machine."
        )


def _int_tuple3(value: Iterable[Any], name: str) -> tuple[int, int, int]:
    items = tuple(int(item) for item in value)
    if len(items) != 3:
        raise ValueError(f"{name} must contain three integers")
    return items


def _silu(x: Any) -> Any:
    return x * mx.sigmoid(x)


def _layer_norm_no_affine(x: Any, eps: float) -> Any:
    x_float = x.astype(mx.float32)
    mean = mx.mean(x_float, axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x_float - mean), axis=-1, keepdims=True)
    return ((x_float - mean) * mx.rsqrt(variance + eps)).astype(x.dtype)


def _timesteps_embedding(timesteps: Any, dim: int, *, scale: float) -> Any:
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.astype(mx.float32) * scale
    half_dim = dim // 2
    exponent = -math.log(10000.0) * mx.arange(half_dim, dtype=mx.float32) / half_dim
    freqs = mx.exp(exponent)
    args = timesteps[:, None] * freqs[None, :]
    embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        embedding = mx.pad(embedding, [(0, 0), (0, 1)])
    return embedding


def _get_1d_rotary_pos_embed(dim: int, length: int, *, theta: int) -> Any:
    if dim == 0:
        return mx.zeros((length, 0), dtype=mx.complex64)
    freqs = 1.0 / (
        theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / max(dim, 1))
    )
    positions = mx.arange(length, dtype=mx.float32)
    angles = positions[:, None] * freqs[None, :]
    return (mx.cos(angles) + 1j * mx.sin(angles)).astype(mx.complex64)


def _apply_rotary_emb(x: Any, freqs_cis: Any) -> Any:
    dtype = x.dtype
    x_float = x.astype(mx.float32)
    x_pair = x_float.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x_complex = x_pair[..., 0] + 1j * x_pair[..., 1]
    freqs = freqs_cis[:, : x.shape[1], :][:, :, None, :]
    out = x_complex * freqs
    out = mx.stack([out.real, out.imag], axis=-1).reshape(x.shape)
    return out.astype(dtype)


def _attention_mask(mask: Any, batch_size: int) -> Any:
    if mask is None:
        return None
    if mask.ndim == 4:
        return mask if mask.dtype == mx.bool_ else mask.astype(mx.bool_)
    mask = mask.astype(mx.bool_)
    if mask.ndim == 2:
        return mask.reshape(batch_size, 1, 1, -1)
    if mask.ndim == 3:
        return mask[:, None, :, :]
    raise ValueError(f"Unsupported attention mask shape: {mask.shape}")


def _mask_from_lengths(lengths: list[int], max_length: int) -> Any:
    arange = mx.arange(max_length)[None, :]
    return arange < mx.array(lengths)[:, None]


def _pad_tokens(tokens: Any, length: int) -> Any:
    pad = length - tokens.shape[0]
    if pad < 0:
        raise ValueError("cannot pad token sequence to a shorter length")
    if pad == 0:
        return tokens
    return mx.pad(tokens, [(0, pad), (0, 0)])


def _pad_positions(position_ids: Any, length: int) -> Any:
    pad = length - position_ids.shape[0]
    if pad == 0:
        return position_ids
    return mx.pad(position_ids, [(0, pad), (0, 0)])


def _split_joint_outputs(
    joint_hidden_states: Any,
    encoder_seq_lengths: list[int],
    seq_lengths: list[int],
    instruct_length: int,
    img_length: int,
) -> tuple[Any, Any]:
    instruct_rows = []
    img_rows = []
    for i, (encoder_seq_len, seq_len) in enumerate(
        zip(encoder_seq_lengths, seq_lengths)
    ):
        instruct_rows.append(
            _pad_tokens(joint_hidden_states[i, :encoder_seq_len], instruct_length)
        )
        img_rows.append(
            _pad_tokens(joint_hidden_states[i, encoder_seq_len:seq_len], img_length)
        )
    return mx.stack(instruct_rows, axis=0), mx.stack(img_rows, axis=0)


def _fuse_joint_hidden_states(
    instruct_hidden_states: Any,
    img_hidden_states: Any,
    encoder_seq_lengths: list[int],
    seq_lengths: list[int],
) -> Any:
    rows = []
    max_seq_len = max(seq_lengths)
    for i, (encoder_seq_len, seq_len) in enumerate(
        zip(encoder_seq_lengths, seq_lengths)
    ):
        row = mx.concatenate(
            [
                instruct_hidden_states[i, :encoder_seq_len],
                img_hidden_states[i, : seq_len - encoder_seq_len],
            ],
            axis=0,
        )
        rows.append(_pad_tokens(row, max_seq_len))
    return mx.stack(rows, axis=0)
