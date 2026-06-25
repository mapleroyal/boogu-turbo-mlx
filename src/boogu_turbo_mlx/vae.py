from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifacts import (
    component_config_path,
    flatten_parameter_shapes as _flatten_parameter_shapes,
    load_indexed_component_weights,
    read_artifact,
    read_json,
)
from .errors import BooguTurboMlxError, ComponentNotImplementedError

try:  # Keep package importable without the optional MLX runtime.
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover - exercised by non-MLX environments.
    mx = None
    nn = None

_MlxModuleBase = object if nn is None else nn.Module


@dataclass(frozen=True)
class AutoencoderKLDecoderConfig:
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    act_fn: str = "silu"
    norm_num_groups: int = 32
    force_upcast: bool = True
    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False
    mid_block_add_attention: bool = True
    up_block_types: tuple[str, ...] = (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )
    down_block_types: tuple[str, ...] = (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AutoencoderKLDecoderConfig":
        config = cls(
            in_channels=int(payload.get("in_channels", cls.in_channels)),
            out_channels=int(payload.get("out_channels", cls.out_channels)),
            latent_channels=int(payload.get("latent_channels", cls.latent_channels)),
            block_out_channels=_tuple_ints(
                payload.get("block_out_channels", cls.block_out_channels),
                "block_out_channels",
            ),
            layers_per_block=int(
                payload.get("layers_per_block", cls.layers_per_block)
            ),
            act_fn=str(payload.get("act_fn", cls.act_fn)),
            norm_num_groups=int(payload.get("norm_num_groups", cls.norm_num_groups)),
            force_upcast=bool(payload.get("force_upcast", cls.force_upcast)),
            scaling_factor=float(
                payload.get("scaling_factor", cls.scaling_factor)
            ),
            shift_factor=float(payload.get("shift_factor", cls.shift_factor)),
            use_quant_conv=bool(payload.get("use_quant_conv", cls.use_quant_conv)),
            use_post_quant_conv=bool(
                payload.get("use_post_quant_conv", cls.use_post_quant_conv)
            ),
            mid_block_add_attention=bool(
                payload.get("mid_block_add_attention", cls.mid_block_add_attention)
            ),
            up_block_types=_tuple_strs(
                payload.get("up_block_types", cls.up_block_types),
                "up_block_types",
            ),
            down_block_types=_tuple_strs(
                payload.get("down_block_types", cls.down_block_types),
                "down_block_types",
            ),
        )
        config.validate()
        return config

    @property
    def decoder_channels(self) -> tuple[int, ...]:
        return tuple(reversed(self.block_out_channels))

    def to_dict(self) -> dict[str, Any]:
        return {
            "_class_name": "AutoencoderKL",
            "act_fn": self.act_fn,
            "block_out_channels": list(self.block_out_channels),
            "down_block_types": list(self.down_block_types),
            "force_upcast": self.force_upcast,
            "in_channels": self.in_channels,
            "latent_channels": self.latent_channels,
            "layers_per_block": self.layers_per_block,
            "mid_block_add_attention": self.mid_block_add_attention,
            "norm_num_groups": self.norm_num_groups,
            "out_channels": self.out_channels,
            "scaling_factor": self.scaling_factor,
            "shift_factor": self.shift_factor,
            "up_block_types": list(self.up_block_types),
            "use_post_quant_conv": self.use_post_quant_conv,
            "use_quant_conv": self.use_quant_conv,
        }

    def validate(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if self.latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if not self.block_out_channels:
            raise ValueError("block_out_channels must not be empty")
        if any(channel <= 0 for channel in self.block_out_channels):
            raise ValueError("block_out_channels entries must be positive")
        if self.layers_per_block <= 0:
            raise ValueError("layers_per_block must be positive")
        if self.norm_num_groups <= 0:
            raise ValueError("norm_num_groups must be positive")
        if any(channel % self.norm_num_groups for channel in self.block_out_channels):
            raise ValueError(
                "all block_out_channels entries must be divisible by norm_num_groups"
            )
        if self.act_fn != "silu":
            raise ValueError("VAE decoder supports only act_fn='silu'")
        if self.use_quant_conv:
            raise ValueError("VAE decoder supports only use_quant_conv=false")
        if self.use_post_quant_conv:
            raise ValueError("VAE decoder supports only use_post_quant_conv=false")
        if not self.mid_block_add_attention:
            raise ValueError(
                "VAE decoder supports only mid_block_add_attention=true"
            )
        if len(self.up_block_types) != len(self.block_out_channels):
            raise ValueError("up_block_types length must match block_out_channels")
        if any(block_type != "UpDecoderBlock2D" for block_type in self.up_block_types):
            raise ValueError("VAE decoder supports only UpDecoderBlock2D")
        if len(self.down_block_types) != len(self.block_out_channels):
            raise ValueError("down_block_types length must match block_out_channels")
        if self.scaling_factor == 0:
            raise ValueError("scaling_factor must be non-zero")


class VaeResnetBlock2D(_MlxModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        groups: int,
        eps: float = 1e-6,
        output_scale_factor: float = 1.0,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.output_scale_factor = output_scale_factor
        self.norm1 = nn.GroupNorm(
            groups,
            in_channels,
            eps=eps,
            pytorch_compatible=True,
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(
            groups,
            out_channels,
            eps=eps,
            pytorch_compatible=True,
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def __call__(self, input_tensor: Any) -> Any:
        hidden_states = self.norm1(input_tensor)
        hidden_states = _silu(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = _silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        return (input_tensor + hidden_states) / self.output_scale_factor


class VaeAttentionBlock(_MlxModuleBase):
    def __init__(
        self,
        channels: int,
        *,
        groups: int,
        eps: float = 1e-6,
        output_scale_factor: float = 1.0,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.heads = 1
        self.head_dim = channels
        self.scale = self.head_dim**-0.5
        self.output_scale_factor = output_scale_factor
        self.group_norm = nn.GroupNorm(
            groups,
            channels,
            eps=eps,
            pytorch_compatible=True,
        )
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_k = nn.Linear(channels, channels, bias=True)
        self.to_v = nn.Linear(channels, channels, bias=True)
        self.to_out = [nn.Linear(channels, channels, bias=True)]

    def __call__(self, hidden_states: Any) -> Any:
        residual = hidden_states
        batch_size, height, width, channels = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, height * width, channels)
        hidden_states = self.group_norm(hidden_states)

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.reshape(batch_size, -1, self.heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        key = key.reshape(batch_size, -1, self.heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        value = value.reshape(batch_size, -1, self.heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        hidden_states = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
        )
        hidden_states = hidden_states.transpose(0, 2, 1, 3).reshape(
            batch_size,
            height * width,
            channels,
        )
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = hidden_states.reshape(batch_size, height, width, channels)
        return (hidden_states + residual) / self.output_scale_factor


class VaeUNetMidBlock2D(_MlxModuleBase):
    def __init__(self, channels: int, *, groups: int) -> None:
        _require_mlx()
        super().__init__()
        self.resnets = [
            VaeResnetBlock2D(channels, channels, groups=groups),
            VaeResnetBlock2D(channels, channels, groups=groups),
        ]
        self.attentions = [VaeAttentionBlock(channels, groups=groups)]

    def __call__(self, hidden_states: Any) -> Any:
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = self.attentions[0](hidden_states)
        return self.resnets[1](hidden_states)


class VaeUpsample2D(_MlxModuleBase):
    def __init__(self, channels: int) -> None:
        _require_mlx()
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def __call__(self, hidden_states: Any) -> Any:
        hidden_states = mx.repeat(hidden_states, 2, axis=1)
        hidden_states = mx.repeat(hidden_states, 2, axis=2)
        return self.conv(hidden_states)


class VaeUpDecoderBlock2D(_MlxModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int,
        groups: int,
        add_upsample: bool,
    ) -> None:
        _require_mlx()
        super().__init__()
        self.resnets = [
            VaeResnetBlock2D(
                in_channels if index == 0 else out_channels,
                out_channels,
                groups=groups,
            )
            for index in range(num_layers)
        ]
        self.upsamplers = [VaeUpsample2D(out_channels)] if add_upsample else None

    def __call__(self, hidden_states: Any) -> Any:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class DiffusersStyleDecoder(_MlxModuleBase):
    def __init__(self, config: AutoencoderKLDecoderConfig) -> None:
        _require_mlx()
        super().__init__()
        self.config = config
        decoder_channels = config.decoder_channels
        self.conv_in = nn.Conv2d(
            config.latent_channels,
            decoder_channels[0],
            kernel_size=3,
            padding=1,
        )
        self.mid_block = VaeUNetMidBlock2D(
            decoder_channels[0],
            groups=config.norm_num_groups,
        )
        self.up_blocks = []
        for index, output_channel in enumerate(decoder_channels):
            input_channel = decoder_channels[index - 1] if index > 0 else output_channel
            is_final_block = index == len(decoder_channels) - 1
            self.up_blocks.append(
                VaeUpDecoderBlock2D(
                    input_channel,
                    output_channel,
                    num_layers=config.layers_per_block + 1,
                    groups=config.norm_num_groups,
                    add_upsample=not is_final_block,
                )
            )
        self.conv_norm_out = nn.GroupNorm(
            config.norm_num_groups,
            config.block_out_channels[0],
            eps=1e-6,
            pytorch_compatible=True,
        )
        self.conv_out = nn.Conv2d(
            config.block_out_channels[0],
            config.out_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, sample: Any) -> Any:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample)
        for up_block in self.up_blocks:
            sample = up_block(sample)
        sample = self.conv_norm_out(sample)
        sample = _silu(sample)
        return self.conv_out(sample)


class AutoencoderKLDecoder(_MlxModuleBase):
    def __init__(self, config: AutoencoderKLDecoderConfig | None = None) -> None:
        if nn is not None:
            super().__init__()
        self.config = config or AutoencoderKLDecoderConfig()
        self.config.validate()
        if nn is None:
            return
        self.decoder = DiffusersStyleDecoder(self.config)

    @classmethod
    def from_pretrained(cls, artifact_dir: str | Path) -> "AutoencoderKLDecoder":
        _require_mlx()
        root = Path(artifact_dir).expanduser()
        artifact = read_artifact(root, "VAE")

        vae_config = read_json(component_config_path(root, artifact, "vae"), "VAE")
        config = AutoencoderKLDecoderConfig.from_dict(vae_config)
        model = cls(config)

        selected_weights = load_indexed_component_weights(
            root=root,
            artifact=artifact,
            component="vae",
            expected_shapes=_flatten_parameter_shapes(model.parameters()),
            mx_module=mx,
            artifact_label="VAE",
            weight_label="VAE",
            weight_transform=lambda key, array: _transform_loaded_weight(
                key,
                array,
                force_upcast=config.force_upcast,
            ),
        )

        model.load_weights(sorted(selected_weights.items()), strict=True)
        mx.eval(*selected_weights.values())
        return model

    def prepare_latents_for_decode(self, latents: Any) -> Any:
        _require_mlx()
        return latents / self.config.scaling_factor + self.config.shift_factor

    def decode(self, latents: Any, *, apply_scaling: bool = True) -> Any:
        _require_mlx()
        if latents.ndim != 4:
            raise ValueError("latents must be shaped [B, C, H, W]")
        if latents.shape[1] != self.config.latent_channels:
            raise ValueError(
                f"latents channel count {latents.shape[1]} does not match config "
                f"latent_channels {self.config.latent_channels}"
            )
        if apply_scaling:
            latents = self.prepare_latents_for_decode(latents)
        if self.config.force_upcast:
            latents = latents.astype(mx.float32)

        sample = latents.transpose(0, 2, 3, 1)
        sample = self.decoder(sample)
        return sample.transpose(0, 3, 1, 2)

    def encode(self, sample: Any) -> Any:
        raise ComponentNotImplementedError(
            "This text-to-image runtime implements VAE decode only."
        )


def _require_mlx() -> None:
    if mx is None or nn is None:
        raise BooguTurboMlxError(
            "The Boogu VAE decoder runtime requires MLX. Install "
            "`boogu-turbo-mlx[runtime]` on an MLX-supported machine."
        )


def _silu(x: Any) -> Any:
    return x * mx.sigmoid(x)


def _tuple_ints(value: Iterable[Any], name: str) -> tuple[int, ...]:
    items = tuple(int(item) for item in value)
    if not items:
        raise ValueError(f"{name} must not be empty")
    return items


def _tuple_strs(value: Iterable[Any], name: str) -> tuple[str, ...]:
    items = tuple(str(item) for item in value)
    if not items:
        raise ValueError(f"{name} must not be empty")
    return items


def _transform_loaded_weight(key: str, array: Any, *, force_upcast: bool) -> Any:
    if force_upcast:
        array = array.astype(mx.float32)
    if key.endswith(".weight") and array.ndim == 4:
        return array.transpose(0, 2, 3, 1)
    return array
