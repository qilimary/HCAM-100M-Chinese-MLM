# -*- coding: utf-8 -*-
"""HCAM 100M inference/model definition.

HCAM = Hybrid Convolution-Attention Model.
This file preserves the published checkpoint tensor names and shapes while
using the public HCAM naming in the Python API.
"""
from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HCAMConfig:
    vocab_size: int = 18000
    seq_len: int = 600
    d_model: int = 1056
    n_layers: int = 10
    n_heads: int = 16
    n_kv_heads: int = 4
    attention_layers: Tuple[int, ...] = (3, 7, 10)
    headwise_attention_gate: bool = True
    local_expand: float = 1.50
    ffn_mult: float = 2.75
    dropout: float = 0.10

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def local_inner(self) -> int:
        return math.ceil(self.d_model * self.local_expand / 32) * 32

    @property
    def ffn_hidden(self) -> int:
        return math.ceil(self.d_model * self.ffn_mult / 64) * 64

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if len(set(self.attention_layers)) != len(self.attention_layers):
            raise ValueError("attention_layers contains duplicates")
        if min(self.attention_layers) < 1 or max(self.attention_layers) > self.n_layers:
            raise ValueError("attention_layers out of range")


def rope_cos_sin(dim: int, length: int, theta: float = 10000.0, device=None):
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    pos = torch.arange(length, dtype=torch.float32, device=device)
    angles = torch.outer(pos, inv)
    angles = torch.stack((angles, angles), dim=-1).flatten(-2)
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.view(1, cos.shape[0], 1, cos.shape[1]).to(dtype=q.dtype)
    sin = sin.view(1, sin.shape[0], 1, sin.shape[1]).to(dtype=q.dtype)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(dtype=x.dtype)
            return F.rms_norm(x, (x.shape[-1],), weight, eps=1e-6).to(x.dtype)
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        return (y * self.weight.float()).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GatedDilatedConv(nn.Module):
    """Bidirectional depthwise dilated convolution with two multiplicative gates."""
    def __init__(self, d: int, kernel: int, dilation: int, dropout: float, local_expand: float = 1.50):
        super().__init__()
        history = dilation * (kernel - 1)
        if history % 2:
            raise ValueError("dilation*(kernel-1) must be even for symmetric padding")
        self.symmetric_padding = history // 2
        inner = math.ceil(d * local_expand / 32) * 32
        self.norm = RMSNorm(d)
        self.in_proj = nn.Linear(d, inner * 3, bias=False)
        self.conv = nn.Conv1d(inner, inner, kernel, dilation=dilation, groups=inner)
        self.out_proj = nn.Linear(inner, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        content, activation_gate, modulation_gate = self.in_proj(self.norm(x)).chunk(3, dim=-1)
        content = content.transpose(1, 2)
        conv = F.conv1d(
            content, self.conv.weight, self.conv.bias,
            padding=self.symmetric_padding, dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        conv = conv.transpose(1, 2)
        mixed = F.silu(conv) * F.silu(activation_gate) * torch.sigmoid(modulation_gate)
        return residual + self.drop(self.out_proj(mixed))


class GatedGQA(nn.Module):
    """Bidirectional grouped-query attention + head-wise output gate + SwiGLU FFN."""
    def __init__(self, d: int, heads: int, kv_heads: int, dropout: float, ffn_mult: float = 2.75,
                 headwise_attention_gate: bool = True):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.hd = d // heads
        self.q_dim = d
        self.kv_dim = kv_heads * self.hd
        self.norm = RMSNorm(d)
        self.use_output_gate = bool(headwise_attention_gate)
        gate_dim = heads if self.use_output_gate else 0
        self.qkv = nn.Linear(d, self.q_dim + 2 * self.kv_dim + gate_dim, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.qn = RMSNorm(self.hd)
        self.kn = RMSNorm(self.hd)
        hidden = math.ceil(d * ffn_mult / 64) * 64
        self.ffn_norm = RMSNorm(d)
        self.ffn = SwiGLU(d, hidden)
        self.drop = nn.Dropout(dropout)
        try:
            self._sdpa_has_gqa = "enable_gqa" in inspect.signature(F.scaled_dot_product_attention).parameters
        except (TypeError, ValueError):
            self._sdpa_has_gqa = "enable_gqa" in (F.scaled_dot_product_attention.__doc__ or "")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        residual = x
        projected = self.qkv(self.norm(x))
        if self.use_output_gate:
            q, k, v, gate = projected.split((self.q_dim, self.kv_dim, self.kv_dim, self.heads), dim=-1)
        else:
            q, k, v = projected.split((self.q_dim, self.kv_dim, self.kv_dim), dim=-1)
            gate = None
        q = q.view(b, t, self.heads, self.hd)
        k = k.view(b, t, self.kv_heads, self.hd)
        v = v.view(b, t, self.kv_heads, self.hd)
        q, k = apply_rope(self.qn(q), self.kn(k), cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        dropout_p = self.drop.p if self.training else 0.0

        # Native GQA is used when the local PyTorch build supports it. The fallback
        # repeats K/V heads and is mathematically equivalent for inference.
        if self._sdpa_has_gqa and self.heads != self.kv_heads:
            try:
                attended = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=dropout_p,
                    is_causal=False, enable_gqa=True,
                )
            except RuntimeError:
                repeat = self.heads // self.kv_heads
                attended = F.scaled_dot_product_attention(
                    q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1),
                    attn_mask=None, dropout_p=dropout_p, is_causal=False,
                )
        else:
            repeat = self.heads // self.kv_heads
            if repeat > 1:
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False)

        attended = attended.transpose(1, 2)
        if gate is not None:
            attended = attended * torch.sigmoid(gate).unsqueeze(-1)
        attended = attended.contiguous().view(b, t, c)
        x = residual + self.drop(self.out_proj(attended))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class HCAM(nn.Module):
    def __init__(self, config: HCAMConfig | None = None):
        super().__init__()
        self.config = config or HCAMConfig()
        self.config.validate()
        c = self.config
        self.vocab_size = c.vocab_size
        self.d = c.d_model
        self.heads = c.n_heads
        self.embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.mask_embedding = nn.Parameter(torch.empty(c.d_model))
        self.emb_drop = nn.Dropout(c.dropout)
        cos, sin = rope_cos_sin(c.head_dim, c.seq_len * 2)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.layers = nn.ModuleList()
        self.types: List[str] = []
        specs = [(3, 1), (3, 2), (5, 2)]
        local_index = 0
        attention_layers = set(c.attention_layers)
        for index in range(c.n_layers):
            if index + 1 in attention_layers:
                self.layers.append(GatedGQA(c.d_model, c.n_heads, c.n_kv_heads, c.dropout, c.ffn_mult,
                                            c.headwise_attention_gate))
                self.types.append("gqa")
            else:
                kernel, dilation = specs[local_index % len(specs)]
                local_index += 1
                self.layers.append(GatedDilatedConv(c.d_model, kernel, dilation, c.dropout, c.local_expand))
                self.types.append("conv")
        self.final_norm = RMSNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self._init_weights()
        self.lm_head.weight = self.embedding.weight

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
                nn.init.normal_(module.weight, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.mask_embedding, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for layer in self.layers:
            nn.init.normal_(layer.out_proj.weight, std=residual_std)
            if isinstance(layer, GatedGQA):
                nn.init.normal_(layer.ffn.w3.weight, std=residual_std)

    def _frequencies(self, length: int):
        if length > len(self.rope_cos):
            cos, sin = rope_cos_sin(
                self.d // self.heads,
                max(length, len(self.rope_cos) * 2),
                device=self.rope_cos.device,
            )
            self.rope_cos, self.rope_sin = cos, sin
        return self.rope_cos[:length], self.rope_sin[:length]

    def forward(self, input_ids: torch.Tensor, mask_embedding_positions: Optional[torch.Tensor] = None,
                prediction_indices: Optional[torch.Tensor] = None, return_hidden: bool = False):
        hidden = self.embedding(input_ids)
        if mask_embedding_positions is not None:
            if mask_embedding_positions.shape != input_ids.shape:
                raise ValueError("mask_embedding_positions must match input_ids shape")
            replacement = self.mask_embedding.to(hidden.dtype).view(1, 1, -1)
            hidden = torch.where(mask_embedding_positions.unsqueeze(-1), replacement, hidden)
        hidden = self.emb_drop(hidden)
        cos, sin = self._frequencies(input_ids.shape[1])
        for layer, layer_type in zip(self.layers, self.types):
            hidden = layer(hidden, cos, sin) if layer_type == "gqa" else layer(hidden)
        hidden = self.final_norm(hidden)
        if return_hidden:
            return hidden
        if prediction_indices is not None:
            if prediction_indices.ndim != 2 or prediction_indices.shape[0] != input_ids.shape[0]:
                raise ValueError("prediction_indices must have shape [B, K]")
            gather_index = prediction_indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            hidden = hidden.gather(1, gather_index)
        return self.lm_head(hidden)


class HCAMTokenizer:
    """Runtime character tokenizer. The file is JSON despite the historic .model suffix."""
    def __init__(self, path: str | Path):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        pieces = obj.get("pieces")
        if not isinstance(pieces, list) or len(pieces) != 18000:
            raise ValueError("tokenizer must contain exactly 18,000 pieces")
        self.pieces = pieces
        self.vocab = {token: i for i, token in enumerate(pieces)}
        self.unk = self.vocab.get("<unk>", 0)
        self.pad = self.vocab.get("<pad>", 1)
        self.bos = self.vocab.get("<bos>", 2)
        self.eos = self.vocab.get("<eos>", 3)

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [self.vocab.get(ch, self.unk) for ch in text]
        return ([self.bos] + ids + [self.eos]) if add_special_tokens else ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        specials = {"<unk>", "<pad>", "<bos>", "<eos>", "<用户>", "<助手>", "<换行>", "<段落>"}
        out = []
        for idx in ids:
            token = self.pieces[int(idx)] if 0 <= int(idx) < len(self.pieces) else "<unk>"
            if skip_special_tokens and token in specials:
                continue
            out.append(token)
        return "".join(out)


def _config_from_checkpoint(payload: dict) -> HCAMConfig:
    cfg = HCAMConfig()
    inf = payload.get("inference_config") or {}
    mapping = {
        "vocab_size": "vocab_size", "seq_len": "seq_len", "d_model": "d_model",
        "n_layers": "n_layers", "n_heads": "n_heads", "n_kv_heads": "n_kv_heads",
        "attention_layers": "attention_layers", "headwise_attention_gate": "headwise_attention_gate",
        "local_expand": "local_expand", "ffn_mult": "ffn_mult", "dropout": "dropout",
    }
    data = asdict(cfg)
    for old, new in mapping.items():
        if old in inf:
            value = inf[old]
            if new == "attention_layers":
                value = tuple(value)
            data[new] = value
    return HCAMConfig(**data)


def load_hcam(checkpoint_path: str | Path, device: str | torch.device = "cpu", dtype=None):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a dict")
    state = payload.get("model", payload)
    config = _config_from_checkpoint(payload)
    model = HCAM(config)
    result = model.load_state_dict(state, strict=True)
    model.eval().to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    return model, payload, result


@torch.inference_mode()
def masked_logits(model: HCAM, input_ids: torch.Tensor, positions: torch.Tensor):
    """Predict selected positions after replacing them with the learned mask embedding.

    input_ids: [B, L]
    positions: [B, K] token positions to mask and classify.
    returns: [B, K, vocab_size]
    """
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask.scatter_(1, positions, True)
    return model(input_ids, mask_embedding_positions=mask, prediction_indices=positions)
