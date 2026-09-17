# -*- coding: utf-8 -*-
"""
HCAM 100M public pretraining reference implementation.

This is a cleaned, standalone version of the training recipe used for HCAM 100M.
It keeps the released architecture, dynamic multi-granularity MLM objective,
10-epoch masking schedule, AdamW settings, token-based LR schedule, AMP/DDP,
gradient accumulation, validation, and checkpoint/export format.

Important reproducibility note:
The original frozen 1.638B-character raw-corpus snapshot is not redistributed.
Therefore this script reproduces the training METHOD/RECIPE, not the exact byte-
for-byte released checkpoint unless the same frozen corpus/token streams are used.

Expected repository files:
  hcam.py
  tokenizer.model
  mask_guide_tokenizer.model   # optional but recommended for whole-piece/span masking

Example:
  python train_pretrain.py prepare --input data/train --output data/train_stream \
      --tokenizer tokenizer.model --mask-guide-tokenizer mask_guide_tokenizer.model

  python train_pretrain.py prepare --input data/val --output data/val_stream \
      --tokenizer tokenizer.model --mask-guide-tokenizer mask_guide_tokenizer.model

  torchrun --standalone --nproc_per_node=2 train_pretrain.py train \
      --train-prefix data/train_stream --val-prefix data/val_stream \
      --tokenizer tokenizer.model --output-dir runs/hcam100m

Single GPU:
  python train_pretrain.py train --train-prefix data/train_stream \
      --val-prefix data/val_stream --tokenizer tokenizer.model \
      --output-dir runs/hcam100m --micro-batch 8
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler

from hcam import HCAM, HCAMConfig, HCAMTokenizer

try:
    import sentencepiece as spm
except ImportError:
    spm = None


# -------------------------------------------------------------------------------------------------
# Released training recipe
# -------------------------------------------------------------------------------------------------

@dataclass
class TrainConfig:
    seed: int = 42
    seq_len: int = 600
    virtual_epochs: int = 10
    effective_global_batch: int = 384
    peak_lr: float = 7e-4
    min_lr_ratio: float = 0.08
    warmup_ratio: float = 0.04
    weight_decay: float = 0.08
    z_loss_weight: float = 1e-4
    grad_clip: float = 1.0
    dropout: float = 0.10

    mask_ratio_start: float = 0.30
    mask_ratio_end: float = 0.15
    validation_mask_ratio: float = 0.15
    validation_seed: int = 20260726

    mask_probability: float = 0.80
    random_probability: float = 0.10
    keep_probability: float = 0.10

    # character / whole SentencePiece group / adjacent-group span
    mask_strategy_1_3: Tuple[float, float, float] = (0.50, 0.375, 0.125)
    mask_strategy_4_6: Tuple[float, float, float] = (0.45, 0.4125, 0.1375)
    mask_strategy_7_9: Tuple[float, float, float] = (0.40, 0.45, 0.15)
    mask_strategy_10: Tuple[float, float, float] = (0.25, 0.50, 0.25)
    adjacent_kernel: int = 3
    final_adjacent_kernel: int = 5
    max_guide_piece_chars: int = 8

    # Original epoch 10 used an LR floor of 1e-4.
    final_epoch_lr_floor: float = 1e-4


SPECIAL_NAMES = ("<unk>", "<pad>", "<bos>", "<eos>", "<用户>", "<助手>", "<换行>", "<段落>")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_distributed():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return distributed, rank, world, local_rank, device


def rank0_print(rank: int, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)


def barrier(distributed: bool):
    if distributed:
        dist.barrier()


def seed_everything(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed((seed + rank) % (2**32 - 1))
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


# -------------------------------------------------------------------------------------------------
# Text -> character stream + SentencePiece boundary-guide stream
# -------------------------------------------------------------------------------------------------

def iter_input_files(inputs: Sequence[str]) -> Iterator[Path]:
    allowed = {".txt", ".text", ".jsonl", ".json"}
    found = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found.extend(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in allowed)
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(item)
    for p in sorted(set(found)):
        yield p


def extract_json_text(obj) -> Iterable[str]:
    if isinstance(obj, str):
        if obj.strip():
            yield obj
    elif isinstance(obj, list):
        for x in obj:
            yield from extract_json_text(x)
    elif isinstance(obj, dict):
        # Common raw-corpus fields first; otherwise recursively inspect values.
        for key in ("text", "content", "document", "body", "sentence"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                yield value
                return
        for value in obj.values():
            yield from extract_json_text(value)


def iter_documents(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text"}:
        # Blank-line separated documents while preserving single line breaks.
        buf = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    buf.append(line.rstrip("\n"))
                elif buf:
                    text = "\n".join(buf).strip()
                    if text:
                        yield text
                    buf.clear()
            if buf:
                text = "\n".join(buf).strip()
                if text:
                    yield text
        return

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield from extract_json_text(obj)
        return

    if suffix == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return
        yield from extract_json_text(obj)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class PublicTokenizer:
    def __init__(self, tokenizer_path: str | Path):
        self.base = HCAMTokenizer(tokenizer_path)
        self.vocab = self.base.vocab
        self.vocab_size = self.base.vocab_size
        self.unk = self.base.unk
        self.pad = self.base.pad
        self.special_ids = tuple(self.vocab.get(x, -1) for x in SPECIAL_NAMES)
        self.special_ids = tuple(x for x in self.special_ids if x >= 0)
        self.newline = self.vocab.get("<换行>", self.unk)
        self.paragraph = self.vocab.get("<段落>", self.unk)

    def encode_text(self, text: str) -> list[int]:
        # Preserve the released special newline/paragraph tokens.
        text = normalize_text(text)
        out: list[int] = []
        parts = re.split(r"(\n\n|\n)", text)
        for part in parts:
            if part == "\n\n":
                out.append(self.paragraph)
            elif part == "\n":
                out.append(self.newline)
            elif part:
                out.extend(self.vocab.get(ch, self.unk) for ch in part)
        return out


class SPBoundaryGuide:
    """Produces one boolean start flag per character token.

    SentencePiece IDs never become HCAM input IDs.  The guide only marks group
    boundaries used by whole-piece and adjacent-piece masking.
    """
    def __init__(self, path: Optional[str | Path]):
        self.sp = None
        if path:
            if spm is None:
                raise RuntimeError("sentencepiece is required when --mask-guide-tokenizer is used")
            self.sp = spm.SentencePieceProcessor(model_file=str(path))

    def starts_for_plain_text(self, text: str) -> list[bool]:
        if not text:
            return []
        if self.sp is None:
            return [True] * len(text)

        # immutable_proto exposes piece begin/end offsets in current SentencePiece builds.
        try:
            proto = self.sp.encode(text, out_type="immutable_proto")
            starts = [False] * len(text)
            starts[0] = True
            for piece in proto.pieces:
                begin = int(piece.begin)
                if 0 <= begin < len(starts):
                    starts[begin] = True
            # Reject clearly incompatible offsets/normalization and fall back safely.
            if sum(starts) >= 1:
                return starts
        except Exception:
            pass

        # Conservative fallback: align piece surfaces after removing SentencePiece word marker.
        starts = [False] * len(text)
        starts[0] = True
        cursor = 0
        for piece in self.sp.encode(text, out_type=str):
            surface = piece.replace("▁", " ")
            surface = surface if surface else piece
            probe = surface.strip()
            if not probe:
                continue
            idx = text.find(probe, cursor)
            if idx < 0:
                return [True] * len(text)
            starts[idx] = True
            cursor = idx + len(probe)
        return starts

    def starts_for_document(self, text: str) -> tuple[list[str], list[bool]]:
        text = normalize_text(text)
        tokens: list[str] = []
        starts: list[bool] = []
        parts = re.split(r"(\n\n|\n)", text)
        for part in parts:
            if part == "\n\n":
                tokens.append("<段落>"); starts.append(True)
            elif part == "\n":
                tokens.append("<换行>"); starts.append(True)
            elif part:
                local = self.starts_for_plain_text(part)
                tokens.extend(list(part)); starts.extend(local)
        return tokens, starts


def prepare_stream(args):
    tok = PublicTokenizer(args.tokenizer)
    guide = SPBoundaryGuide(args.mask_guide_tokenizer)
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    token_tmp = prefix.with_suffix(".tokens.bin.tmp")
    guide_tmp = prefix.with_suffix(".guide.bin.tmp")
    token_out = token_tmp.open("wb")
    guide_out = guide_tmp.open("wb")

    total_tokens = total_docs = 0
    min_chars = max(1, args.min_chars)
    max_chars = max(min_chars, args.max_chars)

    def write_id(token_id: int, is_start: bool):
        nonlocal total_tokens
        np.asarray([token_id], dtype=np.int32).tofile(token_out)
        np.asarray([1 if is_start else 0], dtype=np.uint8).tofile(guide_out)
        total_tokens += 1

    try:
        for file in iter_input_files(args.input):
            print(f"[prepare] {file}", flush=True)
            for text in iter_documents(file):
                text = normalize_text(text)
                if len(text) < min_chars:
                    continue
                if len(text) > max_chars:
                    text = text[:max_chars]
                symbols, starts = guide.starts_for_document(text)
                ids = []
                for s in symbols:
                    if s == "<换行>":
                        ids.append(tok.newline)
                    elif s == "<段落>":
                        ids.append(tok.paragraph)
                    else:
                        ids.append(tok.vocab.get(s, tok.unk))
                if len(ids) != len(starts):
                    raise RuntimeError("token/guide alignment failure")
                np.asarray(ids, dtype=np.int32).tofile(token_out)
                np.asarray(starts, dtype=np.uint8).tofile(guide_out)
                # Explicit document separator.
                write_id(tok.paragraph, True)
                total_docs += 1
                if args.max_tokens and total_tokens >= args.max_tokens:
                    break
            if args.max_tokens and total_tokens >= args.max_tokens:
                break
    finally:
        token_out.close(); guide_out.close()

    token_path = prefix.with_suffix(".tokens.bin")
    guide_path = prefix.with_suffix(".guide.bin")
    token_tmp.replace(token_path); guide_tmp.replace(guide_path)
    meta = {
        "format": "HCAM-public-char-guide-stream-v1",
        "documents": total_docs,
        "tokens": total_tokens,
        "token_dtype": "int32",
        "guide_dtype": "uint8",
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "mask_guide_tokenizer_sha256": sha256_file(args.mask_guide_tokenizer) if args.mask_guide_tokenizer else None,
    }
    prefix.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


# -------------------------------------------------------------------------------------------------
# Memory-mapped sequence dataset
# -------------------------------------------------------------------------------------------------

class StreamDataset(Dataset):
    def __init__(self, prefix: str | Path, seq_len: int, offset: int = 0, max_sequences: int = 0):
        prefix = Path(prefix)
        self.tokens = np.memmap(prefix.with_suffix(".tokens.bin"), mode="r", dtype=np.int32)
        self.guide = np.memmap(prefix.with_suffix(".guide.bin"), mode="r", dtype=np.uint8)
        if len(self.tokens) != len(self.guide):
            raise ValueError("token and guide streams have different lengths")
        self.seq_len = int(seq_len)
        self.offset = int(offset)
        usable = max(0, len(self.tokens) - self.offset)
        self.count = usable // self.seq_len
        if max_sequences > 0:
            self.count = min(self.count, max_sequences)

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        start = self.offset + idx * self.seq_len
        end = start + self.seq_len
        x = torch.from_numpy(np.asarray(self.tokens[start:end], dtype=np.int64).copy())
        g = torch.from_numpy(np.asarray(self.guide[start:end], dtype=np.bool_).copy())
        if len(x) != self.seq_len:
            raise IndexError(idx)
        g[0] = True
        return x, g


class DistributedSequentialSampler(Sampler[int]):
    def __init__(self, n: int, rank: int, world: int, shuffle: bool, seed: int, epoch: int):
        self.n, self.rank, self.world = n, rank, world
        self.shuffle, self.seed, self.epoch = shuffle, seed, epoch

    def __iter__(self):
        indices = np.arange(self.n, dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
            rng.shuffle(indices)
        return iter(indices[self.rank::self.world].tolist())

    def __len__(self):
        return (self.n - self.rank + self.world - 1) // self.world


# -------------------------------------------------------------------------------------------------
# Dynamic multi-granularity masking
# -------------------------------------------------------------------------------------------------

_RANDOM_CANDIDATE_CACHE = {}


def strategy_for_epoch(cfg: TrainConfig, epoch_number: int):
    if epoch_number <= 3: return cfg.mask_strategy_1_3
    if epoch_number <= 6: return cfg.mask_strategy_4_6
    if epoch_number <= 9: return cfg.mask_strategy_7_9
    return cfg.mask_strategy_10


def mask_ratio_for_epoch(cfg: TrainConfig, epoch_number: int):
    if cfg.virtual_epochs <= 1:
        return cfg.mask_ratio_end
    progress = (epoch_number - 1) / (cfg.virtual_epochs - 1)
    return cfg.mask_ratio_start + (cfg.mask_ratio_end - cfg.mask_ratio_start) * progress


def training_generator(device, cfg: TrainConfig, epoch_number: int, step_in_epoch: int, rank: int):
    modulus = (1 << 63) - 1
    seed = (cfg.seed * 1_000_003 + epoch_number * 1_000_000_007 + step_in_epoch * 97_409 + rank * 65_537) % modulus
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def random_replacement_candidates(tok: PublicTokenizer, device):
    key = (tok.vocab_size, str(device))
    if key not in _RANDOM_CANDIDATE_CACHE:
        excluded = set(tok.special_ids)
        values = [i for i in range(tok.vocab_size) if i not in excluded]
        _RANDOM_CANDIDATE_CACHE[key] = torch.tensor(values, dtype=torch.long, device=device)
    return _RANDOM_CANDIDATE_CACHE[key]


def correlated_scores(base: torch.Tensor, kernel: int):
    return F.avg_pool1d(base.unsqueeze(1), kernel_size=kernel, stride=1, padding=kernel // 2).squeeze(1)


def dynamic_mask(clean_ids: torch.Tensor, guide_starts: torch.Tensor, tok: PublicTokenizer,
                 cfg: TrainConfig, ratio: float, generator: torch.Generator,
                 strategy: Tuple[float, float, float], adjacent_kernel: int):
    if clean_ids.ndim != 2 or clean_ids.shape[1] != cfg.seq_len:
        raise ValueError("unexpected MLM batch shape")
    batch, length = clean_ids.shape
    target_count = max(1, min(length - 1, int(round(length * ratio))))
    max_count = min(length, target_count + cfg.max_guide_piece_chars - 1)

    special = torch.zeros_like(clean_ids, dtype=torch.bool)
    for token_id in tok.special_ids:
        special |= clean_ids.eq(int(token_id))
    valid_char = ~special

    starts = guide_starts.bool().clone()
    starts[:, 0] = True
    starts |= special
    group_ids = (starts.long().cumsum(dim=1) - 1).clamp(min=0, max=length - 1)
    group_lengths = torch.zeros((batch, length), device=clean_ids.device, dtype=torch.long)
    group_lengths.scatter_add_(1, group_ids, valid_char.long())

    group_random = torch.rand((batch, length), device=clean_ids.device, generator=generator)
    adjacent_random = correlated_scores(group_random, adjacent_kernel)
    p_char, p_whole, p_adjacent = map(float, strategy)
    if min(strategy) < 0 or abs(p_char + p_whole + p_adjacent - 1.0) > 1e-6:
        raise ValueError("mask strategy probabilities must sum to 1")

    selector = torch.rand((batch, 1), device=clean_ids.device, generator=generator)
    char_rows = selector < p_char
    adjacent_rows = selector >= (p_char + p_whole)
    group_scores = torch.where(adjacent_rows, adjacent_random, group_random)
    group_scores = group_scores.masked_fill(group_lengths.eq(0), float("-inf"))
    order = group_scores.argsort(dim=1, descending=True)
    ordered_lengths = group_lengths.gather(1, order)
    cumulative = ordered_lengths.cumsum(dim=1)
    take = (cumulative - ordered_lengths < target_count) & ordered_lengths.gt(0)
    group_selected = torch.zeros_like(group_lengths, dtype=torch.bool)
    group_selected.scatter_(1, order, take)
    guided_selected = group_selected.gather(1, group_ids) & valid_char

    char_scores = torch.rand((batch, length), device=clean_ids.device, generator=generator)
    char_scores = char_scores.masked_fill(special, float("-inf"))
    char_indices = char_scores.topk(target_count, dim=1, sorted=False).indices
    char_selected = torch.zeros_like(special)
    char_selected.scatter_(1, char_indices, True)
    selected = torch.where(char_rows, char_selected, guided_selected)

    selection_scores = torch.rand((batch, length), device=clean_ids.device, generator=generator)
    selection_scores = selection_scores.masked_fill(~selected, float("-inf"))
    prediction_indices = selection_scores.topk(max_count, dim=1, sorted=False).indices
    prediction_valid = selected.gather(1, prediction_indices)
    targets = clean_ids.gather(1, prediction_indices)

    # Whole-piece/span rows share one 80/10/10 draw per SentencePiece group.
    char_draw = torch.rand((batch, length), device=clean_ids.device, generator=generator)
    group_draw_table = torch.rand((batch, length), device=clean_ids.device, generator=generator)
    group_draw = group_draw_table.gather(1, group_ids)
    replacement_draw = torch.where(char_rows, char_draw, group_draw)

    mask_positions = selected & (replacement_draw < cfg.mask_probability)
    random_upper = cfg.mask_probability + cfg.random_probability
    random_positions = selected & (replacement_draw >= cfg.mask_probability) & (replacement_draw < random_upper)
    candidates = random_replacement_candidates(tok, clean_ids.device)
    random_ids = candidates[torch.randint(0, candidates.numel(), (batch, length), device=clean_ids.device, generator=generator)]
    corrupted = clean_ids.clone()
    corrupted[random_positions] = random_ids[random_positions]
    return corrupted, prediction_indices, mask_positions, targets, prediction_valid


def masked_loss(logits, targets, valid, z_weight: float):
    flat_valid = valid.reshape(-1)
    count = flat_valid.sum().clamp_min(1)
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    ce = losses.masked_select(flat_valid).sum() / count
    log_z = torch.logsumexp(logits, dim=-1).float()
    z = log_z.square().masked_select(valid).sum() / count
    total = ce + z_weight * z
    correct = (logits.detach().argmax(-1).eq(targets) & valid).sum()
    return total, ce.detach(), z.detach(), correct.detach(), count.detach()


# -------------------------------------------------------------------------------------------------
# Optimizer, LR, evaluation, checkpoints
# -------------------------------------------------------------------------------------------------

def make_optimizer(model: HCAM, cfg: TrainConfig):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (no_decay if p.ndim < 2 or name.endswith("bias") or "norm" in name.lower() else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.peak_lr, betas=(0.9, 0.95), eps=1e-8,
        fused=torch.cuda.is_available(),
    )


def lr_for_progress(cfg: TrainConfig, exposed_tokens: int, total_tokens: int, epoch_number: int):
    warm = max(1, int(total_tokens * cfg.warmup_ratio))
    if exposed_tokens < warm:
        lr = cfg.peak_lr * exposed_tokens / warm
    else:
        progress = min(1.0, (exposed_tokens - warm) / max(1, total_tokens - warm))
        min_lr = cfg.peak_lr * cfg.min_lr_ratio
        lr = min_lr + 0.5 * (cfg.peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    if epoch_number >= 10:
        lr = max(lr, cfg.final_epoch_lr_floor)
    return lr


def set_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.inference_mode()
def evaluate(model, loader, tok, cfg: TrainConfig, device, distributed, rank):
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()
    sums = torch.zeros(3, dtype=torch.float64, device=device)  # ce_sum, correct, count
    for step, (clean, guide) in enumerate(loader, 1):
        clean, guide = clean.to(device, non_blocking=True), guide.to(device, non_blocking=True)
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg.validation_seed + step)
        corrupted, indices, mask_pos, targets, valid = dynamic_mask(
            clean, guide, tok, cfg, cfg.validation_mask_ratio, gen,
            cfg.mask_strategy_1_3, cfg.adjacent_kernel,
        )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = raw(corrupted, mask_embedding_positions=mask_pos, prediction_indices=indices)
            _, ce, _, correct, count = masked_loss(logits, targets, valid, cfg.z_loss_weight)
        sums[0] += ce.double() * count.double()
        sums[1] += correct.double()
        sums[2] += count.double()
    if distributed:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    count = max(1.0, float(sums[2].item()))
    raw.train()
    return float(sums[0].item()) / count, float(sums[1].item()) / count


def inference_config(model_cfg: HCAMConfig):
    return {
        "vocab_size": model_cfg.vocab_size,
        "seq_len": model_cfg.seq_len,
        "d_model": model_cfg.d_model,
        "n_layers": model_cfg.n_layers,
        "n_heads": model_cfg.n_heads,
        "n_kv_heads": model_cfg.n_kv_heads,
        "attention_layers": list(model_cfg.attention_layers),
        "headwise_attention_gate": model_cfg.headwise_attention_gate,
        "local_expand": model_cfg.local_expand,
        "ffn_mult": model_cfg.ffn_mult,
        "dropout": model_cfg.dropout,
    }


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_training_checkpoint(path: Path, raw_model: HCAM, optimizer, scaler, cfg: TrainConfig,
                             epoch: int, global_step: int, exposed_tokens: int,
                             best_val: float, tokenizer_sha256: str):
    payload = {
        "checkpoint_kind": "epoch_end",
        "architecture": "HCAM-BiMask",
        "architecture_revision": "bimask-v1",
        "objective": "dynamic_multi_granularity_masked_language_modeling",
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "exposed_tokens": exposed_tokens,
        "best": best_val,
        "vocab": raw_model.config.vocab_size,
        "tokenizer_sha256": tokenizer_sha256,
        "inference_config": inference_config(raw_model.config),
        "training_config": asdict(cfg),
    }
    atomic_torch_save(payload, path)


def save_final_model(path: Path, raw_model: HCAM, cfg: TrainConfig, epoch: int, global_step: int,
                     tokenizer_sha256: str, val_loss: float, val_acc: float, best_val: float):
    payload = {
        "checkpoint_kind": "final_model",
        "architecture": "HCAM-BiMask",
        "architecture_revision": "bimask-v1",
        "objective": "dynamic_multi_granularity_masked_language_modeling",
        "model": raw_model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "vocab": raw_model.config.vocab_size,
        "tokenizer_sha256": tokenizer_sha256,
        "inference_config": inference_config(raw_model.config),
        "final_validation_mlm_loss": val_loss,
        "final_validation_mask_accuracy": val_acc,
        "historical_best_validation_loss": best_val,
        "final_epoch_mask_strategy": list(cfg.mask_strategy_10),
        "final_epoch_adjacent_kernel": cfg.final_adjacent_kernel,
        "final_epoch_lr_floor": cfg.final_epoch_lr_floor,
        "recommended_lr_if_extending_pretraining": cfg.final_epoch_lr_floor,
    }
    atomic_torch_save(payload, path)


# -------------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------------

def train(args):
    distributed, rank, world, local_rank, device = setup_distributed()
    cfg = TrainConfig(
        seq_len=args.seq_len,
        virtual_epochs=args.epochs,
        effective_global_batch=args.global_batch,
        peak_lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )
    seed_everything(cfg.seed, rank)
    tok = PublicTokenizer(args.tokenizer)
    tokenizer_sha = sha256_file(args.tokenizer)

    model_cfg = HCAMConfig(seq_len=cfg.seq_len, dropout=cfg.dropout)
    raw_model = HCAM(model_cfg).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model"], strict=True)
    model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank) if distributed and device.type == "cuda" else (
        DDP(raw_model) if distributed else raw_model
    )

    optimizer = make_optimizer(raw_model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = global_step = exposed_tokens = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        exposed_tokens = int(ckpt.get("exposed_tokens", 0))
        best_val = float(ckpt.get("best", float("inf")))
        del ckpt

    per_update_physical = args.micro_batch * world
    if cfg.effective_global_batch % per_update_physical:
        raise ValueError(
            f"global batch {cfg.effective_global_batch} must be divisible by "
            f"micro_batch({args.micro_batch}) × world_size({world})"
        )
    accum = cfg.effective_global_batch // per_update_physical

    # First 5 virtual epochs use original alignment; last 5 use a half-window offset.
    base_train = StreamDataset(args.train_prefix, cfg.seq_len, offset=0, max_sequences=args.max_train_sequences)
    offset_train = StreamDataset(args.train_prefix, cfg.seq_len, offset=cfg.seq_len // 2,
                                 max_sequences=args.max_train_sequences)
    val_ds = StreamDataset(args.val_prefix, cfg.seq_len, offset=0, max_sequences=args.max_val_sequences)
    val_sampler = DistributedSequentialSampler(len(val_ds), rank, world, False, cfg.seed, 0)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch, sampler=val_sampler,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    # Planned training exposure determines token-based LR.  If --max-train-sequences is
    # used for a smoke run, the LR schedule scales to that reduced run.
    n_first = len(base_train)
    n_second = len(offset_train)
    total_exposure_tokens = cfg.seq_len * (min(5, cfg.virtual_epochs) * n_first +
        max(0, cfg.virtual_epochs - 5) * n_second)
    total_exposure_tokens = max(1, total_exposure_tokens)

    rank0_print(rank, f"HCAM parameters: {sum(p.numel() for p in raw_model.parameters()):,}")
    rank0_print(rank, f"device={device} world={world} micro_batch={args.micro_batch} accum={accum} "
                      f"effective_global_batch={cfg.effective_global_batch}")
    rank0_print(rank, f"planned exposure={total_exposure_tokens:,} character tokens")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    last_path, final_path = out / "last.pt", out / "model.pt"

    last_val_loss = last_val_acc = float("nan")
    for epoch_idx in range(start_epoch, cfg.virtual_epochs):
        epoch_num = epoch_idx + 1
        ds = base_train if epoch_num <= 5 else offset_train
        sampler = DistributedSequentialSampler(len(ds), rank, world, True, cfg.seed, epoch_idx)
        loader = DataLoader(ds, batch_size=args.micro_batch, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False)

        ratio = mask_ratio_for_epoch(cfg, epoch_num)
        strategy = strategy_for_epoch(cfg, epoch_num)
        kernel = cfg.final_adjacent_kernel if epoch_num >= 10 else cfg.adjacent_kernel
        raw_model.train()
        optimizer.zero_grad(set_to_none=True)

        running_ce = running_correct = running_count = 0.0
        t0 = time.time()
        loader_steps = len(loader)
        for step, (clean, guide) in enumerate(loader, 1):
            clean = clean.to(device, non_blocking=True)
            guide = guide.to(device, non_blocking=True)
            gen = training_generator(device, cfg, epoch_num, step, rank)
            corrupted, indices, mask_pos, targets, valid = dynamic_mask(
                clean, guide, tok, cfg, ratio, gen, strategy, kernel
            )

            group_start = ((step - 1) // accum) * accum
            group_end = min(group_start + accum, loader_steps)
            micro_in_group = group_end - group_start
            should_step = step == group_end
            sync_ctx = model.no_sync() if distributed and not should_step else contextlib.nullcontext()

            # Exposed tokens are counted globally across ranks, matching the released recipe.
            exposed_tokens += clean.numel() * world
            lr = lr_for_progress(cfg, exposed_tokens, total_exposure_tokens, epoch_num)
            set_lr(optimizer, lr)

            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(corrupted, mask_embedding_positions=mask_pos, prediction_indices=indices)
                    loss, ce, z, correct, count = masked_loss(logits, targets, valid, cfg.z_loss_weight)
                    loss = loss / micro_in_group
                scaler.scale(loss).backward()

            c = int(count.item())
            running_ce += float(ce.item()) * c
            running_correct += int(correct.item())
            running_count += c

            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if rank == 0 and (step % args.log_every == 0 or step == loader_steps):
                denom = max(1, running_count)
                speed = (step * args.micro_batch * cfg.seq_len) / max(1e-6, time.time() - t0)
                print(
                    f"E{epoch_num}/{cfg.virtual_epochs} {step}/{loader_steps} "
                    f"loss={running_ce/denom:.4f} acc={running_correct/denom:.2%} "
                    f"mask={ratio:.1%} strategy={strategy} lr={lr:.2e} "
                    f"{speed:,.0f} tok/s", flush=True
                )

        last_val_loss, last_val_acc = evaluate(model, val_loader, tok, cfg, device, distributed, rank)
        if rank == 0:
            best_val = min(best_val, last_val_loss)
            print(
                f"Epoch {epoch_num} complete | val_loss={last_val_loss:.4f} "
                f"val_acc={last_val_acc:.2%} masked_ppl={math.exp(min(20,last_val_loss)):.2f}",
                flush=True
            )
            save_training_checkpoint(
                last_path, raw_model, optimizer, scaler, cfg, epoch_num, global_step,
                exposed_tokens, best_val, tokenizer_sha,
            )
        barrier(distributed)

    if rank == 0:
        save_final_model(
            final_path, raw_model, cfg, cfg.virtual_epochs, global_step, tokenizer_sha,
            last_val_loss, last_val_acc, best_val,
        )
        print(f"Final pure model saved: {final_path}", flush=True)
    barrier(distributed)
    if distributed:
        dist.destroy_process_group()


# -------------------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="HCAM 100M public pretraining reference")
    sub = p.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="convert raw text/json/jsonl files to memmapped character streams")
    prep.add_argument("--input", nargs="+", required=True, help="files or directories")
    prep.add_argument("--output", required=True, help="output prefix, e.g. data/train_stream")
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--mask-guide-tokenizer", default=None)
    prep.add_argument("--min-chars", type=int, default=80)
    prep.add_argument("--max-chars", type=int, default=12000)
    prep.add_argument("--max-tokens", type=int, default=0, help="0 = unlimited; useful for smoke tests")
    prep.set_defaults(func=prepare_stream)

    tr = sub.add_parser("train", help="pretrain HCAM from prepared streams")
    tr.add_argument("--train-prefix", required=True)
    tr.add_argument("--val-prefix", required=True)
    tr.add_argument("--tokenizer", required=True)
    tr.add_argument("--output-dir", required=True)
    tr.add_argument("--resume", default=None)
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--seq-len", type=int, default=600)
    tr.add_argument("--global-batch", type=int, default=384)
    tr.add_argument("--micro-batch", type=int, default=16,
                    help="per GPU; original late-stage run used 24/GPU when memory allowed")
    tr.add_argument("--peak-lr", type=float, default=7e-4)
    tr.add_argument("--weight-decay", type=float, default=0.08)
    tr.add_argument("--num-workers", type=int, default=0)
    tr.add_argument("--log-every", type=int, default=50)
    tr.add_argument("--max-train-sequences", type=int, default=0, help="0 = all; smoke/debug only")
    tr.add_argument("--max-val-sequences", type=int, default=0, help="0 = all")
    tr.set_defaults(func=train)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
