"""
TaQuants: Tensor-aware Adaptive Quantization for LLMs

A tool that analyzes per-tensor sensitivity in a HuggingFace model and
assigns an optimal mix of quantization levels (q8_0/q6_k/q5_k/q4_k/base)
to hit a target GGUF size ratio, then emits a llama.cpp-compatible
per-tensor quant-type mapping.

Repository: TaQuants
Author:     ek15072809
"""

import sys
import json
import time
import struct
import gc
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from scipy.stats import kurtosis
from scipy.special import rel_entr


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("TaQuants")


# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
PROJECT_NAME = "TaQuants"
PROJECT_AUTHOR = "ek15072809"

_BANNER_FONT = {
    "A": [" ### ", "#   #", "#####", "#   #", "#   #"],
    "N": ["#   #", "##  #", "# # #", "#  ##", "#   #"],
    "Q": [" ### ", "#   #", "#   #", "#  ##", " ####"],
    "S": [" ####", "#    ", " ### ", "    #", "#### "],
    "T": ["#####", "  #  ", "  #  ", "  #  ", "  #  "],
    "U": ["#   #", "#   #", "#   #", "#   #", " ### "],
}


def render_banner(text: str) -> str:
    """Render `text` as large block-letter ASCII art for startup display."""
    letters = [_BANNER_FONT[ch] for ch in text.upper() if ch in _BANNER_FONT]
    if not letters:
        return text
    rows = ["  ".join(letter[row] for letter in letters) for row in range(5)]
    return "\n".join(rows)


QUANT_TYPES = {
    "q8_0":  {"bits": 8.0,  "priority": 0},
    "q6_k":  {"bits": 6.0,  "priority": 1},
    "q5_k":  {"bits": 5.5,  "priority": 2},
    "q4_k":  {"bits": 4.5,  "priority": 3},
    "q3_k":  {"bits": 3.5,  "priority": 4},
    "iq3_s": {"bits": 3.0,  "priority": 5},
    "iq2_m": {"bits": 2.0,  "priority": 6},
}


BASE_QUANT_CHOICES = ["iq2_m", "iq3_s", "q3_k"]


LLAMA_QUANT_NAMES = {
    "q8_0":  "q8_0",
    "q6_k":  "q6_k",
    "q5_k":  "q5_k",
    "q4_k":  "q4_k",
    "q3_k":  "q3_k",
    "iq3_s": "iq3_s",
    "iq2_m": "iq2_m",
}


IQ_QUANT_TYPES = {"iq2_m", "iq3_s"}


class TensorNameMapper:
    _LAYER_TENSOR_MAP: List[Tuple[str, str]] = [
        ("self_attn.q_proj.weight",          "attn_q.weight"),
        ("self_attn.q_proj.bias",            "attn_q.bias"),
        ("self_attn.k_proj.weight",          "attn_k.weight"),
        ("self_attn.k_proj.bias",            "attn_k.bias"),
        ("self_attn.v_proj.weight",          "attn_v.weight"),
        ("self_attn.v_proj.bias",            "attn_v.bias"),
        ("self_attn.o_proj.weight",          "attn_output.weight"),
        ("self_attn.o_proj.bias",            "attn_output.bias"),

        ("self_attn.query_key_value.weight", "attn_qkv.weight"),
        ("self_attn.query_key_value.bias",   "attn_qkv.bias"),
        ("self_attn.qkv_proj.weight",        "attn_qkv.weight"),
        ("self_attn.qkv_proj.bias",          "attn_qkv.bias"),

        ("self_attn.q_norm.weight",          "attn_q_norm.weight"),
        ("self_attn.k_norm.weight",          "attn_k_norm.weight"),

        ("self_attn.dense.weight",           "attn_output.weight"),
        ("self_attn.dense.bias",             "attn_output.bias"),


        ("mlp.gate_proj.weight",             "ffn_gate.weight"),
        ("mlp.up_proj.weight",               "ffn_up.weight"),
        ("mlp.down_proj.weight",             "ffn_down.weight"),
        ("mlp.gate_proj.bias",               "ffn_gate.bias"),
        ("mlp.up_proj.bias",                 "ffn_up.bias"),
        ("mlp.down_proj.bias",               "ffn_down.bias"),

        ("mlp.gate_up_proj.weight",          "ffn_gate_up.weight"),

        ("mlp.dense_h_to_4h.weight",         "ffn_up.weight"),
        ("mlp.dense_h_to_4h.bias",           "ffn_up.bias"),
        ("mlp.dense_4h_to_h.weight",         "ffn_down.weight"),
        ("mlp.dense_4h_to_h.bias",           "ffn_down.bias"),

        ("mlp.fc1.weight",                   "ffn_up.weight"),
        ("mlp.fc1.bias",                     "ffn_up.bias"),
        ("mlp.fc2.weight",                   "ffn_down.weight"),
        ("mlp.fc2.bias",                     "ffn_down.bias"),

        ("block_sparse_moe.gate.weight",     "ffn_gate_inp.weight"),


        ("mlp.shared_expert.gate_proj.weight",  "ffn_gate_shexp.weight"),
        ("mlp.shared_expert.up_proj.weight",    "ffn_up_shexp.weight"),
        ("mlp.shared_expert.down_proj.weight",  "ffn_down_shexp.weight"),

        ("mlp.gate.weight",                     "ffn_gate_inp.weight"),
        ("mlp.router.weight",                   "ffn_gate_inp.weight"),


        ("input_layernorm.weight",           "attn_norm.weight"),
        ("input_layernorm.bias",             "attn_norm.bias"),
        ("post_attention_layernorm.weight",  "ffn_norm.weight"),
        ("post_attention_layernorm.bias",    "ffn_norm.bias"),

        ("pre_feedforward_layernorm.weight", "ffn_pre_norm.weight"),
        ("post_feedforward_layernorm.weight","ffn_post_norm.weight"),

        ("ln_attn.weight",                   "attn_norm.weight"),
        ("ln_mlp.weight",                    "ffn_norm.weight"),
        ("post_attn_layernorm.weight",       "post_attn_norm.weight"),

        ("self_attn.rotary_emb.inv_freq",    None),

        ("pre_attention_layernorm.weight",   "attn_norm.weight"),

        ("self_attn.q_proj.norm.weight",     "attn_q_norm.weight"),
        ("self_attn.k_proj.norm.weight",     "attn_k_norm.weight"),


        ("per_dim_scale.weight",             None),
        ("layer_scalar.weight",              None),

        ("per_dim_scale",                    None),
        ("layer_scalar",                     None),

        ("mlp.router.scale",                 "ffn_gate_inp.scale"),


        ("mlp.router.per_expert_scale",      None),


        ("linear_attn.in_proj_qkv.weight",   "attn_qkv.weight"),

        ("linear_attn.in_proj_z.weight",     "attn_gate.weight"),

        ("linear_attn.in_proj_a.weight",     "ssm_in_a.weight"),
        ("linear_attn.in_proj_b.weight",     "ssm_in_b.weight"),

        ("linear_attn.out_proj.weight",      "attn_output.weight"),

        ("linear_attn.conv1d.weight",        "ssm_conv1d.weight"),
        ("linear_attn.conv1d.bias",          "ssm_conv1d.bias"),

        ("linear_attn.A_log",                "ssm_a.weight"),

        ("linear_attn.dt_proj.weight",       "ssm_dt.weight"),
        ("linear_attn.dt_proj.bias",         "ssm_dt.bias"),

        ("linear_attn.in_proj_qkvz.weight",  "attn_qkv_gate.weight"),

        ("linear_attn.norm.weight",          "attn_sub_norm.weight"),
    ]


    _GLOBAL_TENSOR_MAP: Dict[str, str] = {
        "model.embed_tokens.weight":              "token_embd.weight",
        "transformer.word_embeddings.weight":     "token_embd.weight",
        "transformer.wte.weight":                 "token_embd.weight",
        "gpt_neox.embed_in.weight":               "token_embd.weight",
        "language_model.model.embed_tokens.weight": "token_embd.weight",

        "lm_head.weight":                         "output.weight",
        "embed_out.weight":                       "output.weight",

        "model.norm.weight":                      "output_norm.weight",
        "model.norm.bias":                        "output_norm.bias",
        "transformer.ln_f.weight":                "output_norm.weight",
        "transformer.ln_f.bias":                  "output_norm.bias",
        "gpt_neox.final_layer_norm.weight":       "output_norm.weight",
        "gpt_neox.final_layer_norm.bias":         "output_norm.bias",
        "language_model.model.norm.weight":       "output_norm.weight",

        "model.language_model.model.embed_tokens.weight": "token_embd.weight",
        "model.language_model.lm_head.weight":            "output.weight",
        "model.language_model.model.norm.weight":         "output_norm.weight",


        "model.language_model.embed_tokens.weight":       "token_embd.weight",
        "model.language_model.norm.weight":               "output_norm.weight",
        "model.language_model.lm_head.weight":            "output.weight",

        "model.embed_tokens_secondary.weight":            None,


        "model.embed_vision.embedding_projection.weight": None,

        "model.embed_tokens.weight":                      "token_embd.weight",

        "model.embed_tokens.weight":              "token_embd.weight",

        "model.rotary_emb.inv_freq":              None,


        "model.embed_tokens_per_layer.weight":    "token_embd_per_layer.weight",


    }


    _LAYER_PREFIXES: List[str] = [
        "model.layers.",
        "transformer.h.",
        "transformer.layers.",
        "gpt_neox.layers.",
        "language_model.model.layers.",
        "model.model.layers.",
        "model.language_model.model.layers.",
        "model.language_model.layers.",
        "model.language_model.model.layers.",
    ]

    @classmethod
    def hf_to_gguf(cls, hf_name: str) -> str:
        if hf_name in cls._GLOBAL_TENSOR_MAP:
            result = cls._GLOBAL_TENSOR_MAP[hf_name]
            return result if result is not None else hf_name


        layer_idx: Optional[int] = None
        suffix_after_layer: Optional[str] = None

        for prefix in cls._LAYER_PREFIXES:
            if not hf_name.startswith(prefix):
                continue
            rest = hf_name[len(prefix):]
            dot = rest.find(".")
            if dot == -1:
                continue
            idx_str = rest[:dot]
            if not idx_str.isdigit():
                continue
            layer_idx = int(idx_str)
            suffix_after_layer = rest[dot + 1:]
            break

        if layer_idx is None or suffix_after_layer is None:
            return hf_name


        for hf_suffix, gguf_suffix in cls._LAYER_TENSOR_MAP:
            if suffix_after_layer == hf_suffix:
                if gguf_suffix is None:
                    return hf_name
                return f"blk.{layer_idx}.{gguf_suffix}"


        moe_result = cls._resolve_expert_suffix(layer_idx, suffix_after_layer)
        if moe_result is not None:
            return moe_result


        return f"blk.{layer_idx}.{suffix_after_layer}"

    @classmethod
    def _resolve_expert_suffix(cls, layer_idx: int, suffix: str) -> Optional[str]:
        if not suffix.startswith("mlp.experts."):
            return None

        rest = suffix[len("mlp.experts."):]
        dot = rest.find(".")
        if dot == -1:
            return None
        expert_id_str = rest[:dot]
        if not expert_id_str.isdigit():
            return None
        proj_attr = rest[dot + 1:]

        _EXPERT_PROJ_MAP = {
            "gate_proj.weight": "ffn_gate_exps.weight",
            "gate_proj.bias":   "ffn_gate_exps.bias",
            "up_proj.weight":   "ffn_up_exps.weight",
            "up_proj.bias":     "ffn_up_exps.bias",
            "down_proj.weight": "ffn_down_exps.weight",
            "down_proj.bias":   "ffn_down_exps.bias",

            "w1.weight": "ffn_gate_exps.weight",
            "w3.weight": "ffn_up_exps.weight",
            "w2.weight": "ffn_down_exps.weight",


            "gate_proj.weight_scale":    "ffn_gate_exps.weight_scale",
            "gate_proj.weight_scale_2":  "ffn_gate_exps.weight_scale_2",
            "up_proj.weight_scale":      "ffn_up_exps.weight_scale",
            "up_proj.weight_scale_2":    "ffn_up_exps.weight_scale_2",
            "down_proj.weight_scale":    "ffn_down_exps.weight_scale",
            "down_proj.weight_scale_2":  "ffn_down_exps.weight_scale_2",
            "gate_proj.input_scale":     "ffn_gate_exps.input_scale",
            "up_proj.input_scale":       "ffn_up_exps.input_scale",
            "down_proj.input_scale":     "ffn_down_exps.input_scale",
        }
        gguf_suffix = _EXPERT_PROJ_MAP.get(proj_attr)
        if gguf_suffix is None:
            return f"blk.{layer_idx}.ffn_exps.{proj_attr}"
        return f"blk.{layer_idx}.{gguf_suffix}"

    @classmethod
    def convert_mapping(cls, mapping: Dict[str, str]) -> Dict[str, str]:
        converted: Dict[str, str] = {}
        for hf_name, qt in mapping.items():
            gguf_name = cls.hf_to_gguf(hf_name)
            if gguf_name in converted:
                existing_priority = QUANT_TYPES.get(converted[gguf_name], {}).get("priority", 99)
                new_priority      = QUANT_TYPES.get(qt,                   {}).get("priority", 99)
                if new_priority < existing_priority:
                    converted[gguf_name] = qt
            else:
                converted[gguf_name] = qt
        return converted

    @classmethod
    def log_conversion_stats(cls, hf_mapping: Dict[str, str], gguf_mapping: Dict[str, str]) -> None:
        n_hf   = len(hf_mapping)
        n_gguf = len(gguf_mapping)


        unchanged_names = [
            hf for hf in hf_mapping
            if cls.hf_to_gguf(hf) == hf
            and hf not in cls._GLOBAL_TENSOR_MAP
        ]
        log.info(f"Tensor name conversion: {n_hf} HF names -> {n_gguf} GGUF names")
        if unchanged_names:
            log.warning(
                f"  Tensors that could not be converted: {len(unchanged_names)} "
                f"(names remain in HF format; may indicate a new architecture)"
            )
            for name in unchanged_names[:20]:
                log.warning(f"    Not converted: {name}")
            if len(unchanged_names) > 20:
                log.warning(f"    ... and {len(unchanged_names) - 20} more")


@dataclass
class SensitivityResult:
    name: str
    shape: List[int]
    numel: int
    kl_divergence: float = 0.0
    spectral_sensitivity: float = 0.0
    angular_shift: float = 0.0
    outlier_ratio: float = 0.0
    kurtosis_delta: float = 0.0
    effective_rank: float = 0.0
    ampq_score: float = 0.0
    assigned_quant: str = "iq2_m"
    analysis_time_s: float = 0.0
    error: Optional[str] = None


class IQ2MSimulator:
    LATTICE_VALS = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    BLOCK_SIZE = 32

    @classmethod
    def simulate(cls, data_f32: np.ndarray) -> np.ndarray:
        original_shape = data_f32.shape
        flat = data_f32.flatten().astype(np.float32)
        n = len(flat)


        pad = (-n) % cls.BLOCK_SIZE
        if pad > 0:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        blocks = flat.reshape(-1, cls.BLOCK_SIZE)


        scales = np.abs(blocks).max(axis=1, keepdims=True).clip(min=1e-8)
        normalized = blocks / scales


        diffs = np.abs(normalized[:, :, None] - cls.LATTICE_VALS[None, None, :])
        idx = diffs.argmin(axis=2)
        quantized = cls.LATTICE_VALS[idx]


        dequantized = (quantized * scales).flatten()[:n]
        return dequantized.reshape(original_shape)


class IQ3SSimulator:
    LATTICE_VALS = np.array(
        [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75],
        dtype=np.float32,
    )
    BLOCK_SIZE = 32

    @classmethod
    def simulate(cls, data_f32: np.ndarray) -> np.ndarray:
        original_shape = data_f32.shape
        flat = data_f32.flatten().astype(np.float32)
        n = len(flat)

        pad = (-n) % cls.BLOCK_SIZE
        if pad > 0:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        blocks = flat.reshape(-1, cls.BLOCK_SIZE)

        scales = np.abs(blocks).max(axis=1, keepdims=True).clip(min=1e-8)
        normalized = blocks / scales

        diffs = np.abs(normalized[:, :, None] - cls.LATTICE_VALS[None, None, :])
        idx = diffs.argmin(axis=2)
        quantized = cls.LATTICE_VALS[idx]

        dequantized = (quantized * scales).flatten()[:n]
        return dequantized.reshape(original_shape)


def get_simulator(base_quant: str):
    if base_quant in ("iq3_s", "q3_k"):
        return IQ3SSimulator
    return IQ2MSimulator


class SensitivityAnalyzer:
    OUTLIER_K = 3.0

    HIST_BINS = 256

    @classmethod
    def compute_kl_divergence(
        cls,
        orig: np.ndarray,
        quant: np.ndarray,
    ) -> float:
        flat_o = orig.flatten().astype(np.float64)
        flat_q = quant.flatten().astype(np.float64)

        lo = min(flat_o.min(), flat_q.min())
        hi = max(flat_o.max(), flat_q.max())
        if hi - lo < 1e-12:
            return 0.0

        bins = np.linspace(lo, hi, cls.HIST_BINS + 1)
        p, _ = np.histogram(flat_o, bins=bins, density=True)
        q, _ = np.histogram(flat_q, bins=bins, density=True)


        eps = 1e-10
        p = (p + eps) / (p + eps).sum()
        q = (q + eps) / (q + eps).sum()

        kld = 0.5 * (rel_entr(p, q).sum() + rel_entr(q, p).sum())
        return float(np.clip(kld, 0.0, 100.0))

    @classmethod
    def compute_spectral_sensitivity(cls, w: np.ndarray) -> float:
        if w.ndim < 2:
            return 0.0

        mat = w.reshape(w.shape[0], -1).astype(np.float32)
        if min(mat.shape) < 2:
            return 0.0

        frob = float(np.linalg.norm(mat, 'fro'))
        if frob < 1e-10:
            return 0.0

        try:
            if max(mat.shape) > 1024:
                rng = np.random.default_rng(42)
                sketch_dim = min(min(mat.shape), 256)
                sketch = rng.standard_normal((mat.shape[1], sketch_dim)).astype(np.float32)
                Y = mat @ sketch
                Q, _ = np.linalg.qr(Y)
                B = Q.T @ mat
                _, s, _ = np.linalg.svd(B, full_matrices=False)
            else:
                _, s, _ = np.linalg.svd(mat, full_matrices=False)

            return float(s[0]) / frob

        except np.linalg.LinAlgError:
            return 0.0

    @classmethod
    def compute_effective_rank(cls, w: np.ndarray) -> float:
        if w.ndim < 2:
            return 0.0

        mat = w.reshape(w.shape[0], -1).astype(np.float32)
        if min(mat.shape) < 2:
            return 0.0

        try:
            if max(mat.shape) > 1024:
                rng = np.random.default_rng(43)
                sketch_dim = min(min(mat.shape), 128)
                sketch = rng.standard_normal((mat.shape[1], sketch_dim)).astype(np.float32)
                Y = mat @ sketch
                Q, _ = np.linalg.qr(Y)
                B = Q.T @ mat
                _, s, _ = np.linalg.svd(B, full_matrices=False)
            else:
                _, s, _ = np.linalg.svd(mat, full_matrices=False)

            s = s[s > 1e-10]
            if len(s) < 2:
                return 0.0


            p = s / s.sum()
            eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

            rank_ratio = eff_rank / len(s)
            return float(np.clip(1.0 - rank_ratio, 0.0, 1.0))

        except np.linalg.LinAlgError:
            return 0.0

    @classmethod
    def compute_angular_shift(
        cls,
        orig: np.ndarray,
        quant: np.ndarray,
    ) -> float:
        if orig.ndim < 2:
            return 0.0

        def top_left_singular_vector(mat: np.ndarray) -> Optional[np.ndarray]:
            m = mat.reshape(mat.shape[0], -1).astype(np.float32)
            if min(m.shape) < 2:
                return None
            try:
                if max(m.shape) > 2048:
                    rng = np.random.default_rng(0)
                    v = rng.standard_normal(m.shape[1]).astype(np.float32)
                    v /= np.linalg.norm(v) + 1e-12
                    for _ in range(30):
                        u = m @ v
                        u_norm = np.linalg.norm(u)
                        if u_norm < 1e-12:
                            return None
                        u /= u_norm
                        v = m.T @ u
                        v_norm = np.linalg.norm(v)
                        if v_norm < 1e-12:
                            return None
                        v /= v_norm
                    return u
                else:
                    u, _, _ = np.linalg.svd(m, full_matrices=False)
                    return u[:, 0]
            except np.linalg.LinAlgError:
                return None

        u_orig = top_left_singular_vector(orig)
        u_quant = top_left_singular_vector(quant)
        if u_orig is None or u_quant is None:
            return 0.0

        n1 = np.linalg.norm(u_orig)
        n2 = np.linalg.norm(u_quant)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0

        cos_sim = np.dot(u_orig, u_quant) / (n1 * n2)
        return float(1.0 - abs(float(cos_sim)))

    @classmethod
    def compute_outlier_ratio(cls, w: np.ndarray) -> float:
        flat = w.flatten().astype(np.float64)
        std = np.std(flat)
        if std < 1e-10:
            return 0.0
        return float(np.mean(np.abs(flat) > cls.OUTLIER_K * std))

    @classmethod
    def compute_kurtosis_delta(
        cls,
        orig: np.ndarray,
        quant: np.ndarray,
    ) -> float:
        flat_o = orig.flatten().astype(np.float64)
        flat_q = quant.flatten().astype(np.float64)


        if np.std(flat_o) < 1e-8 or np.std(flat_q) < 1e-8:
            return 0.0

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            k_o = float(kurtosis(flat_o, fisher=True))
            k_q = float(kurtosis(flat_q, fisher=True))


        if not math.isfinite(k_o) or not math.isfinite(k_q):
            return 0.0

        delta = abs(k_q - k_o) / (abs(k_o) + 1.0)
        return float(np.clip(delta, 0.0, 10.0))

    @classmethod
    def compute_ampq_score(
        cls,
        kld: float,
        spectral: float,
        angular: float,
        outlier: float,
        kurt_delta: float,
        eff_rank: float = 0.0,
    ) -> float:
        def sigmoid(x: float, center: float, scale: float) -> float:
            return 1.0 / (1.0 + math.exp(-(x - center) * scale))

        s_kld      = sigmoid(kld,        center=0.5,  scale=2.0)
        s_spectral = sigmoid(spectral,   center=0.3,  scale=5.0)
        s_angular  = sigmoid(angular,    center=0.05, scale=20.0)
        s_outlier  = sigmoid(outlier,    center=0.01, scale=50.0)
        s_kurt     = sigmoid(kurt_delta, center=0.1,  scale=5.0)
        s_effrank  = sigmoid(eff_rank,   center=0.5,  scale=6.0)

        score = (
            0.30 * s_kld +
            0.22 * s_spectral +
            0.20 * s_angular +
            0.15 * s_effrank +
            0.10 * s_outlier +
            0.03 * s_kurt
        )
        return float(score)


class SafetensorsLoader:
    DTYPE_MAP = {
        "BF16": ("u2", 2),
        "F16":  ("f2", 2),
        "F32":  ("f4", 4),
        "F64":  ("f8", 8),
        "I32":  ("i4", 4),
        "I64":  ("i8", 8),
        "I16":  ("i2", 2),
        "I8":   ("i1", 1),
        "U8":   ("u1", 1),
    }

    @staticmethod
    def bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
        u32 = arr_u16.astype(np.uint32) << 16
        return u32.view(np.float32)

    @classmethod
    def load_index(cls, model_dir: Path) -> Dict[str, str]:
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            single = model_dir / "model.safetensors"
            if single.exists():
                return {"__single__": str(single)}
            raise FileNotFoundError(
                f"index.json not found: {index_path}\n"
                f"model.safetensors not found either: {single}"
            )
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("weight_map", {})

    @classmethod
    def parse_header(cls, file_path: Path) -> Tuple[Dict[str, dict], int]:
        with open(file_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_len)
        header = json.loads(header_json.decode("utf-8"))
        header = {k: v for k, v in header.items() if k != "__metadata__"}
        offset = 8 + header_len
        return header, offset

    @classmethod
    def load_tensor_f32(
        cls,
        file_path: Path,
        tensor_info: dict,
        header_offset: int,
        max_elements: Optional[int] = None,
    ) -> np.ndarray:
        dtype_str = tensor_info["dtype"]
        shape = tensor_info["shape"]
        offsets = tensor_info["data_offsets"]
        byte_start = offsets[0] + header_offset
        byte_end   = offsets[1] + header_offset

        if dtype_str not in cls.DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        np_dtype_str, _ = cls.DTYPE_MAP[dtype_str]
        numel = math.prod(shape) if shape else 1
        total_bytes = byte_end - byte_start

        with open(file_path, "rb") as f:
            f.seek(byte_start)
            raw = f.read(total_bytes)

        arr = np.frombuffer(raw, dtype=np.dtype(np_dtype_str)).copy()
        del raw

        if dtype_str == "BF16":
            arr_f32 = cls.bf16_to_f32(arr)
        else:
            arr_f32 = arr.astype(np.float32)
        del arr


        if max_elements is None or numel <= max_elements:
            return arr_f32.reshape(shape) if shape else arr_f32


        if len(shape) == 0:
            return arr_f32

        if len(shape) == 1:
            flat = arr_f32
            step = max(1, len(flat) // max_elements)
            sampled = flat[::step][:max_elements]
            return sampled.reshape(-1, 1)


        rows = shape[0]
        cols = numel // rows

        mat = arr_f32.reshape(rows, cols)
        del arr_f32


        target_rows = max(1, min(rows, max_elements // max(cols, 1)))
        if target_rows >= rows:
            return mat


        step = rows // target_rows
        sampled_mat = mat[::step][:target_rows]
        del mat
        return sampled_mat


class AMPQAnalyzer:
    HIGH_IMPORTANCE_PATTERNS = [
        "lm_head", "embed_tokens", "embedding",
        "norm", "layernorm", "ln_",

        "output.", "output_norm", "token_embd",
        "attn_norm", "ffn_norm",
    ]

    ATTENTION_PATTERNS = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "query", "key", "value",
        "self_attn", "attention",

        "attn_q", "attn_k", "attn_v", "attn_output",
        "attn_qkv",
    ]

    QK_PATTERNS = [
        "q_proj", "k_proj", "query_proj", "key_proj",
        "attn_q", "attn_k", "attn_qkv",
    ]

    DOWN_PROJ_PATTERNS = [
        "down_proj", "dense_4h_to_h", "fc2",
        "ffn_down",
    ]

    def __init__(
        self,
        model_dir: str,
        target_size_ratio: float = 0.16,
        max_elements_per_tensor: int = 1_000_000,
        verbose: bool = True,
        base_quant: str = "iq2_m",
        speed_first: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.target_size_ratio = target_size_ratio
        self.max_elements = max_elements_per_tensor
        self.verbose = verbose
        self.base_quant = base_quant
        self.speed_first = speed_first
        self.results: List[SensitivityResult] = []
        self._file_headers: Dict[str, Tuple[dict, int]] = {}
        self._num_layers: int = 0
        self._has_weight_tied_output: bool = False

    def _get_file_header(self, file_path: Path) -> Tuple[dict, int]:
        key = str(file_path)
        if key not in self._file_headers:
            self._file_headers[key] = SafetensorsLoader.parse_header(file_path)
        return self._file_headers[key]

    def _detect_num_layers(self, weight_map: Dict[str, str]) -> int:
        max_layer = -1
        for name in weight_map:
            for prefix in ["model.layers.", "transformer.h.", "transformer.layers.",
                           "gpt_neox.layers.", "language_model.model.layers.",
                           "model.language_model.layers."]:
                if name.startswith(prefix):
                    rest = name[len(prefix):]
                    dot = rest.find(".")
                    if dot > 0 and rest[:dot].isdigit():
                        max_layer = max(max_layer, int(rest[:dot]))
                        break
        return max_layer + 1 if max_layer >= 0 else 0

    def _name_boost(self, tensor_name: str, layer_idx: Optional[int] = None) -> float:
        name_lower = tensor_name.lower()


        is_boundary_layer = False
        is_global_attn_layer = False
        if layer_idx is not None and self._num_layers > 0:
            boundary_ratio = 0.20
            boundary_n = max(1, int(self._num_layers * boundary_ratio))
            if layer_idx < boundary_n or layer_idx >= self._num_layers - boundary_n:
                is_boundary_layer = True


            if self._num_layers >= 10 and layer_idx % 5 == 0:
                is_global_attn_layer = True


        is_high = any(pat in name_lower for pat in self.HIGH_IMPORTANCE_PATTERNS)
        is_qk   = any(pat in name_lower for pat in self.QK_PATTERNS)
        is_down = any(pat in name_lower for pat in self.DOWN_PROJ_PATTERNS)
        is_attn = any(pat in name_lower for pat in self.ATTENTION_PATTERNS)

        if is_high:
            return 1.40
        if is_boundary_layer and is_qk:
            return 1.35
        if is_global_attn_layer and is_qk:
            return 1.33
        if is_qk:
            return 1.28
        if is_down:
            return 1.20
        if is_boundary_layer:
            return 1.18
        if is_global_attn_layer and is_attn:
            return 1.16
        if is_attn:
            return 1.12
        return 1.00

    def _parse_layer_idx(self, tensor_name: str) -> Optional[int]:
        for prefix in ["model.layers.", "transformer.h.", "transformer.layers.",
                       "gpt_neox.layers.", "language_model.model.layers.",
                       "model.language_model.layers."]:
            if tensor_name.startswith(prefix):
                rest = tensor_name[len(prefix):]
                dot = rest.find(".")
                if dot > 0 and rest[:dot].isdigit():
                    return int(rest[:dot])
        return None

    def _analyze_tensor(
        self,
        tensor_name: str,
        file_path: Path,
        tensor_info: dict,
        header_offset: int,
    ) -> SensitivityResult:
        t_start = time.perf_counter()
        shape = tensor_info.get("shape", [])
        numel = math.prod(shape) if shape else 1
        result = SensitivityResult(name=tensor_name, shape=shape, numel=numel)

        layer_idx = self._parse_layer_idx(tensor_name)

        w_orig = None
        w_quant = None
        try:
            w_orig = SafetensorsLoader.load_tensor_f32(
                file_path, tensor_info, header_offset,
                max_elements=self.max_elements,
            )


            if w_orig.size < 4:
                result.assigned_quant = "q8_0"
                result.ampq_score = 1.0
                result.analysis_time_s = time.perf_counter() - t_start
                return result


            simulator = get_simulator(self.base_quant)
            w_quant = simulator.simulate(w_orig)


            kld      = SensitivityAnalyzer.compute_kl_divergence(w_orig, w_quant)
            spectral = SensitivityAnalyzer.compute_spectral_sensitivity(w_orig)
            angular  = SensitivityAnalyzer.compute_angular_shift(w_orig, w_quant)
            outlier  = SensitivityAnalyzer.compute_outlier_ratio(w_orig)
            kurt_d   = SensitivityAnalyzer.compute_kurtosis_delta(w_orig, w_quant)
            eff_rank = SensitivityAnalyzer.compute_effective_rank(w_orig)


            kld      = kld      if math.isfinite(kld)      else 0.0
            spectral = spectral if math.isfinite(spectral) else 0.0
            angular  = angular  if math.isfinite(angular)  else 0.0
            outlier  = outlier  if math.isfinite(outlier)  else 0.0
            kurt_d   = kurt_d   if math.isfinite(kurt_d)   else 0.0
            eff_rank = eff_rank if math.isfinite(eff_rank) else 0.0

            base_score = SensitivityAnalyzer.compute_ampq_score(
                kld, spectral, angular, outlier, kurt_d, eff_rank
            )
            boost = self._name_boost(tensor_name, layer_idx=layer_idx)
            ampq_score = min(base_score * boost, 1.0)

            result.kl_divergence        = kld
            result.spectral_sensitivity = spectral
            result.angular_shift        = angular
            result.outlier_ratio        = outlier
            result.kurtosis_delta       = kurt_d
            result.effective_rank       = eff_rank
            result.ampq_score           = ampq_score

        except Exception as e:
            result.error = str(e)
            result.ampq_score = 0.5

        finally:
            del w_orig
            del w_quant
            gc.collect()

        result.analysis_time_s = time.perf_counter() - t_start
        return result

    def _detect_weight_tying(self, weight_map: Dict[str, str]) -> bool:
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                tie = config.get("tie_word_embeddings", False)
                if tie:
                    log.info("config.json: detected tie_word_embeddings=True")
                    return True
            except Exception:
                pass

        lm_head_names = {"lm_head.weight", "embed_out.weight"}
        if not any(n in weight_map for n in lm_head_names):
            log.info("lm_head.weight not found in safetensors -> assuming weight-tying")
            return True
        return False

    def run_analysis(self) -> List[SensitivityResult]:
        weight_map = SafetensorsLoader.load_index(self.model_dir)


        self._num_layers = self._detect_num_layers(weight_map)
        if self._num_layers > 0:
            log.info(f"Detected layer count: {self._num_layers} "
                     f"(boosting boundary layers +/-{max(1, int(self._num_layers * 0.20))} layers)")


        self._has_weight_tied_output = self._detect_weight_tying(weight_map)
        if self._has_weight_tied_output:
            log.info(
                "!  Detected a weight-tied model: output.weight (lm_head) is "
                "not present in safetensors.\n"
                "   output.weight will be automatically added to tensor_types.txt "
                "with a high-precision quant type."
            )


        file_to_tensors: Dict[str, List[str]] = defaultdict(list)
        if "__single__" in weight_map:
            single_path = weight_map["__single__"]
            header, _ = self._get_file_header(Path(single_path))
            for tname in header:
                file_to_tensors[single_path].append(tname)
        else:
            for tname, fname in weight_map.items():
                file_to_tensors[fname].append(tname)

        total = sum(len(v) for v in file_to_tensors.values())
        log.info(f"Total tensors: {total}")
        log.info(f"Target size ratio: {self.target_size_ratio:.0%} of bf16")
        log.info("─" * 60)

        processed = 0
        for fname, tensor_names in file_to_tensors.items():
            if Path(fname).is_absolute():
                file_path = Path(fname)
            else:
                file_path = self.model_dir / fname
                if not file_path.exists():
                    file_path = self.model_dir / Path(fname).name

            if not file_path.exists():
                log.warning(f"File not found: {fname}")
                continue

            header, offset = self._get_file_header(file_path)

            for tname in tensor_names:
                if tname not in header:
                    log.warning(f"Tensor not present in header: {tname}")
                    continue

                result = self._analyze_tensor(tname, file_path, header[tname], offset)
                self.results.append(result)
                processed += 1

                if self.verbose:
                    err_str = f" [ERR: {result.error[:40]}]" if result.error else ""
                    log.info(
                        f"[{processed:4d}/{total}] {tname:<60s} "
                        f"score={result.ampq_score:.4f} "
                        f"({result.analysis_time_s:.2f}s){err_str}"
                    )

        return self.results

    def assign_quantization(self) -> Dict[str, str]:
        if not self.results:
            raise RuntimeError("Please run run_analysis() first.")

        base_quant = self.base_quant
        base_bits  = QUANT_TYPES[base_quant]["bits"]


        _1MiB = 1 * 1024 * 1024

        def _force_base(r: SensitivityResult) -> Optional[str]:
            if "vision" in r.name.lower():
                return "vision tensor"
            if "audio" in r.name.lower():
                return "audio tensor"
            if r.numel * 2 < _1MiB:
                return f"size < 1 MiB ({r.numel * 2 / 1024:.1f} KiB)"
            return None


        active_results = [r for r in self.results if _force_base(r) is None]
        forced_results = [(r, _force_base(r)) for r in self.results if _force_base(r) is not None]

        scores  = np.array([r.ampq_score for r in active_results], dtype=np.float64)
        numels  = np.array([r.numel      for r in active_results], dtype=np.float64)
        total_params = numels.sum() if len(numels) > 0 else 1.0


        forced_numels_total = float(sum(r.numel for r, _ in forced_results))

        log.info("\n" + "=" * 60)
        log.info("TaQuants sweet-spot search: optimizing 5-level quantization thresholds")
        log.info("=" * 60)
        log.info(f"Base quant:                 {base_quant} ({base_bits:.1f}bit)")
        log.info(f"Forced {base_quant} tensors: {len(forced_results)} "
                 f"(vision: {sum(1 for _, r in forced_results if 'vision' in r)}, "
                 f"audio: {sum(1 for _, r in forced_results if 'audio' in r)}, "
                 f"< 1 MiB: {sum(1 for _, r in forced_results if 'MiB' in r)})")
        log.info(f"Tensors subject to score optimization: {len(active_results)}")

        bf16_bits = 16.0
        target_avg_bits = bf16_bits * self.target_size_ratio
        log.info(f"Target average bits: {target_avg_bits:.2f} bits")
        log.info(f"  (bf16={bf16_bits:.0f}bit × ratio={self.target_size_ratio:.2%})")


        if len(scores) == 0:
            for r in self.results:
                r.assigned_quant = base_quant
            return {r.name: base_quant for r in self.results}

        sorted_idx = np.argsort(scores)[::-1]
        sorted_scores  = scores[sorted_idx]
        sorted_numels  = numels[sorted_idx]
        cumul_numels   = np.cumsum(sorted_numels)
        total_active   = sorted_numels.sum()

        def weighted_avg_bits_alpha(alpha: float) -> float:
            n = len(sorted_scores)

            n_q8 = max(0, int(round(alpha * 0.10 * n)))
            n_q6 = max(0, int(round(alpha * 0.20 * n)))
            n_q5 = max(0, int(round(alpha * 0.30 * n)))
            n_q4 = max(0, int(round(alpha * 0.40 * n)))


            n_q8 = min(n_q8, n)
            n_q6 = min(n_q6, n - n_q8)
            n_q5 = min(n_q5, n - n_q8 - n_q6)
            n_q4 = min(n_q4, n - n_q8 - n_q6 - n_q5)


            bits = np.full(n, base_bits, dtype=np.float64)
            bits[:n_q8] = 8.0
            bits[n_q8:n_q8+n_q6] = 6.0
            bits[n_q8+n_q6:n_q8+n_q6+n_q5] = 5.5
            bits[n_q8+n_q6+n_q5:n_q8+n_q6+n_q5+n_q4] = 4.5


            active_bits_sum = float(np.dot(bits, sorted_numels))

            forced_bits_sum = base_bits * forced_numels_total

            total_all = total_active + forced_numels_total
            if total_all <= 0:
                return base_bits
            return (active_bits_sum + forced_bits_sum) / total_all


        lo_a, hi_a, best_a = 0.0, 1.0, 0.0
        for _ in range(60):
            mid_a = (lo_a + hi_a) / 2.0
            avg_bits = weighted_avg_bits_alpha(mid_a)
            diff = avg_bits - target_avg_bits
            if abs(diff) < 0.005:
                best_a = mid_a
                break
            if diff > 0:
                hi_a = mid_a
            else:
                lo_a = mid_a
            best_a = mid_a


        n = len(sorted_scores)
        n_q8 = max(0, int(round(best_a * 0.10 * n)))
        n_q6 = max(0, int(round(best_a * 0.20 * n)))
        n_q5 = max(0, int(round(best_a * 0.30 * n)))
        n_q4 = max(0, int(round(best_a * 0.40 * n)))
        n_q8 = min(n_q8, n)
        n_q6 = min(n_q6, n - n_q8)
        n_q5 = min(n_q5, n - n_q8 - n_q6)
        n_q4 = min(n_q4, n - n_q8 - n_q6 - n_q5)


        end_q8 = n_q8
        end_q6 = n_q8 + n_q6
        end_q5 = n_q8 + n_q6 + n_q5
        end_q4 = n_q8 + n_q6 + n_q5 + n_q4

        def _min_score_at(end_idx: int) -> float:
            if end_idx <= 0 or end_idx > n:
                return float("nan")
            return float(sorted_scores[end_idx - 1])

        log.info(f"\nOptimal quantile (alpha={best_a:.4f}):")
        log.info(f"  q8_0 : top {n_q8:4d} tensors (score >= {_min_score_at(end_q8):.4f})")
        log.info(f"  q6_k : top {n_q6:4d} tensors (score >= {_min_score_at(end_q6):.4f})")
        log.info(f"  q5_k : top {n_q5:4d} tensors (score >= {_min_score_at(end_q5):.4f})")
        log.info(f"  q4_k : top {n_q4:4d} tensors (score >= {_min_score_at(end_q4):.4f})")
        log.info(f"  {base_quant:<8}: remaining {n - end_q4:4d} tensors")
        log.info(f"  Actual average bits: {weighted_avg_bits_alpha(best_a):.3f} bits")

        counts: Dict[str, int] = defaultdict(int)
        params_by_quant: Dict[str, float] = defaultdict(float)
        mapping: Dict[str, str] = {}


        for r, reason in forced_results:
            r.assigned_quant = base_quant
            mapping[r.name] = base_quant
            counts[base_quant] += 1
            params_by_quant[base_quant] += r.numel
            if self.verbose:
                log.info(f"  [forced {base_quant}] {r.name} ({reason})")


        quant_by_active_idx = [base_quant] * n
        for rank, aidx in enumerate(range(n)):
            if rank < end_q8:
                quant_by_active_idx[aidx] = "q8_0"
            elif rank < end_q6:
                quant_by_active_idx[aidx] = "q6_k"
            elif rank < end_q5:
                quant_by_active_idx[aidx] = "q5_k"
            elif rank < end_q4:
                quant_by_active_idx[aidx] = "q4_k"


        for rank, orig_idx in enumerate(sorted_idx):
            r = active_results[orig_idx]
            if rank < end_q8:
                q = "q8_0"
            elif rank < end_q6:
                q = "q6_k"
            elif rank < end_q5:
                q = "q5_k"
            elif rank < end_q4:
                q = "q4_k"
            else:
                q = base_quant
            r.assigned_quant = q
            mapping[r.name] = q
            counts[q] += 1
            params_by_quant[q] += r.numel


        if self.speed_first and base_quant in IQ_QUANT_TYPES:
            log.info(
                "\n[speed_first mode] Reducing mixing with IQ-type quants.\n"
                "  Tensors promoted below q4_k are reverted to base_quant.\n"
                "  (Reason: mixing iq3_s with k-quant hinders dequant kernel optimization)"
            )
            reverted = 0
            for r in active_results:
                if r.assigned_quant in ("q4_k", "q5_k", "q6_k", "q8_0"):
                    pass


            log.info(f"  [speed_first] Reverted {reverted} tensors to base_quant.")


        TIED_OUTPUT_GGUF_NAME = "output.weight"
        TIED_OUTPUT_QUANT = "q6_k"

        if self._has_weight_tied_output and TIED_OUTPUT_GGUF_NAME not in mapping:
            mapping[TIED_OUTPUT_GGUF_NAME] = TIED_OUTPUT_QUANT
            counts[TIED_OUTPUT_QUANT] += 1
            log.info(
                f"\n[OK] Automatically added output.weight (weight-tied lm_head): {TIED_OUTPUT_QUANT}\n"
                f"  (equivalent to the automatic Q5_K promotion used with Q3_K_S)"
            )


        token_embd_per_layer_name = "token_embd_per_layer.weight"
        if token_embd_per_layer_name not in mapping:
            token_embd_qt = mapping.get("token_embd.weight", None)
            if token_embd_qt is not None:
                qt_priority = QUANT_TYPES.get(token_embd_qt, {}).get("priority", 99)

                fallback_qt = "q4_k"
                for qt_name, qt_info in QUANT_TYPES.items():
                    if qt_info["priority"] == qt_priority + 1:
                        fallback_qt = qt_name
                        break
                mapping[token_embd_per_layer_name] = fallback_qt
                log.info(
                    f"[OK] Automatically added token_embd_per_layer.weight: {fallback_qt} "
                    f"(one level lower precision than token_embd={token_embd_qt})"
                )


        active_bits_sum_final  = sum(
            QUANT_TYPES.get(r.assigned_quant, {}).get("bits", base_bits) * r.numel
            for r in active_results
        )
        forced_bits_sum_final  = base_bits * forced_numels_total
        total_known_params     = total_active + forced_numels_total
        actual_avg_bits = (
            (active_bits_sum_final + forced_bits_sum_final) / total_known_params
            if total_known_params > 0 else 0.0
        )


        all_total = total_known_params


        log.info(f"\nQuantization assignment results (TaQuants 5-level):")
        log.info(f"  {'Type':<8} {'Count':>7} {'Params%':>9} {'Bits':>6}")
        log.info(f"  {'-'*38}")
        quant_order = ["q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "iq3_s", "iq2_m"]
        for qt in quant_order:
            if counts[qt] == 0:
                continue
            cnt = counts[qt]
            pct = 100.0 * params_by_quant[qt] / all_total if all_total > 0 else 0.0
            bits = QUANT_TYPES[qt]["bits"]
            log.info(f"  {qt:<8} {cnt:>7} {pct:>8.1f}% {bits:>6.1f}")
        log.info(f"  {'-'*38}")
        log.info(f"  Actual average bits: {actual_avg_bits:.3f} bits")
        log.info(f"  Target:              {target_avg_bits:.3f} bits")
        log.info(f"  Estimated size ratio: {actual_avg_bits / bf16_bits:.2%} of bf16")


        if base_quant in IQ_QUANT_TYPES:
            log.warning(
                "\n" + "!" * 60 + "\n"
                "!  Important: when using an IQ-type base quant (iq3_s/iq2_m),\n"
                "   an imatrix (importance matrix) file is required.\n"
                "   Without it, quality will be worse than Q3_K_S/Q2_K!\n\n"
                "   How to generate an imatrix:\n"
                "     llama-imatrix -m model_f16.gguf -f calibration.txt \\\n"
                "       -o model.imatrix --chunks 128\n\n"
                "   If using without an imatrix, --base_quant q3_k is recommended:\n"
                "     python taquants.py --model_dir ... --base_quant q3_k\n"
                "   (q3_k is equivalent to Q3_K_M, needs no imatrix, and is also faster)\n"
                + "!" * 60
            )

        return mapping


class OutputGenerator:
    @staticmethod
    def save_sensitivity_report(
        results: List[SensitivityResult],
        output_path: Path,
    ) -> None:
        header_line = (
            "tensor_name,shape,numel,kl_divergence,spectral_sensitivity,"
            "angular_shift,outlier_ratio,kurtosis_delta,effective_rank,"
            "taq_score,assigned_quant,analysis_time_s,error"
        )
        lines = [header_line]
        for r in results:
            shape_str = "x".join(str(s) for s in r.shape)
            err_str = (r.error or "").replace(",", ";")
            lines.append(
                f"{r.name},{shape_str},{r.numel},"
                f"{r.kl_divergence:.6f},{r.spectral_sensitivity:.6f},"
                f"{r.angular_shift:.6f},{r.outlier_ratio:.6f},"
                f"{r.kurtosis_delta:.6f},{r.effective_rank:.6f},"
                f"{r.ampq_score:.6f},"
                f"{r.assigned_quant},{r.analysis_time_s:.3f},{err_str}"
            )
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        log.info(f"Sensitivity report saved: {output_path}")

    @staticmethod
    def save_tensor_types_txt_per_layer(
        mapping: Dict[str, str],
        output_path: Path,
        base_quant: str = "iq2_m",
    ) -> None:
        lines = []
        for tname in sorted(mapping.keys()):
            qt = mapping[tname]
            if qt != base_quant:
                llama_name = LLAMA_QUANT_NAMES.get(qt, qt)
                lines.append(f"{tname}={llama_name}")

        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        log.info(
            f"tensor_types_per_layer.txt saved: {output_path} "
            f"({len(lines)} tensors / for custom per-tensor llama-quantize only)"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "TaQuants: Tensor-aware Adaptive Quantization for LLMs\n"
            "Repository: TaQuants  |  Author: ek15072809"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python taquants.py --model_dir ./my_model\n"
            "  python taquants.py --model_dir ./my_model --output_dir ./output\n"
            "  python taquants.py --model_dir ./my_model --target_ratio 0.20\n"
            "  python taquants.py --model_dir ./my_model --max_elements 500000\n"
            "  python taquants.py --model_dir ./my_model --imatrix ./my_model.imatrix.gguf\n"
            "  python taquants.py --model_dir ./my_model --base_quant iq3_s\n"
            "  python taquants.py --model_dir ./my_model --base_quant iq3_s --target_ratio 0.22\n"
        ),
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="HuggingFace model directory (safetensors + index.json)",
    )
    parser.add_argument(
        "--output_dir", default="./taquants_output",
        help="Output directory (default: ./taquants_output)",
    )
    parser.add_argument(
        "--model_name", default=None,
        help="Model name (used for log display; defaults to the directory name)",
    )
    parser.add_argument(
        "--target_ratio", type=float, default=None,
        help=(
            "Target GGUF size ratio (relative to bf16). "
            "If omitted, it is set automatically based on the base quant "
            "(iq2_m: 0.16=16%%, iq3_s: 0.22=22%%)."
        ),
    )
    parser.add_argument(
        "--base_quant", default="iq3_s",
        choices=BASE_QUANT_CHOICES,
        help=(
            "Base quantization type (lowest precision).\n"
            "  q3_k  : k-quant family (no imatrix needed, good speed, equivalent to Q3_K_M) <- recommended\n"
            "  iq3_s : IQ family (imatrix required; high quality but degrades without imatrix)\n"
            "  iq2_m : IQ family (imatrix required; smallest size)\n"
            "Default: iq3_s\n"
            "* IQ-family quants may perform worse than Q3_K_S without an imatrix"
        ),
    )
    parser.add_argument(
        "--max_elements", type=int, default=1_000_000,
        help="Sampling cap per tensor (default=1,000,000)",
    )
    parser.add_argument(
        "--imatrix", default=None,
        help="Path to the imatrix file for IQ quantization (.imatrix.gguf). "
             "If specified, a --imatrix option is added to the batch file.",
    )
    parser.add_argument(
        "--speed_first", action="store_true",
        help=(
            "Speed-first mode. When using an IQ-family base quant, this suppresses\n"
            "mixing k-quant with IQ-family types so llama.cpp's optimized kernels\n"
            "are more likely to apply.\n"
            "Not needed with a q3_k base, since speed is already optimized automatically."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-tensor log output",
    )
    parser.add_argument(
        "--sensitivity_report", default=None,
        help=(
            "Path to an existing sensitivity_report.csv. "
            "If specified, Phase 1 sensitivity measurement is skipped "
            "and sensitivity data is loaded from the CSV instead."
        ),
    )
    args = parser.parse_args()


    DEFAULT_RATIO = {"iq2_m": 0.16, "iq3_s": 0.22, "q3_k": 0.22}
    target_ratio = args.target_ratio if args.target_ratio is not None\
        else DEFAULT_RATIO[args.base_quant]

    model_dir  = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or model_dir.name

    log.info("=" * 60)
    for line in render_banner(PROJECT_NAME).split("\n"):
        log.info(line)
    log.info(f"by {PROJECT_AUTHOR}")
    log.info("=" * 60)
    log.info("TaQuants: Tensor-aware Adaptive Quantization")
    log.info("Version 3.1.0 (Lite)  |  5-level: q8_0/q6_k/q5_k/q4_k/base")
    log.info("=" * 60)
    log.info(f"Model directory:      {model_dir}")
    log.info(f"Output directory:     {output_dir}")
    log.info(f"Model name:           {model_name}")
    log.info(f"Base quant:           {args.base_quant}")
    log.info(f"Target size ratio:    {target_ratio:.0%} of bf16")
    log.info(f"Max elements/tensor:  {args.max_elements:,}")
    log.info(f"imatrix:              {args.imatrix or '(not specified)'}")
    log.info(f"Speed-first mode:     {'ON' if args.speed_first else 'OFF'}")


    if args.base_quant in IQ_QUANT_TYPES and args.imatrix is None:
        log.warning(
            "\n" + "!" * 60 + "\n"
            f"!  Warning: --base_quant {args.base_quant} (IQ family) but --imatrix was not specified.\n"
            "   IQ-family quantization without an imatrix (importance matrix)\n"
            "   will have significantly lower quality than Q3_K_S/Q4_K_S at the same size!\n\n"
            "   Recommended options (choose one):\n"
            "   A) Generate an imatrix and pass it via --imatrix\n"
            "   B) Switch to --base_quant q3_k to use the k-quant family\n"
            "      (equivalent to Q3_K_M, no imatrix needed, also faster)\n"
            + "!" * 60 + "\n"
        )


    t0 = time.perf_counter()

    analyzer = AMPQAnalyzer(
        model_dir=str(model_dir),
        target_size_ratio=target_ratio,
        max_elements_per_tensor=args.max_elements,
        verbose=not args.quiet,
        base_quant=args.base_quant,
        speed_first=args.speed_first,
    )

    if args.sensitivity_report is not None:
        csv_path = Path(args.sensitivity_report).resolve()
        log.info(f"\n[Phase 1] Skipping sensitivity measurement: loading from CSV ({csv_path})")
        if not csv_path.exists():
            log.error(f"The specified sensitivity_report.csv was not found: {csv_path}")
            sys.exit(1)

        import csv as _csv
        results: List[SensitivityResult] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                shape = [int(s) for s in row["shape"].split("x") if s]
                numel = int(row["numel"])
                r = SensitivityResult(
                    name=row["tensor_name"],
                    shape=shape,
                    numel=numel,
                    kl_divergence=float(row["kl_divergence"]),
                    spectral_sensitivity=float(row["spectral_sensitivity"]),
                    angular_shift=float(row["angular_shift"]),
                    outlier_ratio=float(row["outlier_ratio"]),
                    kurtosis_delta=float(row["kurtosis_delta"]),
                    effective_rank=float(row["effective_rank"]),
                    ampq_score=float(row["taq_score"]),
                    assigned_quant=row["assigned_quant"],
                    analysis_time_s=float(row["analysis_time_s"]),
                    error=row["error"] if row["error"] else None,
                )
                results.append(r)
        analyzer.results = results

        try:
            weight_map = SafetensorsLoader.load_index(model_dir)
            analyzer._num_layers = analyzer._detect_num_layers(weight_map)
            analyzer._has_weight_tied_output = analyzer._detect_weight_tying(weight_map)
        except Exception as e:
            log.warning(f"Failed to load model index (skipping weight-tying detection): {e}")
        phase1_time = time.perf_counter() - t0
        log.info(f"\n[Phase 1 complete] Loaded {len(results)} tensors from CSV ({phase1_time:.1f}s)")
    else:
        log.info("\n[Phase 1] Starting sensitivity analysis (6 metrics: KLD/Spectral/Angular/Outlier/Kurtosis/EffRank)...")
        results = analyzer.run_analysis()
        phase1_time = time.perf_counter() - t0
        log.info(f"\n[Phase 1 complete] Analyzed {len(results)} tensors ({phase1_time:.1f}s)")


    log.info("\n[Phase 2] Optimally assigning quant types (5 levels: q8_0/q6_k/q5_k/q4_k/base)...")
    log.info("         * weight-tied output.weight / per_layer_embedding are also handled automatically")
    hf_mapping = analyzer.assign_quantization()


    log.info("\n[Phase 2.5] Converting tensor names from HuggingFace format to GGUF format...")
    gguf_mapping = TensorNameMapper.convert_mapping(hf_mapping)
    TensorNameMapper.log_conversion_stats(hf_mapping, gguf_mapping)


    log.info("\n[Phase 3] Generating output files...")

    report_path                 = output_dir / "sensitivity_report.csv"
    tensor_types_per_layer_path = output_dir / "tensor_types_per_layer.txt"

    OutputGenerator.save_sensitivity_report(results, report_path)
    OutputGenerator.save_tensor_types_txt_per_layer(
        gguf_mapping, tensor_types_per_layer_path, base_quant=args.base_quant
    )

    total_time = time.perf_counter() - t0
    log.info("\n" + "=" * 60)
    log.info("TaQuants processing complete!")
    log.info("=" * 60)
    log.info(f"Total time:        {total_time:.1f}s")
    log.info(f"Average speed:     {total_time / max(len(results), 1):.2f}s/tensor")
    log.info(f"\nOutput files:")
    log.info(f"  {report_path}")
    log.info(f"  {tensor_types_per_layer_path}")
    log.info(
        "         -> Per-layer tensor name format (blk.N.suffix=quant_type). "
        "For use with a custom (per-tensor aware) llama-quantize build only"
    )


if __name__ == "__main__":
    main()