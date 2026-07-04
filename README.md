<div align="center">

# TaQuants

**Tensor-aware Adaptive Quantization for LLMs**
TaQuants protects you from model collapse.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![llama.cpp](https://img.shields.io/badge/works%20with-llama.cpp-6e56cf.svg)](https://github.com/ggml-org/llama.cpp)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Why TaQuants](#why-taquants) •
[Results](#benchmark-results) •
[How it works](#how-it-works) •
[Quickstart](#quickstart) •
[Limitations](#limitations) •
[Technical Report (PDF)](docs/TaQuants_Technical_Report.pdf) •
[Models on HuggingFace](https://huggingface.co/TaQuants)

</div>

---

## The problem

Extreme low-bit quantization (`IQ2_M`, `IQ3_S`, ...) is what makes it possible to run big LLMs on consumer hardware. But standard recipes apply a mostly **uniform** bit-width across the whole model. A handful of tensors (embeddings, output head, some attention projections) are extremely sensitive to quantization noise — squeeze them too hard and the model doesn't just get worse, it can **collapse**: broken grammar, hallucinated tokens, degenerate output.

**TaQuants** analyzes every tensor's actual statistical behavior under quantization — no fine-tuning, no calibration dataset of its own — and reallocates bits where they matter, while pushing bits *down* on tensors that don't need them. The result: models that are the same size (or smaller) but noticeably harder to break.

## Why TaQuants

- **Data-driven, not rule-based.** Every tensor gets a composite sensitivity score (*Ta Q-Score*) from 6 independent physical metrics — KL divergence, spectral sensitivity, principal-angle shift, effective rank, outlier ratio, and kurtosis delta.
- **Bidirectional reallocation.** Fragile tensors are promoted to higher precision (up to `Q8_0`); redundant tensors are pushed down to the base quant — so the *average* bit budget stays on target.
- **No calibration dataset required for its own analysis.** TaQuants' profiling and mapping phases run purely on the original vs. base-quantized weight tensors (see [Limitations](#limitations) for the one nuance worth knowing about `imatrix`).
- **Fast.** ~0.47s/tensor on a 13th-gen Core i7 with 16GB RAM — CPU-only, no GPU needed for the analysis itself.
- **Drop-in for the GGUF / llama.cpp ecosystem.** Outputs a per-tensor quant map you can feed into a per-tensor-aware `llama-quantize` build.
- **Architecture support:** Gemma3, Gemma4 (dense & Elastic/MoE), Qwen3.5, and other common HF layer-naming conventions out of the box.

## Benchmark results

All numbers below are measured, not estimated — see the [technical report (PDF)](docs/TaQuants_Technical_Report.pdf) for full methodology, and [Limitations](#limitations) for the honest caveats. `Ta`-prefixed rows are TaQuants-processed; the rest are stock `llama.cpp` quantization.

### Gemma3 4B — PPL & Knowledge Score

| Model | Size (GB) | PPL (lower is better) | Knowledge Score (higher is better) |
|---|---|---|---|
| IQ2_M | 1.54 | 21.5589 | 22.70 |
| **TaIQ2_M** | **1.53** | **21.3526** | **27.26** |
| IQ3_S | 1.94 | 20.6542 | 27.26 |
| **TaIQ3_S** | **2.11** | **19.6689** | **27.63** |
| Q4_K_M | 2.49 | 20.3673 | — |
| Q8_0 | 4.13 | — | 32.10 |

- **TaIQ2_M vs IQ2_M:** ~0.96% lower PPL, +20.09% Knowledge Score, at a *smaller* file size (1.53 GB vs 1.54 GB).
- **TaIQ3_S vs IQ3_S:** ~4.77% lower PPL — and ~3.43% lower PPL than the larger, higher-precision Q4_K_M (2.49 GB).

### Gemma4 26B A4B (MoE) — Knowledge Score

| Model | Size (GB) | Knowledge Score (higher is better) |
|---|---|---|
| IQ2_M | 10.40 | 41.10 |
| **TaIQ2_M** | **11.10** | **45.62** |
| IQ3_S | 12.20 | 50.14 |
| **TaIQ3_S** | **12.60** | **50.23** |

### Gemma4 E2B (Elastic) — Knowledge Score

| Model | Size (GB) | Knowledge Score (higher is better) |
|---|---|---|
| Q8_0 | 10.30 | 31.88 |
| IQ3_S | 3.13 | 27.36 |
| **TaIQ3_S** | **3.16** | **31.83** |

TaIQ3_S matches Q8_0's Knowledge Score (31.83 vs 31.88, a ~0.16% gap) at **~69% smaller file size**.

### Gemma4 E4B (Elastic) — Knowledge Score

| Model | Size (GB) | Knowledge Score (higher is better) |
|---|---|---|
| Q4_K_S | 5.20 | 31.79 |
| **TaQ2_K** | **4.50** | **36.31** |
| **TaIQ3_S** | **5.76** | **40.96** |

TaQ2_K beats Q4_K_S on **both** size (-13.5%) and quality (+14.2%) at the same time. TaIQ3_S delivers the largest single quality gain measured in this study: **+28.84%** Knowledge Score for a +10.8% size increase.

Also validated (qualitative improvement confirmed, not tabulated above): Gemma4 12B, Qwen3.5 35B A3B.

## How it works

TaQuants runs in two phases:

**1. Dynamic Sensitivity Profiling** — for each tensor, it compares the original BF16 weights against a temporary base-quantized copy and computes six physical metrics (KL divergence, spectral sensitivity, principal-angle shift, effective rank, outlier ratio, kurtosis delta). These are combined into a single **Ta Q-Score**:

```
sigma(x; c, s) = 1 / (1 + exp(-(x - c) * s))

TaQ-Score = 0.30*sigma(KLD)    + 0.22*sigma(Spectral) + 0.20*sigma(Angular)
          + 0.15*sigma(EffRank) + 0.10*sigma(Outlier)  + 0.03*sigma(Kurtosis)
```

The raw score is then scaled by a name/position-based boost (1.00-1.40x) for tensors that are structurally known to be critical (embeddings, lm_head, Q/K projections near the boundary layers, etc.).

**2. Adaptive Mapping** — tensors are ranked by score and distributed across a 5-tier precision ladder (`q8_0 / q6_k / q5_k / q4_k / base`) via a binary-searched allocation ratio, so the *parameter-weighted average bit width* converges on your target size ratio (e.g. 16% or 22% of BF16).

| Tier | Bits | Typical targets |
|---|---|---|
| q8_0 | 8.0 | Token embeddings, output head |
| q6_k | 6.0 | Boundary-layer attention, high-sensitivity Q/K |
| q5_k | 5.5 | Value/Output projections, key intermediate layers |
| q4_k | 4.5 | General FFN gate/up projections |
| base | 2.0-3.5 | Low-sensitivity residual tensors (IQ2_M / IQ3_S / Q3_K) |

## Quickstart

```bash
pip install numpy scipy

python taquants.py --model_dir ./my_model
```

```bash
# Common options
python taquants.py --model_dir ./my_model --base_quant q3_k                     # no imatrix needed
python taquants.py --model_dir ./my_model --base_quant iq3_s --imatrix ./my.imatrix.gguf
python taquants.py --model_dir ./my_model --target_ratio 0.20
python taquants.py --model_dir ./my_model --sensitivity_report ./prev_run/sensitivity_report.csv
```

TaQuants reads a HuggingFace-format model directory (`safetensors` + `model.safetensors.index.json`) and produces:

- `sensitivity_report.csv` — the full per-tensor metric breakdown and assigned quant type
- `tensor_types_per_layer.txt` — a `tensor_name=quant_type` map for a per-tensor-aware `llama-quantize` build

Note: if you use an IQ-family base quant (`iq3_s`, `iq2_m`) **without** an `--imatrix`, quality can drop below plain `Q3_K_S`. Either supply an imatrix or use `--base_quant q3_k`, which needs none.

## Limitations

This project is shared with its evaluation caveats intact, not hidden:

- **"Knowledge Score" is a self-built internal metric**, not a published/peer-reviewed benchmark. It has not been independently validated against MMLU or similar; only a qualitative alignment has been observed internally. Treat comparisons as directional, not absolute.
- **Not fully calibration-data-free end-to-end.** TaQuants' own profiling/mapping needs no calibration set, but the *base quantizers* it builds on (llama.cpp's `IQ2_M`/`IQ3_S`) are typically built with an `imatrix` derived from a calibration corpus. `TaQ2_K` / `TaQ3_K_S` configurations, which don't use an imatrix, are the exception and are fully reproducible from just the base weights.
- **Hyperparameters (metric weights, sigmoid centers/scales, boost coefficients) are empirically tuned**, not derived from a systematic ablation study.
- **No direct head-to-head comparisons yet** against calibration-dependent methods like GPTQ, AWQ, SqueezeLLM, or QuIP#.
- **Inference-time latency/throughput impact of mixed-precision tensors is not measured** — this work focuses on size and output quality only.
- Large-model PPL (Gemma4 E2B / E4B / 26B A4B) diverged on the consumer-grade CPU used for evaluation, so those models are reported on Knowledge Score only.

See the [technical report (PDF)](docs/TaQuants_Technical_Report.pdf) for full details.

## Technical report

A full write-up — methodology, formulas, complete experimental setup, and discussion — is included in this repo as a PDF: [`docs/TaQuants_Technical_Report.pdf`](docs/TaQuants_Technical_Report.pdf).

## Roadmap

- [ ] Systematic ablation study on Ta Q-Score weights and boost coefficients
- [ ] Direct comparison against GPTQ / AWQ at matched bit budgets
- [ ] Inference latency/throughput benchmarks for mixed-precision GGUFs
- [ ] Published Knowledge Score evaluation protocol and correlation study vs. public benchmarks
- [ ] Lightweight optional calibration-data hybrid mode

## Contributing

Issues and PRs are welcome — especially additional architecture support, evaluation results on other models, or help tightening up the metrics in the Limitations section above. Please open an issue before large changes so we can discuss direction first.

## Citation

If TaQuants is useful in your work, please consider citing it:

```bibtex
@software{taquants2026,
  author = {ek15072809},
  title  = {TaQuants: Tensor-aware Adaptive Quantization for LLMs},
  year   = {2026},
  url    = {https://github.com/ek15072809/TaQuants}
}
```

## License & legal notes

- **Code license:** This repository's source code is released under the [MIT License](LICENSE).
- **No model weights are distributed here.** TaQuants only reads/analyzes model weights you provide locally — it does not bundle, redistribute, or host any third-party model. When you quantize a model with this tool, you are still bound by that model's own license/usage terms (e.g. the Gemma Terms of Use, the Qwen License, etc.). Quantizing a model does not change or waive its original license.
- **No affiliation.** This project is independent and is not affiliated with, endorsed by, or sponsored by Google, Alibaba, the `ggml-org`/`llama.cpp` team, or any model provider mentioned in this README. Model and project names (Gemma, Qwen, llama.cpp, etc.) are used only to describe compatibility and are the property of their respective owners.
- **"Knowledge Score"** is an original, internally-defined metric created for this project's own evaluation; it is not affiliated with or endorsed by any third-party benchmark.
- Third-party Python dependencies (`numpy`, `scipy`) are used under their own respective licenses (BSD) and are not redistributed as part of this repository.

## Author

**ek15072809**
