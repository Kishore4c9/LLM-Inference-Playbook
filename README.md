<div align="center">

# 🧠 LLM Inference Playbook

### *From your first API call to KV-cached multi-turn inference — a hands-on progression*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HF Inference API](https://img.shields.io/badge/Inference-Serverless%20API-blue?logo=huggingface)](https://huggingface.co/docs/api-inference/index)
[![Qwen2.5-7B](https://img.shields.io/badge/Model-Qwen2.5--7B--1M-ee3f24?logo=huggingface)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M)
[![Llama-2-7b](https://img.shields.io/badge/Model-Llama--2--7b-047db7?logo=meta)](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
[![Gemma-3-4b](https://img.shields.io/badge/Model-Gemma--3--4b-4285f4?logo=google)](https://huggingface.co/google/gemma-3-4b-it)
[![Transformers](https://img.shields.io/badge/Library-Transformers-ffd21e?logo=huggingface&logoColor=black)](https://github.com/huggingface/transformers)

</div>

---

## What is this?

Most LLM tutorials stop at "send a prompt, get a response." This playbook picks up where those leave off.

It's a **structured, code-first guide** to understanding how LLM inference actually works — not just how to call an API, but why stateless calls fail in conversation, how chat history changes model behaviour, and what KV caching does at the token level to make multi-turn inference fast and cheap.

**Who this is for:** ML engineers, backend developers, and curious practitioners who want to move from stitching together API calls to genuinely understanding the inference layer.

**What you'll come away with:**
- A working mental model of stateless vs. stateful inference
- Hands-on intuition for KV caching and why it matters for latency and cost
- Runnable code for both serverless (free tier) and local GPU/CPU deployments

---

## Prerequisites

Before you start, make sure you have the following:

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| A Hugging Face account + API token | [Get one here](https://huggingface.co/settings/tokens) — free tier is enough for `src/` scripts |
| CUDA 12.1 + compatible GPU | Required for `KV_Caching/` only; CPU will work but is very slow |
| ~14 GB VRAM | For running 7B models locally (e.g. Qwen2.5-7B, Llama-2-7b) |

> **No GPU?** The scripts in `src/` run entirely via Hugging Face's Serverless Inference API — no local hardware needed.

---

## Repository Structure

```
LLM-Inference-Playbook/
│
├── src/                        # Serverless inference scripts (start here)
│   ├── 01-Basic.py             # Minimal stateless inference loop
│   ├── 02-Basic_Pro.py         # Same, with streaming responses
│   └── 03-Chat_History.py      # Stateful multi-turn chat with history
│
├── KV_Caching/                 # Advanced: local inference with KV caching
│   ├── src/
│   │   └── chat_kv_cache.py    # Chat inference reusing cached K/V states
│   └── main.py                 # Entry point
│
├── files/                      # GIFs and supporting assets for this README
├── requirements.txt
└── README.md
```

---

## Learning Path

Follow these in order. Each script builds directly on the previous one.

```
① Basic Inference          →     ② Streaming             →     ③ Chat History          →     ④ KV Caching
   (stateless, minimal)           (real-time UX)                (stateful, multi-turn)         (token-level cache)
   src/01-Basic.py                src/02-Basic_Pro.py           src/03-Chat_History.py         KV_Caching/main.py
```

---

## The Core Insight: Stateless vs. Stateful

This is the most important thing to understand before writing production LLM code.

Take this three-turn conversation:

```
Q1: What is the capital of France?
Q2: How many airports are there in Paris?
Q3: How is the weather in the month of June?
```

**`01-Basic.py` — Stateless (no history)**

Each query is sent in isolation. On Q3, the model has no idea what city you're asking about:

> *"Could you clarify which city you're asking about?"*

![Stateless vs Stateful comparison](files/01-Basic.gif)

**`03-Chat_History.py` — Stateful (history maintained)**

The full conversation is passed on every turn. On Q3, the model correctly infers the context:

> *"Paris in June is typically warm and sunny, with average highs around 25°C..."*

![Chat history in action](files/03-Chat_History.gif)

The difference isn't magic — it's just whether you include previous messages in the `messages` array. But the downstream impact on UX and correctness is enormous.

---

## Advanced: KV Caching

### The problem KV caching solves

In standard autoregressive decoding, every time you generate a new token, the transformer recomputes attention over the **entire input sequence** — including all the tokens it already processed. In a multi-turn conversation, this means the model re-reads your entire chat history on every single token it generates.

This is wasteful. The key and value projections for past tokens don't change.

### What KV caching does

Instead of recomputing, the model stores the **key (K) and value (V) attention states** for processed tokens and reuses them on subsequent forward passes. Only *new* tokens require fresh computation.

```
Turn 1:   [Prompt tokens] → Compute K, V → Store in cache → Generate response
                                                ↓
Turn 2:   [New tokens only] → Compute K, V → Append to cache → Generate response
                                                ↓
Turn N:   [New tokens only] → Append → Generate  (past context costs near zero)
```

**The result:** Latency drops significantly in long conversations, and compute cost scales with *new* tokens only — not total conversation length.

`KV_Caching/src/chat_kv_cache.py` implements this using HuggingFace Transformers' `past_key_values` mechanism, giving you a working reference implementation you can adapt.

**Run it:**
```bash
python ./KV_Caching/main.py
```

---

## Setup & Installation

**1. Clone the repo**
```bash
git clone https://github.com/Kishore4c9/LLM-Inference-Playbook.git
cd LLM-Inference-Playbook
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> Note: `requirements.txt` installs PyTorch with CUDA 12.1 support. If you're on CPU only or a different CUDA version, replace the `--index-url` line with the appropriate one from [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

**4. Set your Hugging Face token**
```bash
export HF_TOKEN="hf_your_token_here"
```
Or create a `.env` file and load it — just don't commit it. The `.gitignore` already excludes `.env`.

---

## Running the Scripts

| Script | Command | Requires GPU? |
|---|---|---|
| Basic stateless inference | `python ./src/01-Basic.py` | No |
| Streaming responses | `python ./src/02-Basic_Pro.py` | No |
| Stateful chat with history | `python ./src/03-Chat_History.py` | No |
| KV-cached local inference | `python ./KV_Caching/main.py` | Yes (recommended) |

---

## Models Used

| Model | Provider | Used in | Inference type |
|---|---|---|---|
| [Qwen2.5-7B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M) | Alibaba | KV_Caching | Local |
| [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) | Meta | src/ | Serverless API |
| [Gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | Google | src/ | Serverless API |

Model access may require accepting terms on the Hugging Face model page before your token will work with them.

---

## Concepts Covered

| Concept | Where |
|---|---|
| Stateless inference | `src/01-Basic.py` |
| Streaming token output | `src/02-Basic_Pro.py` |
| Multi-turn chat with history | `src/03-Chat_History.py` |
| KV cache mechanics (theory) | This README |
| KV cache implementation | `KV_Caching/src/chat_kv_cache.py` |
| Context window management | `KV_Caching/` |
| Latency & compute trade-offs | `KV_Caching/` |

---

<div align="center">

*Built to make LLM inference less of a black box.*

</div>
