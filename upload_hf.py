
import os
from huggingface_hub import HfApi

REPO_ID = "Srihari2020/CueCraft-9B-GGUF"
GGUF_PATH = "CueCraft-GGUF_gguf/Qwythos-9B-Claude-Mythos-5-1M.Q4_K_M.gguf"
LORA_DIR = "CueCraft-LoRA"

MODEL_CARD = """---
base_model: empero-ai/Qwythos-9B-Claude-Mythos-5-1M
library_name: gguf
license: apache-2.0
tags:
  - prompt-engineering
  - unsloth
  - qlora
  - gguf
  - ollama
  - qwen3.5
---

# CueCraft-9B-GGUF

**CueCraft** is a local, multi-modal prompt engineering LLM fine-tuned from `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` using 4-bit QLoRA (Unsloth) on 520 curated, validated examples across `[TEXT]`, `[IMAGE]`, `[VIDEO]`, and `[MOVIE]` modes.

## Capabilities
- **`[TEXT]`** — Structured, production-grade LLM/agent system prompts and Master Prompts.
- **`[IMAGE]`** — Concrete optical photography, lighting, and texture prompts (Positive + Negative) for SDXL/Flux.
- **`[VIDEO]`** — Single-shot 5s continuous-motion prompts with camera trajectories, physics, and audio cues.
- **`[MOVIE]`** — Multi-shot cinematic storyboards with locked Character & Style Bibles and synced audio.

## Quickstart (Ollama)
Download `CueCraft-9B.Q4_K_M.gguf` and `Modelfile`, then run:

```bash
ollama create cuecraft -f Modelfile
ollama run cuecraft "[VIDEO] A cybernetic owl swooping through a neon-lit rainstorm."
