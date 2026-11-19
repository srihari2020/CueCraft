# CueCraft

**CueCraft** is a local, multi-modal prompt engineering LLM fine-tuned from `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` using 4-bit QLoRA (Unsloth) and exported to `Q4_K_M` GGUF for Ollama.

## Modes
- **`[TEXT]`** — Strict, zero-drift system prompts with hard `IN-SCOPE`/`OUT-OF-SCOPE` rules and locked output schemas.
- **`[IMAGE]`** — Concrete optical photography and lighting prompts (Positive + Negative) for SDXL/Flux.
- **`[VIDEO]`** — Single-shot 5s continuous-motion prompts with camera trajectories, physics, and audio cues.
- **`[MOVIE]`** — Multi-shot cinematic storyboards with a locked Character & Style Bible.

## Quickstart (Ollama)
```bash
ollama create cuecraft -f Modelfile
ollama run cuecraft "[VIDEO] A cybernetic owl swooping through a neon-lit rainstorm."
