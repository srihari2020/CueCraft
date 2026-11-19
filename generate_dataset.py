import json
import os
import random
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "cuecraft"
OUTPUT_FILE = "data/cuecraft_train.jsonl"

# Target counts per mode (Total: 520)
TARGETS = {"TEXT": 180, "IMAGE": 140, "VIDEO": 120, "MOVIE": 80}

SEEDS = {
    "TEXT": (
        ["SQL query generator", "JSON data extractor", "Python bug fixer", "regex builder",
         "API docs writer", "log anomaly triager", "commit message writer", "unit test generator",
         "meeting summarizer", "resume bullet optimizer", "legal clause extractor", "bash script helper"],
        ["only outputs raw code with zero explanations", "enforces strict JSON schema output",
         "rejects out-of-scope questions in one line", "never adds greetings or filler",
         "asks one clarifying question if ambiguous", "flags missing inputs instead of guessing",
         "uses bullet points max 12 words each", "outputs minimal diffs only"],
        ["for backend engineers", "for SOC analysts", "for DevOps pipelines", "for data scientists",
         "for frontend developers", "for technical writers"]
    ),
    "IMAGE": (
        ["a lone astronaut", "a weathered samurai", "a brutalist concrete library", "a vintage 1990s arcade",
         "a street food vendor", "a cyberpunk courier", "an ancient sunken temple", "a minimalist ceramic vase",
         "a snowy mountain observatory", "a bio-mechanical beetle", "a jazz saxophonist", "a rain-soaked diner"],
        ["in neon-lit Tokyo at night", "at golden hour in the desert", "inside a foggy pine forest",
         "under harsh fluorescent subway lights", "during a twilight thunderstorm", "in an overgrown greenhouse",
         "on a wet cobblestone alley", "in a sunlit minimalist studio"],
        ["35mm Kodak Vision3 film photography", "medium format editorial portrait", "macro studio photography",
         "architectural wide-angle photography", "cinematic anamorphic still", "moody chiaroscuro lighting"]
    ),
    "VIDEO": (
        ["a matte-black sports car", "a cybernetic owl", "a lone surfer", "a steam locomotive",
         "a hooded parkour runner", "a bioluminescent jellyfish", "a vintage fighter jet", "a street dancer",
         "a golden retriever", "a sleek humanoid robot"],
        ["drifting around a wet mountain hairpin", "swooping through a neon rainstorm",
         "weaving through a crowded night market", "gliding across a misty canyon",
         "sprinting across rain-slicked rooftops", "floating through a dark coral reef",
         "cutting through low coastal fog", "spinning under flickering streetlights"],
        ["at dusk", "at midnight", "at sunrise", "during a heavy downpour", "in thick winter snow"]
    ),
    "MOVIE": (
        ["a rogue hacker", "a deep-sea diver", "a lone astronaut", "an underground botanist",
         "a retired detective", "a young archaeologist", "a courier on a hoverbike", "a lighthouse keeper"],
        ["discovers a glowing encrypted drive", "escapes a flooded subway blackout",
         "finds an old cassette player in a derelict station", "unlocks a hidden vault beneath a library",
         "tracks a signal through an abandoned mall", "intercepts a mysterious package on a night train"],
        ["cyberpunk noir", "grounded sci-fi thriller", "moody neo-noir", "atmospheric mystery", "gritty survival drama"]
    ),
}

TEACHER_PROMPTS = {
    "TEXT": (
        "You are an expert prompt engineer. Convert the user's request into a strict, zero-drift system prompt.\n"
        "You MUST start with a brief <think>...</think> block, then output EXACTLY these sections:\n"
        "**Role:** ...\n**In-Scope:** ...\n**Out-of-Scope:** ...\n**Rules:**\n1. ...\n**Output Format:** ..."
    ),
    "IMAGE": (
        "You are an expert AI photography prompt engineer for SDXL/Flux. Convert the user's idea into a concrete, optical still-image prompt.\n"
        "You MUST start with a brief <think>...</think> block, then output EXACTLY these sections:\n"
        "**Positive Prompt:** (subject, setting, camera/lens like 35mm f/1.8, lighting, texture — NO video terms like fps, motion blur, or audio)\n"
        "**Negative Prompt:** (comma-separated artifacts to avoid)"
    ),
    "VIDEO": (
        "You are an expert AI video prompt engineer for Runway/Kling/Veo. Convert the user's idea into a single 5-second continuous-motion prompt.\n"
        "You MUST start with a brief <think>...</think> block, then output EXACTLY:\n"
        "**Video Prompt (5s):** (explicit camera trajectory, continuous physical motion, environmental reaction, lighting, and Audio cues). Do NOT split into multiple shots."
    ),
    "MOVIE": (
        "You are an expert cinematic storyboard prompt engineer. Convert the user's idea into a 3-shot linked movie sequence.\n"
        "You MUST start with a brief <think>...</think> block, then output EXACTLY these sections:\n"
        "**Character & Style Bible:** (fixed character look, wardrobe, and film aesthetic)\n"
        "**Shot 1 (Wide Establishing — 5s):** ... Audio: ...\n"
        "**Shot 2 (Medium/Tracking — 5s):** ... Audio: ...\n"
        "**Shot 3 (Close-Up/Reveal — 5s):** ... Audio: ..."
    ),
}

def is_valid(mode: str, text: str) -> bool:
    if "<think>" not in text or "</think>" not in text:
        return False
    low = text.lower()
    if mode == "TEXT":
        return "**role:**" in low and "**in-scope:**" in low and "**out-of-scope:**" in low
    if mode == "IMAGE":
        banned = ["24fps", "60fps", "motion blur", "dolly-in", "tracking shot", "audio:"]
        return "**positive prompt:**" in low and "**negative prompt:**" in low and not any(b in low for b in banned)
    if mode == "VIDEO":
        return "**video prompt" in low and "**shot 1" not in low
    if mode == "MOVIE":
        return "**character & style bible:**" in low and "**shot 1" in low and "**shot 2" in low and "**shot 3" in low
    return False

def build_instruction(mode: str) -> str:
    a, b, c = [random.choice(pool) for pool in SEEDS[mode]]
    if mode == "TEXT":
        return f"[TEXT] Write a prompt for a {a} {c} that {b}."
    if mode == "IMAGE":
        return f"[IMAGE] {a.capitalize()} {b}, {c}."
    if mode == "VIDEO":
        return f"[VIDEO] {a.capitalize()} {b} {c}."
    return f"[MOVIE] In a {c} style, {a} {b}."

def call_ollama(mode: str, instruction: str) -> str:
    payload = {
        "model": MODEL,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.95, "repeat_penalty": 1.05},
        "messages": [
            {"role": "system", "content": TEACHER_PROMPTS[mode]},
            {"role": "user", "content": instruction},
        ],
    }
    res = requests.post(OLLAMA_URL, json=payload, timeout=120)
    res.raise_for_status()
    msg = res.json()["message"]
    content = msg.get("content", "").strip()
    thinking = msg.get("thinking", "").strip()

    if thinking:
        return f"<think>\n{thinking}\n</think>\n\n{content}"
    if "</think>" in content and "<think>" not in content:
        return f"<think>\n{content}"
    return content

def main():
    os.makedirs("data", exist_ok=True)
    seen = set()
    counts = {m: 0 for m in TARGETS}

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                seen.add(row["instruction"])
                for m in TARGETS:
                    if row["instruction"].startswith(f"[{m}]"):
                        counts[m] += 1

    print("Current counts:", counts)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for mode, target in TARGETS.items():
            while counts[mode] < target:
                inst = build_instruction(mode)
                if inst in seen:
                    continue
                try:
                    out = call_ollama(mode, inst)
                    if is_valid(mode, out):
                        f.write(json.dumps({"instruction": inst, "output": out}, ensure_ascii=False) + "\n")
                        f.flush()
                        seen.add(inst)
                        counts[mode] += 1
                        print(f"[{mode}] {counts[mode]}/{target} saved")
                    else:
                        print(f"[{mode}] Rejected (format/bleed-through), retrying...")
                except Exception as e:
                    print(f"[{mode}] Error: {e}")

    print("Dataset generation complete:", counts)

if __name__ == "__main__":
    main()
