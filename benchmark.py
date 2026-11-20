import time
import requests
import matplotlib.pyplot as plt
import numpy as np

OLLAMA_URL = "http://localhost:11434/api/chat"

# Update BASE_MODEL to match your exact `ollama list` name for Qwythos
BASE_MODEL = "hf.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF:Q4_K_M"
CUECRAFT_MODEL = "cuecraft"

TEST_SUITE = [
    ("TEXT", "[TEXT] Build a prompt for a SQL assistant that only outputs valid PostgreSQL queries."),
    ("TEXT", "[TEXT] Write a prompt for a JSON log extractor for SOC analysts that never adds filler."),
    ("IMAGE", "[IMAGE] A brutalist concrete library overgrown with bioluminescent moss, 35mm film photography."),
    ("IMAGE", "[IMAGE] A weathered samurai in a foggy pine forest at dawn, medium format editorial portrait."),
    ("VIDEO", "[VIDEO] A cybernetic owl swooping through a neon-lit rainstorm at midnight."),
    ("VIDEO", "[VIDEO] A matte-black sports car drifting around a wet mountain hairpin at dusk."),
    ("MOVIE", "[MOVIE] In a grounded sci-fi style, a lone astronaut finds an old cassette player in a derelict station."),
    ("MOVIE", "[MOVIE] In a cyberpunk noir style, a rogue hacker escapes a flooded subway blackout."),
]

def check_accuracy(mode: str, text: str) -> bool:
    low = text.lower()
    if mode == "TEXT":
        return any(k in low for k in ["**role:**", "**persona:**"]) and any(
            k in low for k in ["**rules:**", "**constraints:**", "**output format:**", "**output structure:**"]
        )
    if mode == "IMAGE":
        banned = ["24fps", "60fps", "motion blur", "dolly-in", "tracking shot", "audio:"]
        return "**positive prompt:**" in low and "**negative prompt:**" in low and not any(b in low for b in banned)
    if mode == "VIDEO":
        return "**video prompt" in low and "**shot 1" not in low and "audio:" in low
    if mode == "MOVIE":
        return ("**shot 1" in low and "**shot 2" in low and "**shot 3" in low)
    return False

def run_model(model_name: str):
    passed = 0
    speeds = []
    latencies = []

    print(f"\n--- Benchmarking: {model_name} ---")
    for mode, prompt in TEST_SUITE:
        t0 = time.time()
        res = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "stream": False,
                "options": {"temperature": 0.6, "top_p": 0.95, "num_predict": 1024},
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=180,
        )
        res.raise_for_status()
        data = res.json()
        elapsed = time.time() - t0

        content = data["message"].get("content", "")
        eval_count = data.get("eval_count", 0)
        eval_dur = data.get("eval_duration", 1) / 1e9  # ns to seconds
        tok_s = eval_count / eval_dur if eval_dur > 0 else 0

        ok = check_accuracy(mode, content)
        passed += int(ok)
        speeds.append(tok_s)
        latencies.append(elapsed)
        print(f"[{mode}] {'PASS' if ok else 'FAIL'} | {tok_s:.1f} tok/s | {elapsed:.1f}s")

    acc = (passed / len(TEST_SUITE)) * 100
    avg_speed = sum(speeds) / len(speeds)
    avg_lat = sum(latencies) / len(latencies)
    return {"accuracy": acc, "speed": avg_speed, "latency": avg_lat}

def plot_results(base_res, cue_res):
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    fig.patch.set_facecolor("#0e1117")

    models = ["Qwythos-9B (Base)", "CueCraft-9B (Fine-Tuned)"]
    colors = ["#6c757d", "#ff4b4b"]

    # Plot 1: Accuracy
    accs = [base_res["accuracy"], cue_res["accuracy"]]
    bars1 = ax1.bar(models, accs, color=colors, width=0.5)
    ax1.set_title("Mode Format & Zero-Bleed Accuracy (%)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylim(0, 115)
    ax1.set_facecolor("#0e1117")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{h:.1f}%", ha="center", fontweight="bold")

    # Plot 2: Speed
    spds = [base_res["speed"], cue_res["speed"]]
    bars2 = ax2.bar(models, spds, color=colors, width=0.5)
    ax2.set_title("Generation Speed (Tokens / sec — RTX 5060)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylim(0, max(spds) * 1.25 if max(spds) > 0 else 50)
    ax2.set_facecolor("#0e1117")
    ax2.grid(axis="y", linestyle="--", alpha=0.25)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + (max(spds) * 0.03), f"{h:.1f} tok/s", ha="center", fontweight="bold")

    fig.suptitle("CueCraft v1 vs. Base Qwythos-9B Benchmark", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig("benchmark_results.png", facecolor=fig.get_facecolor())
    print("\nSaved chart to benchmark_results.png")

if __name__ == "__main__":
    base_res = run_model(BASE_MODEL)
    cue_res = run_model(CUECRAFT_MODEL)

    print("\n=== FINAL SUMMARY ===")
    print(f"Base ({BASE_MODEL}):     Accuracy = {base_res['accuracy']:.1f}% | Speed = {base_res['speed']:.1f} tok/s | Avg Latency = {base_res['latency']:.1f}s")
    print(f"CueCraft ({CUECRAFT_MODEL}): Accuracy = {cue_res['accuracy']:.1f}% | Speed = {cue_res['speed']:.1f} tok/s | Avg Latency = {cue_res['latency']:.1f}s")

    plot_results(base_res, cue_res)
