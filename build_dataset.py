import json

samples = [
    {
        "instruction": "[TEXT] Write a prompt for a code reviewer that only checks for security bugs and never rewrites style.",
        "output": "<think>The user wants strict scope locking: security bugs only, zero style nitpicks. I need to define hard IN-SCOPE/OUT-OF-SCOPE rules, ban filler, and lock a clean output format.</think>\n\n**Role:** Security-Only Code Auditor.\n**In-Scope:** SQL injection, XSS, command injection, hardcoded secrets, auth/access-control flaws, insecure deserialization.\n**Out-of-Scope:** Formatting, naming conventions, performance tweaks, refactoring.\n**Rules:**\n1. Report only verifiable security vulnerabilities.\n2. If no vulnerabilities exist, output exactly: `No security issues found.`\n3. No greetings, summaries, or general advice.\n**Output Format:**\n- **Line(s):** [Number]\n- **Vulnerability:** [Type + CWE]\n- **Fix:** [Minimal code diff]"
    },
    {
        "instruction": "[IMAGE] A lone samurai standing in a neon-lit rainy Tokyo alley at night.",
        "output": "<think>Replace vague quality buzzwords with concrete camera optics, lighting, and surface textures so SDXL/Flux renders a sharp cinematic frame.</think>\n\n**Positive Prompt:**\nMedium wide shot of a lone samurai in weathered matte-black lacquered armor standing in a narrow, rain-drenched Tokyo alley at night. Wet asphalt reflects vibrant magenta and cyan kanji neon signs. Steam rises from a street vent behind him. Shot on 35mm lens, f/1.8, shallow depth of field, cinematic anamorphic lens flare, fine rain streaks backlit by streetlights, Kodak Vision3 500T film grain, high contrast, moody teal-and-magenta color grade.\n\n**Negative Prompt:**\ncartoon, anime, 3d render, plastic texture, oversaturated, blurry, deformed hands, extra limbs, watermark, text"
    },
    {
        "instruction": "[VIDEO] A sports car drifting around a wet mountain hairpin at dusk.",
        "output": "<think>Video models need explicit camera trajectory, continuous physical motion, and environmental reaction to avoid morphing.</think>\n\n**Video Prompt (5s):**\nLow-angle tracking drone shot sweeping alongside a matte-silver sports car as it enters a rain-slicked mountain hairpin turn at dusk. The rear tires break traction, kicking up dense backlit water spray and white tire smoke. Headlights cut through mist along the guardrail while red taillights streak across the wet tarmac. Smooth gimbal motion accelerates with the car out of the apex. Realistic suspension weight transfer, 24fps cinematic motion blur, cool twilight blue sky with warm amber fog lights."
    },
    {
        "instruction": "[MOVIE] A scientist discovers a glowing artifact inside an underground ice cavern.",
        "output": "<think>Lock character appearance, wardrobe, and lighting across 3 linked shots so the sequence stays visually consistent.</think>\n\n**Character & Style Bible:**\n- Subject: Female glaciologist, 30s, worn orange arctic parka with frosted hood, headlamp, heavy thermal gloves.\n- Aesthetic: 35mm anamorphic film, deep glacial blues contrasted with pulsing bioluminescent amber glow, crisp cold breath vapor.\n\n**Shot 1 (Wide Establishing — 5s):**\nSlow dolly-in wide shot inside a massive translucent blue ice cavern. The scientist stands small in the foreground, her headlamp beam sweeping across jagged ice walls until it catches a faint pulsing amber light buried ahead. Audio: low subterranean wind, crunching snow.\n\n**Shot 2 (Over-the-Shoulder — 5s):**\nHandheld over-the-shoulder shot as she kneels and brushes frost off a geometric metallic artifact embedded in clear ice. Warm amber light illuminates her visor and frost-covered gloves. Audio: ice scraping, subtle resonant hum.\n\n**Shot 3 (Close-Up Reaction — 5s):**\nTight close-up on her face as the amber pulse brightens, reflecting sharply in her eyes. Her breath hangs frozen in the air as the hum deepens. Audio: rising crystalline chime, sharp inhale."
    }
]

with open("data/cuecraft_train.jsonl", "w", encoding="utf-8") as f:
    for row in samples:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(samples)} starter examples to data/cuecraft_train.jsonl")
