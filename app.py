import re
import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "cuecraft"

st.set_page_config(page_title="CueCraft Studio", page_icon="⚡", layout="wide")

st.title("⚡ CueCraft — Multi-Modal Prompt Architect")
st.caption("Local fine-tuned Qwythos-9B (Q4_K_M) for zero-drift [TEXT], [IMAGE], [VIDEO], and [MOVIE] prompts.")

# Sidebar controls
with st.sidebar:
    st.header("Generation Settings")
    temperature = st.slider("Temperature", 0.1, 1.2, 0.6, 0.05)
    top_p = st.slider("Top-P", 0.5, 1.0, 0.95, 0.05)
    repeat_penalty = st.slider("Repeat Penalty", 1.0, 1.3, 1.05, 0.01)
    st.divider()
    st.markdown("**Model:** `cuecraft` (Ollama)")
    st.markdown("**Modes:** `[TEXT]` `[IMAGE]` `[VIDEO]` `[MOVIE]`")

# Mode selection
mode = st.segmented_control(
    "Select Mode",
    options=["TEXT", "IMAGE", "VIDEO", "MOVIE"],
    default="TEXT",
)

placeholders = {
    "TEXT": "e.g., Build a prompt for a Python code reviewer that only flags security bugs and outputs minimal diffs.",
    "IMAGE": "e.g., A brutalist concrete library overgrown with bioluminescent moss at twilight.",
    "VIDEO": "e.g., A cybernetic owl swooping through a neon-lit rainstorm.",
    "MOVIE": "e.g., A lone astronaut finds an old cassette player floating inside a derelict space station.",
}

user_idea = st.text_area("Describe your idea", placeholder=placeholders[mode], height=110)

def parse_response(content: str, thinking: str):
    """Separates reasoning trace from final prompt, even if </think> is misplaced."""
    if thinking:
        return thinking.strip(), content.strip()

    # Standard <think>...</think> or ... </think>
    if "</think>" in content:
        think_part, prompt_part = content.split("</think>", 1)
        think_part = think_part.replace("<think>", "").strip()
        prompt_part = prompt_part.strip()

        # Handle [VIDEO] quirk where **Video Prompt appeared before </think>
        if not prompt_part and "**Video Prompt" in think_part:
            idx = think_part.index("**Video Prompt")
            return think_part[:idx].strip(), think_part[idx:].strip()

        return think_part, prompt_part

    return "", content.strip()

if st.button("Craft Prompt", type="primary", use_container_width=True):
    if not user_idea.strip():
        st.warning("Enter an idea first.")
    else:
        full_prompt = f"[{mode}] {user_idea.strip()}"
        with st.spinner(f"Crafting [{mode}] prompt..."):
            try:
                res = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL_NAME,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "top_p": top_p,
                            "repeat_penalty": repeat_penalty,
                        },
                        "messages": [{"role": "user", "content": full_prompt}],
                    },
                    timeout=120,
                )
                res.raise_for_status()
                msg = res.json()["message"]
                think_text, final_prompt = parse_response(
                    msg.get("content", ""), msg.get("thinking", "")
                )

                if think_text:
                    with st.expander("🧠 Reasoning Trace", expanded=False):
                        st.markdown(think_text)

                st.subheader(f"Engineered [{mode}] Prompt")
                st.markdown(final_prompt)

                with st.expander("📋 Copy Raw Markdown", expanded=True):
                    st.code(final_prompt, language="markdown")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach Ollama at http://localhost:11434. Make sure Ollama is running.")
            except Exception as e:
                st.error(f"Error: {e}")
