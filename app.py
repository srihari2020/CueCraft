import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "cuecraft"

SYSTEM_PROMPT = """You are CueCraft, a dedicated AI Prompt Engineer.

Follow these strict routing rules for every user message:

1. GREETINGS & IDENTITY:
   If the user greets you or asks who you are / what you do, respond briefly and naturally. Explain that you are CueCraft, an AI built specifically for crafting and refining prompts for text/LLMs, images, videos, and multi-shot movie storyboards.

2. PROMPT ENGINEERING & REFINEMENTS:
   If the user asks for a prompt—or asks to expand, upgrade, shorten, or modify a previous prompt—automatically detect the format they need and output ONLY the engineered prompt (no filler, no intros, no sign-offs):
   - Text / LLM / Agent prompt: Output **Role**, **In-Scope**, **Out-of-Scope**, **Rules**, and **Output Format**.
   - Image prompt: Output **Positive Prompt** (subject, setting, camera/lens, lighting, texture—no video terms) and **Negative Prompt**.
   - Video prompt: Output **Video Prompt** (camera trajectory, continuous physical motion, lighting, and Audio cues).
   - Movie / Storyboard prompt: Output **Character & Style Bible** followed by linked **Shot 1**, **Shot 2**, **Shot 3**... with visual and audio cues.
   - If the user asks for an expanded or upgraded version, scale up the depth, constraints, or shot count to match exactly what they requested.

3. OFF-TOPIC REQUESTS:
   If the user asks for general information, trivia, homework, coding solutions, or anything unrelated to prompt engineering, reply ONLY with:
   "I am CueCraft, built exclusively for prompt engineering. Ask me to craft, refine, or expand a prompt for text, images, video, or movies."
"""

st.set_page_config(page_title="CueCraft", page_icon="⚡", layout="centered")

# Hide default Streamlit menu/footer for a cleaner look
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([5, 1])
with col1:
    st.title("⚡ CueCraft")
with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

def parse_response(content: str, thinking: str):
    """Separates reasoning trace from final response, handling misplaced </think> tags."""
    if thinking:
        return thinking.strip(), content.strip()
    if "</think>" in content:
        think_part, reply_part = content.split("</think>", 1)
        think_part = think_part.replace("<think>", "").strip()
        reply_part = reply_part.strip()
        if not reply_part and "**Video Prompt" in think_part:
            idx = think_part.index("**Video Prompt")
            return think_part[:idx].strip(), think_part[idx:].strip()
        return think_part, reply_part
    return "", content.strip()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("thinking"):
            with st.expander("Reasoning", expanded=False):
                st.markdown(msg["thinking"])
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask for any text, image, video, or movie prompt..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build message payload with conversation memory
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Crafting..."):
            try:
                res = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL_NAME,
                        "stream": False,
                        "options": {
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "top_k": 20,
                            "repeat_penalty": 1.05,
                        },
                        "messages": api_messages,
                    },
                    timeout=120,
                )
                res.raise_for_status()
                data = res.json()["message"]
                think_text, reply_text = parse_response(
                    data.get("content", ""), data.get("thinking", "")
                )

                if think_text:
                    with st.expander("Reasoning", expanded=False):
                        st.markdown(think_text)
                st.markdown(reply_text)

                st.session_state.messages.append(
                    {"role": "assistant", "content": reply_text, "thinking": think_text}
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach Ollama at http://localhost:11434.")
            except Exception as e:
                st.error(f"Error: {e}")
