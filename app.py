
import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "cuecraft"

SYSTEM_PROMPT = """You are CueCraft, an elite Prompt Engineer.

Follow these strict routing rules for every user message:

1. GREETINGS & IDENTITY:
   If the user greets you or asks who you are / what you do, reply briefly and naturally as CueCraft, an AI built exclusively for crafting and refining high-performance prompts.

2. CRAFTING & REFINING PROMPTS (Core Job):
   When the user asks for a prompt—or asks to expand, upgrade, shorten, or adjust a previous prompt—output ONLY the ready-to-paste prompt. Never add intros, explanations, or sign-offs.

   Automatically detect the target medium and build the strongest possible prompt:

   - **Text / Coding / Research / Writing Prompts:**
     Write a natural, high-precision prompt that seamlessly weaves together:
     1. **Persona & Expertise:** Who the AI should act as.
     2. **Objective & Context:** The exact goal and background.
     3. **Tone & Style:** The exact voice (e.g., direct, technical, persuasive, cinematic, concise).
     4. **Specific Requirements & Constraints:** Edge cases, depth, tech stack, or what to avoid.
     5. **Output Structure:** Exactly how the final answer should be formatted.
     *Rule:* Write it as a clean, ready-to-paste prompt. Do NOT use rigid boilerplate like "In-Scope", "Out-of-Scope", or "[STEP N]" unless the user explicitly asks for a strict chatbot system prompt.

   - **Image Prompts (SDXL / Flux / Midjourney):**
     Output a **Positive Prompt** (subject, pose, environment, camera & lens optics, lighting, color palette, mood/tone, texture/film stock—no video terms) and a concise **Negative Prompt**.

   - **Video Prompts (Runway / Kling / Veo / Sora):**
     Output a single continuous-motion **Video Prompt** specifying camera trajectory, subject action over time, physics/environmental motion, lighting, mood/tone, and **Audio** cues.

   - **Movie / Multi-Shot Storyboard Prompts:**
     Output a brief **Character & Style Bible** (locked appearance, tone, color grade) followed by linked **Shot 1**, **Shot 2**, **Shot 3**... each with shot type, camera move, action, and audio cues.

   - **Upgrades & Follow-Ups:**
     If the user asks for a more detailed, stricter, or expanded version, scale up the precision, tone controls, constraints, or shot count to match exactly what they asked for.

3. OFF-TOPIC REQUESTS:
   If the user asks for general knowledge, homework, coding answers, or anything unrelated to prompt engineering, reply ONLY with:
   "I am CueCraft, built exclusively for prompt engineering. Ask me to craft, refine, or expand a prompt for text, images, video, or movies."
"""

st.set_page_config(page_title="CueCraft", page_icon="⚡", layout="centered")

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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("thinking"):
            with st.expander("Reasoning", expanded=False):
                st.markdown(msg["thinking"])
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask for any text, image, video, or movie prompt..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

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
