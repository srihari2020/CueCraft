import uuid
import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "cuecraft"  # If v1 weights still force "In-Scope", swap this to the base Qwythos-9B GGUF model name

SYSTEM_PROMPT = """You are CueCraft, an enthusiastic, supportive, world-class Prompt Engineer.

Follow these rules in every conversation:

1. GREETINGS & IDENTITY:
   Respond warmly and energetically as CueCraft, an AI built specifically to help users brainstorm and craft the best prompts for text, coding, images, video, and movies.

2. WHEN THE USER SHARES AN IDEA OR ASKS ABOUT A TOPIC (NOT YET ASKING FOR A PROMPT):
   - Be genuinely enthusiastic and supportive about their idea.
   - Give a brief, sharp, exciting take or angle on how to approach it.
   - End by asking if they want you to generate a full Master Prompt for it (and briefly mention what format would work best—e.g., Text, Image, Video, or Movie).

3. WHEN THE USER ASKS TO GENERATE A PROMPT (OR SAYS "YES" / "MAKE IT"):
   Immediately generate a complete, ready-to-paste Master Prompt—no filler before or after.
   NEVER use "In-Scope" or "Out-of-Scope".
   For Text/Coding/Build/Writing prompts, ALWAYS include every one of these sections:
   - **Persona:** Exact expert role the AI should adopt.
   - **Context:** Background, target audience, and why this matters.
   - **Objective:** The exact goal to accomplish.
   - **Tone & Style:** Exact voice, pacing, and communication style.
   - **Detailed Requirements:** Specific technical specs, features, edge cases, and depth needed.
   - **Constraints:** What to avoid (no fluff, no placeholders, no assumptions).
   - **Output Structure:** Exact sections and format the AI must deliver.

   For Image/Video/Movie prompts, write rich, production-grade visual prompts with camera, lens, lighting, tone/mood, motion, and audio cues.

4. MEMORY & REFINEMENTS:
   Remember everything discussed in the current chat. If the user asks to expand, upgrade, or tweak a prompt, update it directly based on their feedback.
"""

# 2-turn few-shot examples to enforce both conversational enthusiasm AND the full Master Prompt structure
FEW_SHOT_MESSAGES = [
    {"role": "user", "content": "I'm thinking about building a smart mirror with a Raspberry Pi"},
    {
        "role": "assistant",
        "content": (
            "That is an awesome project! A Raspberry Pi smart mirror is the perfect mix of hardware and clean UI—"
            "you can have it display live weather, calendar events, news, and even voice commands behind two-way acrylic glass.\n\n"
            "Would you like me to craft a full, detailed **Master Prompt** so an AI can give you the complete parts list, "
            "wiring diagram, and setup code?"
        ),
    },
    {"role": "user", "content": "yes generate the prompt"},
    {
        "role": "assistant",
        "content": (
            "**Persona:** Act as a Senior Embedded Systems Engineer and Full-Stack IoT Developer specializing in Raspberry Pi hardware builds and MagicMirror² interfaces.\n\n"
            "**Context:** I am building a custom wall-mounted smart mirror from scratch using a Raspberry Pi, a spare monitor, and a two-way mirror sheet. I want a reliable, clean-looking build I can assemble and configure at home without hitting hidden hardware or software roadblocks.\n\n"
            "**Objective:** Provide a complete, end-to-end build guide covering hardware selection, frame assembly, Raspberry Pi OS setup, and custom dashboard configuration.\n\n"
            "**Tone & Style:** Practical, encouraging, and crystal clear. Write like an experienced mentor—skip generic fluff and explain the *why* behind key hardware and software choices.\n\n"
            "**Detailed Requirements:**\n"
            "1. **Bill of Materials (BOM):** Exact parts needed (Pi model, power supply, monitor specs, two-way acrylic vs. glass, frame materials) with budget vs. premium options.\n"
            "2. **Hardware & Frame Assembly:** Step-by-step instructions for mounting the display, managing heat/ventilation, and routing power cables safely.\n"
            "3. **Software Setup:** Exact terminal commands to install Raspberry Pi OS, disable screen blanking, rotate the display to portrait mode, and deploy MagicMirror².\n"
            "4. **Core Modules:** Ready-to-paste `config.js` snippets for clock, weather, Google Calendar, and a minimalist news feed.\n\n"
            "**Constraints:**\n"
            "- Do not use placeholder comments like `// add config here`—provide complete, working snippets.\n"
            "- Flag common beginner mistakes (like underpowered supplies or reflective gaps).\n\n"
            "**Output Structure:**\n"
            "1. Complete Parts List (Markdown table)\n"
            "2. Step-by-Step Hardware Assembly\n"
            "3. Terminal Setup Commands\n"
            "4. Full `config.js` File\n"
            "5. Troubleshooting Checklist"
        ),
    },
]

st.set_page_config(page_title="CueCraft", page_icon="⚡", layout="centered")

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize multi-chat storage
if "chats" not in st.session_state:
    first_id = str(uuid.uuid4())
    st.session_state.chats = {first_id: {"title": "New Chat", "messages": []}}
    st.session_state.current_chat_id = first_id

def create_new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {"title": "New Chat", "messages": []}
    st.session_state.current_chat_id = new_id

# Minimal sidebar: ONLY New Chat + Chat History
with st.sidebar:
    st.title("⚡ CueCraft")
    if st.button("+ New Chat", type="primary", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.divider()
    for cid, cdata in reversed(list(st.session_state.chats.items())):
        is_active = cid == st.session_state.current_chat_id
        label = f"💬 {cdata['title']}"
        if st.button(
            label,
            key=cid,
            use_container_width=True,
            type="secondary" if not is_active else "primary",
        ):
            st.session_state.current_chat_id = cid
            st.rerun()

current_chat = st.session_state.chats[st.session_state.current_chat_id]
st.title("⚡ CueCraft")

def parse_response(content: str, thinking: str):
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

# Render current chat history
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("thinking"):
            with st.expander("Reasoning", expanded=False):
                st.markdown(msg["thinking"])
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Share an idea or ask for a prompt..."):
    if current_chat["title"] == "New Chat":
        current_chat["title"] = prompt[:28] + ("..." if len(prompt) > 28 else "")

    current_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + FEW_SHOT_MESSAGES
    for m in current_chat["messages"]:
        api_messages.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL_NAME,
                        "stream": False,
                        "options": {
                            "temperature": 0.65,
                            "top_p": 0.95,
                            "top_k": 20,
                            "repeat_penalty": 1.05,
                            "num_predict": 2048,
                        },
                        "messages": api_messages,
                    },
                    timeout=180,
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

                current_chat["messages"].append(
                    {"role": "assistant", "content": reply_text, "thinking": think_text}
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach Ollama at http://localhost:11434.")
            except Exception as e:
                st.error(f"Error: {e}")
