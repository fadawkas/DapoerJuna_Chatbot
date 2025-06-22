import streamlit as st
from agent import build_agent
from memory import memory, remember
from tools import df
import random

# Page config
st.set_page_config("DAPOERJUNA", "🍳", layout="wide")

# UI header
st.markdown("""
<div style='font-size:40px;font-weight:700;margin-bottom:0.3em'>
🍳 DAPOERJUNA – Masakan Indonesia Gak Perlu Ribet
</div>
<p style='font-size:22px;margin-top:-10px'>
👨‍🍳 Saya <strong>Chef Juna</strong>. Gak bisa masak? Sini gue marahin—eh, ajarin maksudnya.
</p>
<p style='font-size:17px;color:#444'>
Tanya resep biar masakan enak, bingung mau masak apa, atau biar dimarahin Chef Juna?—semuanya bisa di sini!
</p>
""", unsafe_allow_html=True)

with st.expander("🤯 Bingung? Coba tanya kayak gini dulu:"):
    st.markdown("""
* _"Gimana sih cara bikin ayam geprek yang kriuk di luar, juicy di dalam?"_
* _"Berikan saya resep yang paling banyak disukai"_
* _"Chef Juna, aku mau resep makanan yang mudah dan cocok untuk diet vegan.!"_
""")

# Sidebar
with st.sidebar:
    st.header("Mood Chef Juna")
    chef_mood = st.selectbox("Pilih Mood:", ["Chef Juna Mengayomi 👼", "Chef Juna Galak 😈", "Random Mood 🎭"])
    if st.button("🗑️ Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.session_state.memory = memory
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_recipes_blob" not in st.session_state:
    st.session_state.last_recipes_blob = ""

# Mood map
mood_map = {
    "Chef Juna Mengayomi 👼": "baik",
    "Chef Juna Galak 😈": "galak",
    "Random Mood 🎭": "random"
}

# Display chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍🍳" if m["role"] == "assistant" else "🧑"):
        st.markdown(m["content"])

# Main input
if prompt := st.chat_input("Mau masak apa hari ini? Ketik aja…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="👨‍🍳"):
        with st.spinner("Chef Juna sedang mikir…"):
            remember("user", prompt)
            agent = build_agent()
            init_state = {
                "messages": [f"User: {prompt}"],
                "steps": 0,
                "attitude": mood_map[chef_mood],
            }
            out = agent.invoke(init_state, config={"max_loops": 6})
            reply = out["messages"][-1].split("</tool>")[-1].strip()
            remember("ai", reply)

            if "Langkah:" in reply:  # save blob for filtering
                st.session_state.last_recipes_blob = reply

        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
