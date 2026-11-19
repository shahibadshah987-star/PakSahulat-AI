import sys
from pathlib import Path
import streamlit as st

# Allow importing from src/
sys.path.append(str(Path(__file__).parent / "src"))

from agent_graph import graph
from memory_manager import get_user_facts, get_recent_turns

st.set_page_config(page_title="PakSahulat AI", page_icon="🇵🇰", layout="wide")

st.title("🇵🇰 PakSahulat AI — Sarkari Kaam Asaan")
st.caption("Grounded Multi-Agent Pakistani Government Services Assistant (PECA 2016/2025 • HEC • NADRA)")

# Sidebar: User Identity, Session & Persistent Memory Inspector
with st.sidebar:
    st.header("🧠 Agent Memory & Session")
    user_id = st.text_input("Citizen User ID (Long-Term Memory)", value="shan_peshawar")
    session_id = st.text_input("Session ID (Short-Term Memory)", value="session_01")

    st.divider()
    st.subheader("📌 Long-Term User Profile")
    st.info(get_user_facts(user_id))

    st.subheader("💬 Recent Session Turns")
    recent = get_recent_turns(session_id, limit=6)
    st.text(recent if recent else "No messages in this session yet.")

    st.divider()
    st.markdown("**Try Sample Queries:**")
    st.markdown("- *Compare PECA 2016 investigating agency with the 2025 Authority.*")
    st.markdown("- *NADRA FRC ke liye kya documents aur fee chahiye?*")
    st.markdown("- *How do I apply for HEC degree attestation?*")

# Initialize UI message state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous chat messages in the UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🔍 View Multi-Agent Trace & Grounded Sources"):
                st.markdown(f"**Router Decision:** Category = `{msg['trace']['category']}` | Version Filter = `{msg['trace']['version']}` | Rewritten Query = `{msg['trace']['query']}`")
                st.markdown(f"**Critic Fact-Check Verdict:** `{msg['trace']['critique']}`")
                st.markdown("**Retrieved Official Chunks:**")
                for idx, chunk in enumerate(msg["trace"]["context"], 1):
                    st.text_area(f"Source {idx}", chunk, height=110, disabled=True, key=f"{msg['id']}_src_{idx}")

# Chat input box
if prompt := st.chat_input("Ask in English, Urdu, or Roman Urdu..."):
    msg_id = len(st.session_state.messages)
    st.session_state.messages.append({"id": msg_id, "role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Router → Retriever → Writer → Critic collaborating..."):
            result = graph.invoke({
                "user_id": user_id,
                "session_id": session_id,
                "question": prompt,
            })

        answer = result["draft_answer"]
        trace_data = {
            "category": result.get("category", "N/A"),
            "version": result.get("version_filter", "N/A"),
            "query": result.get("standalone_query", prompt),
            "critique": result.get("critique", "APPROVED"),
            "context": result.get("context", []),
        }

        st.markdown(answer)
        with st.expander("🔍 View Multi-Agent Trace & Grounded Sources"):
            st.markdown(f"**Router Decision:** Category = `{trace_data['category']}` | Version Filter = `{trace_data['version']}` | Rewritten Query = `{trace_data['query']}`")
            st.markdown(f"**Critic Fact-Check Verdict:** `{trace_data['critique']}`")
            st.markdown("**Retrieved Official Chunks:**")
            for idx, chunk in enumerate(trace_data["context"], 1):
                st.text_area(f"Source {idx}", chunk, height=110, disabled=True, key=f"new_{msg_id}_src_{idx}")

    st.session_state.messages.append({
        "id": msg_id + 1,
        "role": "assistant",
        "content": answer,
        "trace": trace_data,
    })
    st.rerun()