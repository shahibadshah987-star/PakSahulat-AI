import sys
from pathlib import Path
import streamlit as st

# Allow importing from src/
sys.path.append(str(Path(__file__).parent / "src"))

from agent_graph import graph
from memory_manager import get_user_facts, get_recent_turns

st.set_page_config(
    page_title="PakSahulat AI — Sarkari Kaam Asaan",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Emerald & Gold Civic UI Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&family=Playfair+Display:wght@600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0b1914;
        color: #eef5f1;
        font-family: 'Plus Jakarta Sans', 'Noto Nastaliq Urdu', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #07291c;
        border-right: 1px solid rgba(212, 175, 55, 0.28);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e6c665 !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        letter-spacing: 0.3px;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #d6e6de !important;
    }
    section[data-testid="stSidebar"] input {
        background-color: #0d3827 !important;
        color: #ffffff !important;
        border: 1px solid rgba(212, 175, 55, 0.35) !important;
        border-radius: 8px !important;
    }

    .hero-banner {
        background: linear-gradient(135deg, #073b27 0%, #0b4f35 55%, #062c1d 100%);
        border: 1px solid rgba(212, 175, 55, 0.45);
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.25rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
    }
    .hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.45rem;
        font-weight: 700;
        color: #e8c766;
        margin: 0;
        letter-spacing: 0.6px;
    }
    .hero-subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        font-weight: 500;
        color: #f3f7f5;
        margin-top: 0.35rem;
        letter-spacing: 0.4px;
    }
    .hero-subtitle span {
        color: #d4af37;
        margin: 0 8px;
    }
    .hero-badges {
        margin-top: 0.75rem;
        display: flex;
        justify-content: center;
        gap: 0.55rem;
        flex-wrap: wrap;
    }
    .civic-badge {
        background-color: rgba(6, 36, 24, 0.75);
        border: 1px solid rgba(212, 175, 55, 0.45);
        color: #e6cf87;
        padding: 0.22rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    div[data-testid="stChatMessage"] {
        background-color: #10241d;
        border: 1px solid rgba(212, 175, 55, 0.22);
        border-radius: 12px;
        padding: 0.95rem 1.1rem;
        margin-bottom: 0.75rem;
    }

    div[data-testid="stExpander"] {
        background-color: #0b1d17;
        border: 1px solid rgba(212, 175, 55, 0.25);
        border-radius: 10px;
    }

    .stButton > button {
        background-color: #0d3b29;
        color: #f0dfa8;
        border: 1px solid rgba(212, 175, 55, 0.55);
        border-radius: 999px;
        font-size: 0.83rem;
        font-weight: 600;
        padding: 0.42rem 0.9rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #14543b;
        color: #ffffff;
        border-color: #e8c766;
        box-shadow: 0 0 12px rgba(212, 175, 55, 0.25);
    }

    .demo-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #c7b26b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
    }

    div[data-testid="stChatInput"] textarea {
        background-color: #10261e !important;
        color: #f3f7f5 !important;
        border: 1px solid rgba(212, 175, 55, 0.45) !important;
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Hero Header Banner ---
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">PakSahulat AI</div>
        <div class="hero-subtitle"><span>━━</span> Sarkari Kaam Asaan <span>━━</span></div>
        <div class="hero-badges">
            <span class="civic-badge">🇵🇰 Grounded Multi-Agent RAG</span>
            <span class="civic-badge">PECA 2016 &amp; 2025</span>
            <span class="civic-badge">HEC Attestation</span>
            <span class="civic-badge">NADRA Identity &amp; FRC</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: User Identity, Session & Persistent Memory Inspector ---
with st.sidebar:
    st.header("🧠 Agent Memory & Session")
    import uuid
    if "visitor_id" not in st.session_state:
        st.session_state.visitor_id = uuid.uuid4().hex[:8]
    user_id = st.text_input("Citizen User ID (Long-Term Memory)", value="citizen_" + st.session_state.visitor_id)
    session_id = st.text_input("Session ID (Short-Term Memory)", value="session_" + st.session_state.visitor_id)

    if st.button("🧹 Clear Session Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("📌 Long-Term User Profile")
    facts = get_user_facts(user_id)
    st.info(facts if facts else "No saved profile facts yet.")

    st.subheader("💬 Recent Session Turns")
    recent = get_recent_turns(session_id, limit=6)
    st.text(recent if recent else "No messages in this session yet.")

# --- Initialize UI Message State ---
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_trace(trace_data: dict, prefix: str) -> None:
    """Safely render the multi-agent trace and grounded source chunks."""
    with st.expander("🔍 View Multi-Agent Trace & Grounded Sources"):
        st.markdown(
            f"**Router Decision:** Category = `{trace_data.get('category', 'N/A')}` | "
            f"Version Filter = `{trace_data.get('version', 'N/A')}` | "
            f"Rewritten Query = `{trace_data.get('query', '')}`"
        )
        st.markdown(f"**Critic Fact-Check Verdict:** `{trace_data.get('critique', 'APPROVED')}`")
        st.markdown("**Retrieved Official Chunks:**")
        chunks = trace_data.get("context", [])
        if not chunks:
            st.caption("No document chunks retrieved (Out-of-Scope or general greeting).")
        for idx, chunk in enumerate(chunks, 1):
            st.text_area(
                f"Source {idx}",
                chunk,
                height=110,
                disabled=True,
                key=f"{prefix}_src_{idx}",
            )


def handle_user_query(prompt_text: str) -> None:
    """Run the LangGraph pipeline and append the turn to session state."""
    msg_id = len(st.session_state.messages)
    st.session_state.messages.append({"id": msg_id, "role": "user", "content": prompt_text})

    with st.chat_message("user"):
        st.markdown(prompt_text)

    trace_data = None
    with st.chat_message("assistant"):
        with st.spinner("Router → Retriever → Writer → Critic collaborating..."):
            try:
                result = graph.invoke(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "question": prompt_text,
                    }
                )
                answer = result.get("draft_answer", "No response generated.")
                trace_data = {
                    "category": result.get("category", "N/A"),
                    "version": result.get("version_filter", "N/A"),
                    "query": result.get("standalone_query", prompt_text),
                    "critique": result.get("critique", "APPROVED"),
                    "context": result.get("context", []),
                }
            except Exception as exc:
                answer = (
                    "⚠️ Temporary service interruption while contacting the AI model. "
                    f"Please retry in a few seconds.\n\n*(Details: {exc})*"
                )

        st.markdown(answer)
        if trace_data is not None:
            render_trace(trace_data, prefix=f"new_{msg_id}")

    st.session_state.messages.append(
        {
            "id": msg_id + 1,
            "role": "assistant",
            "content": answer,
            "trace": trace_data,
        }
    )
    st.rerun()


# --- Render Previous Chat Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            render_trace(msg["trace"], prefix=str(msg["id"]))

# --- 1-Click Presentation Demo Chips ---
st.markdown('<div class="demo-label">⚡ 1-Click Demo Prompts</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

selected_demo = None
with col1:
    if st.button("HEC Degree (Urdu)"):
        selected_demo = "ایچ ای سی (HEC) سے ڈگری تصدیق کرانے کا آن لائن طریقہ اور فیس کیا ہے؟"
with col2:
    if st.button("NADRA FRC (Eng)"):
        selected_demo = "What are the categories, required documents, and official fee for a NADRA Family Registration Certificate (FRC)?"
with col3:
    if st.button("PECA 2025 (Roman)"):
        selected_demo = "PECA Amendment Act 2025 ke tehat fake news phailane ki kya saza aur jurmana hai?"
with col4:
    if st.button("Out of Scope Test"):
        selected_demo = "What is the current rate of petrol and 10-year passport renewal fee in Pakistan?"

# --- Handle Input from Either Demo Button or Chat Box ---
user_typed = st.chat_input("Ask in English, Urdu (اردو), or Roman Urdu...")

if selected_demo:
    handle_user_query(selected_demo)
elif user_typed:
    handle_user_query(user_typed)
