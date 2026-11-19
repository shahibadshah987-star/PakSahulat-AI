import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from memory_manager import save_turn, get_recent_turns, save_user_fact, get_user_facts

load_dotenv(override=True)
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "").strip()

# Initialize LLMs across separate free-tier daily quota pools + catch 429 & 503
primary_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, max_retries=1)
fallback_1 = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0, max_retries=1)
fallback_2 = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0, max_retries=1)
fallback_3 = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, max_retries=1)

llm = primary_llm.with_fallbacks(
    [fallback_1, fallback_2, fallback_3],
    exceptions_to_handle=(Exception,),
)

def ask_llm(prompt: str) -> str:
    content = llm.invoke(prompt).content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        return "\n".join(parts).strip()
    return str(content).strip()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db",
    collection_name="sarkari_docs",
)

# 1. Shared Graph State (now includes user_id, session_id, and memory context)
class AgentState(TypedDict):
    user_id: str
    session_id: str
    question: str
    standalone_query: str
    category: str
    version_filter: str
    context: List[str]
    draft_answer: str
    critique: str
    revisions: int

# 2. Node 1: Router + Follow-Up Query Rewriter
def router_node(state: AgentState) -> dict:
    q = state["question"]
    history = get_recent_turns(state["session_id"], limit=4)

    prompt = f"""You are the Router for PakSahulat AI (Pakistani Government Services Assistant).
Recent Conversation:
{history if history else "None"}

Current User Message: {q}

1. Rewrite the message into a clear standalone search query using recent conversation context if needed.
2. Classify the category: Cyber_Fraud | Identity | Education | Out_Of_Scope
3. Choose document version: recent | previous | both

Reply in EXACTLY three lines:
QUERY: <standalone search query>
CATEGORY: <Cyber_Fraud | Identity | Education | Out_Of_Scope>
VERSION: <recent | previous | both>"""
    res = ask_llm(prompt)
    standalone_q, category, version = q, "Identity", "recent"
    for line in res.splitlines():
        if line.startswith("QUERY:"):
            standalone_q = line.split(":", 1)[1].strip()
        elif line.startswith("CATEGORY:"):
            category = line.split(":", 1)[1].strip()
        elif line.startswith("VERSION:"):
            version = line.split(":", 1)[1].strip().lower()

    print(f"[Router] Category={category} | Version={version} | Query='{standalone_q}'")
    return {
        "standalone_query": standalone_q,
        "category": category,
        "version_filter": version,
        "revisions": 0,
    }

# 3. Node 2: Retriever Agent
def retriever_node(state: AgentState) -> dict:
    if state["category"] == "Out_Of_Scope":
        return {"context": []}

    # Inside retriever_node:
    q = (state.get("standalone_query") or state["question"]).strip()
    if not q:
        return {"context": []}
    v = state["version_filter"]
    if v == "both":
        docs_prev = vectorstore.similarity_search(q, k=3, filter={"version": "previous"})
        docs_rec = vectorstore.similarity_search(q, k=3, filter={"version": "recent"})
        docs = docs_prev + docs_rec
    else:
        docs = vectorstore.similarity_search(q, k=6, filter={"version": v})
    formatted_chunks = []
    for d in docs:
        src = f"{d.metadata.get('title')} (v:{d.metadata.get('version')}, p.{d.metadata.get('page', 0) + 1})"
        formatted_chunks.append(f"[{src}]\n{d.page_content}")

    print(f"[Retriever] Fetched {len(formatted_chunks)} grounded chunks.")
    return {"context": formatted_chunks}

# 4. Node 3: Memory-Aware Writer Agent
def writer_node(state: AgentState) -> dict:
    if state["category"] == "Out_Of_Scope" or not state["context"]:
        return {
            "draft_answer": "I currently have verified documents for PECA (Cyber Crime 2016 & 2025), HEC Degree Attestation, and NADRA FRC/Identity policies. Please ask a question in those areas."
        }

    context_text = "\n\n---\n\n".join(state["context"])
    user_facts = get_user_facts(state["user_id"])
    history = get_recent_turns(state["session_id"], limit=4)
    feedback = f"\nFix this issue from reviewer: {state['critique']}" if state.get("critique") else ""

    prompt = f"""You are 'PakSahulat AI', a helpful Pakistani citizen assistant.
Answer the question using ONLY the provided official excerpts.
- Cite the document title and page number in brackets like [NADRA FRC & Identity Policy, p.1].
- Respond in the same language as the user (English, Urdu, or Roman Urdu).
- Personalize gently if relevant to saved user profile facts, without inventing policy rules.{feedback}

SAVED USER PROFILE (Long-Term Memory):
{user_facts}

RECENT CHAT (Short-Term Memory):
{history if history else "None"}

OFFICIAL EXCERPTS:
{context_text}

USER QUESTION: {state['question']}"""
    res = ask_llm(prompt)
    print("[Writer] Draft generated.")
    return {"draft_answer": res, "revisions": state.get("revisions", 0) + 1}

# 5. Node 4: Critic (Fact-Checker) Agent
def critic_node(state: AgentState) -> dict:
    if not state["context"]:
        return {"critique": "APPROVED"}

    context_text = "\n\n".join(state["context"])
    prompt = f"""Check if the DRAFT ANSWER is strictly supported by the EXCERPTS and includes citations.
If accurate and grounded, reply ONLY with: APPROVED
Otherwise, briefly state what is unsupported or missing citations.

EXCERPTS:
{context_text}

DRAFT ANSWER:
{state['draft_answer']}"""
    res = ask_llm(prompt)
    print(f"[Critic] Verdict: {res[:60]}")
    return {"critique": res}

# 6. Node 5: Persistent Memory Updater
def memory_node(state: AgentState) -> dict:
    save_turn(state["session_id"], "user", state["question"])
    save_turn(state["session_id"], "assistant", state["draft_answer"])

    fact_prompt = f"""Extract ONE durable user fact (city, university, degree, family status, or language preference) from this message if present.
If no personal fact is stated, reply ONLY with: NONE
Message: {state['question']}"""
    extracted = ask_llm(fact_prompt)
    if extracted.upper() != "NONE":
        save_user_fact(state["user_id"], extracted)
        print(f"[Memory] Saved long-term fact: {extracted}")
    return {}

def should_continue(state: AgentState) -> str:
    if state["critique"].startswith("APPROVED") or state["revisions"] >= 2:
        return "memory"
    return "writer"

# 7. Build and Compile the Graph
builder = StateGraph(AgentState)
builder.add_node("router", router_node)
builder.add_node("retriever", retriever_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)
builder.add_node("memory", memory_node)

builder.add_edge(START, "router")
builder.add_edge("router", "retriever")
builder.add_edge("retriever", "writer")
builder.add_edge("writer", "critic")
builder.add_conditional_edges("critic", should_continue, {"writer": "writer", "memory": "memory"})
builder.add_edge("memory", END)

graph = builder.compile()

if __name__ == "__main__":
    uid = "shan_peshawar"
    sid = "session_01"

    print("--- Turn 1 ---")
    r1 = graph.invoke({
        "user_id": uid,
        "session_id": sid,
        "question": "I live in Peshawar and need a NADRA FRC for my parents and siblings. Which category applies?",
    })
    print(r1["draft_answer"])

    print("\n--- Turn 2 (Follow-up using Short-Term & Long-Term Memory) ---")
    r2 = graph.invoke({
        "user_id": uid,
        "session_id": sid,
        "question": "What is the fee for that category and how can I apply online?",
    })
    print(r2["draft_answer"])