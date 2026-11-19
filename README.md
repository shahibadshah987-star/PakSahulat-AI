# 🇵🇰 PakSahulat AI — Sarkari Kaam Asaan

A grounded, bilingual Multi-Agent AI Research & Citizen Services Assistant built with **LangGraph**, **ChromaDB**, **Google Gemini**, **SQLite**, and **Streamlit**. PakSahulat AI guides Pakistani citizens through official policies using verified government documents across Cyber Crime (**PECA 2016 & 2025 Amendment Act**), Higher Education (**HEC Degree Attestation Manual**), and National Identity (**NADRA FRC & CNIC Policies**).

---

## 🏗️ Multi-Agent Architecture (LangGraph)

1. **Router Agent (`router_node`)**: Classifies user intent (`Cyber_Fraud`, `Education`, `Identity`, or `Out_Of_Scope`), detects version comparison needs (`previous`, `recent`, or `both`), and rewrites context-dependent follow-ups using short-term memory.
2. **Versioned Retriever Agent (`retriever_node`)**: Queries a 129-chunk **ChromaDB** vector store (`gemini-embedding-001`) with metadata filtering (`version="previous"` vs. `version="recent"`, or split `both` retrieval) so historical and updated laws are compared accurately side by side.
3. **Memory-Aware Writer Agent (`writer_node`)**: Synthesizes grounded answers strictly matching the user's current language/script (**English**, **Urdu script**, or **Roman Urdu**), personalizes responses using SQLite long-term profile facts, and enforces inline bracketed citations (`[Document Title (v:version, p.X)]`).
4. **Critic Fact-Checker Agent (`critic_node`)**: Audits every draft response against retrieved chunks before delivery to prevent hallucinations and enforce citation integrity (triggering a self-correction loop if unsupported claims are detected).
5. **Memory Manager (`memory_node`)**: Persists conversation turns into SQLite **Short-Term Session Memory** and extracts durable citizen facts (e.g., city, degree, profession) into SQLite **Long-Term User Profile Memory**.

---

## 📁 Project Structure

```text
PakSahulat AI/
├── app.py                   # Interactive Streamlit Web UI with live agent traces & memory inspector
├── chroma_db/               # Persistent ChromaDB vector database (129 grounded chunks)
├── data/
│   ├── pak_sahulat_memory.db # SQLite database for short-term & long-term memory
│   └── eval_report.json      # Automated evaluation metrics & benchmark results
├── src/
│   ├── fetch_docs.py        # Official government PDF ingestion & preprocessing
│   ├── build_vector_db.py   # Chunking, metadata tagging, and ChromaDB indexing
│   ├── memory_manager.py    # SQLite dual-memory persistence & LLM fact extraction
│   ├── agent_graph.py       # 5-node LangGraph state machine with quota fallback chains
│   └── evaluate.py          # Automated RAG & agent evaluation suite
└── .env                     # Gemini API configuration