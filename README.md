# PakSahulat AI (Sarkari Kaam Asaan)

**PakSahulat AI** is a bilingual, multi-agent Retrieval-Augmented Generation (RAG) citizen assistant built to simplify Pakistani public service procedures across **NADRA**, **HEC**, and **PECA (2016 & 2025)**. Powered by **LangGraph**, **ChromaDB**, **Google Gemini**, **SQLite**, and **Streamlit**, it delivers grounded, citation-backed answers in **English**, **Urdu script**, and **Roman Urdu** while preventing hallucinations through a self-correcting **Writer-Critic** reflection loop.

---

## Multi-Agent Architecture

```mermaid
flowchart TD
    U["Citizen Query: EN / UR / Roman UR"] --> R["Node 1: Router and Rewriter"]
    R -->|"Category, Version, Query"| RET["Node 2: Filtered ChromaDB Retriever"]
    RET -->|"Top-K Chunks"| W["Node 3: Writer Agent"]
    W -->|"Draft Answer"| C["Node 4: Critic Fact-Checker"]
    C -->|"Needs Revision and Passes lt 2"| W
    C -->|"APPROVED or Max Passes"| M["Node 5: SQLite Memory"]
    M --> OUT["Streamlit UI and Agent Trace"]
