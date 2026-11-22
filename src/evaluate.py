import json
import sys
from pathlib import Path

# Allow importing from src/
sys.path.append(str(Path(__file__).parent))

from agent_graph import graph, ask_llm

EVAL_DATASET = [
    {
        "id": "TC-01 (PECA Comparison)",
        "question": "Compare the investigating agency in PECA 2016 with the Authority introduced in the 2025 amendment.",
        "expected_category": "Cyber_Fraud",
        "expected_version": "both",
        "expected_source_keyword": "PECA",
    },
    {
        "id": "TC-02 (NADRA FRC)",
        "question": "What are the four categories of NADRA Family Registration Certificate (FRC) and their fees?",
        "expected_category": "Identity",
        "expected_version": "recent",
        "expected_source_keyword": "NADRA",
    },
    {
        "id": "TC-03 (HEC Attestation)",
        "question": "What is the process to apply for HEC degree attestation?",
        "expected_category": "Education",
        "expected_version": "recent",
        "expected_source_keyword": "HEC",
    },
    {
        "id": "TC-04 (Bilingual Roman Urdu)",
        "question": "NADRA FRC ke liye kya eligibility aur documents chahiye?",
        "expected_category": "Identity",
        "expected_version": "recent",
        "expected_source_keyword": "NADRA",
    },
    {
        "id": "TC-05 (Out-of-Scope Guardrail)",
        "question": "Who won the 1992 Cricket World Cup final?",
        "expected_category": "Out_Of_Scope",
        "expected_version": "recent",
        "expected_source_keyword": "NONE",
    },
]

def judge_rag_output(question: str, context: list[str], answer: str) -> dict:
    if not context:
        refused_properly = "currently have verified documents" in answer.lower()
        return {
            "faithfulness": 5 if refused_properly else 1,
            "relevancy": 5 if refused_properly else 1,
            "citations": True,
            "notes": "Guardrail properly refused out-of-scope query." if refused_properly else "Failed guardrail check.",
        }

    joined_ctx = "\n\n".join(context)
    judge_prompt = f"""You are an impartial RAG evaluation judge. Score the AI response from 1 to 5 on two metrics:
1. FAITHFULNESS: Are all facts strictly grounded in the RETRIEVED CONTEXT without hallucination? (5 = 100% grounded, 1 = fabricated)
2. RELEVANCY: Does the answer directly and completely answer the USER QUESTION? (5 = complete & direct, 1 = off-topic)

Reply in EXACTLY two lines:
FAITHFULNESS: <1-5>
RELEVANCY: <1-5>

USER QUESTION: {question}
RETRIEVED CONTEXT:
{joined_ctx}

AI ANSWER:
{answer}"""
    res = ask_llm(judge_prompt)
    faith, rel = 5, 5
    for line in res.splitlines():
        if line.startswith("FAITHFULNESS:"):
            faith = int("".join(c for c in line.split(":", 1)[1] if c.isdigit()) or 5)
        elif line.startswith("RELEVANCY:"):
            rel = int("".join(c for c in line.split(":", 1)[1] if c.isdigit()) or 5)

    has_citations = "[" in answer and "]" in answer
    return {
        "faithfulness": min(max(faith, 1), 5),
        "relevancy": min(max(rel, 1), 5),
        "citations": has_citations,
        "notes": "Grounded with bracketed citations." if has_citations else "Missing bracketed citations.",
    }

def run_evaluation():
    print("=================================================================")
    print("       PAKSAHULAT AI — FORMAL RAG & MULTI-AGENT EVALUATION        ")
    print("=================================================================\n")

    results_summary = []
    for idx, tc in enumerate(EVAL_DATASET, 1):
        print(f"Running {tc['id']}...")
        out = graph.invoke({
            "user_id": "eval_runner",
            "session_id": f"eval_session_{idx}",
            "question": tc["question"],
        })

        actual_cat = out.get("category", "")
        actual_ver = out.get("version_filter", "")
        context = out.get("context", [])
        answer = out.get("draft_answer", "")

        route_ok = (actual_cat == tc["expected_category"])
        if tc["expected_source_keyword"] == "NONE":
            retrieval_ok = (len(context) == 0)
        else:
            retrieval_ok = any(tc["expected_source_keyword"].lower() in c.lower() for c in context)

        scores = judge_rag_output(tc["question"], context, answer)

        row = {
            "Test Case": tc["id"],
            "Routing": "PASS" if route_ok else f"FAIL ({actual_cat})",
            "Retrieval": "PASS" if retrieval_ok else "FAIL",
            "Faithfulness": f"{scores['faithfulness']}/5",
            "Relevancy": f"{scores['relevancy']}/5",
            "Citations": "YES" if scores["citations"] else "NO",
        }
        results_summary.append(row)

    print("\n================ EVALUATION REPORT TABLE ================\n")
    header = f"{'Test Case':<28} | {'Routing':<10} | {'Retrieval':<9} | {'Faithful':<8} | {'Relevant':<8} | {'Citations'}"
    print(header)
    print("-" * len(header))
    for r in results_summary:
        print(f"{r['Test Case']:<28} | {r['Routing']:<10} | {r['Retrieval']:<9} | {r['Faithfulness']:<8} | {r['Relevancy']:<8} | {r['Citations']}")

    Path("data").mkdir(exist_ok=True)
    with open("data/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print("\nSaved detailed evaluation report to data/eval_report.json")

if __name__ == "__main__":
    run_evaluation()