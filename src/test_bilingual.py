from agent_graph import graph

test_cases = [
    {
        "label": "1. English Query",
        "session": "test_en",
        "question": "What are the categories and fees for a NADRA Family Registration Certificate (FRC)?",
    },
    {
        "label": "2. Roman Urdu Query",
        "session": "test_roman_ur",
        "question": "NADRA FRC ki kitni categories hain aur unki fees kya hai?",
    },
    {
        "label": "3. Urdu Script Query",
        "session": "test_urdu",
        "question": "نادرا فیملی رجسٹریشن سرٹیفکیٹ (FRC) کی اقسام اور فیس کیا ہے؟",
    },
]

for tc in test_cases:
    print(f"\n==================== {tc['label']} ====================")
    res = graph.invoke({
        "user_id": "bilingual_tester",
        "session_id": tc["session"],
        "question": tc["question"],
    })
    print(f"Rewritten Search Query: {res.get('standalone_query')}")
    print(f"Critic Verdict: {res.get('critique')}")
    print("\nAnswer:\n" + res["draft_answer"])