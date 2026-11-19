import sqlite3
from pathlib import Path

DB_PATH = Path("data/pak_sahulat_memory.db")

def init_memory_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        # Short-term session chat history
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Long-term user facts across sessions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, fact)
            )
        """)

def save_turn(session_id: str, role: str, message: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, role, message) VALUES (?, ?, ?)",
            (session_id, role, message),
        )

def get_recent_turns(session_id: str, limit: int = 6) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, message FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    rows.reverse()
    return "\n".join(f"{r.upper()}: {m}" for r, m in rows)

def save_user_fact(user_id: str, fact: str):
    fact = fact.strip()
    if not fact or fact.upper() == "NONE":
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_memory (user_id, fact) VALUES (?, ?)",
            (user_id, fact),
        )

def get_user_facts(user_id: str) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT fact FROM user_memory WHERE user_id = ? ORDER BY updated_at DESC LIMIT 10",
            (user_id,),
        ).fetchall()
    return "\n".join(f"- {r[0]}" for r in rows) if rows else "No saved user preferences yet."

# Initialize tables immediately on import
init_memory_db()

if __name__ == "__main__":
    save_user_fact("citizen_1", "Lives in Peshawar and prefers concise step-by-step answers.")
    save_turn("session_demo", "user", "How do I get an FRC?")
    save_turn("session_demo", "assistant", "You can apply via Pak-Identity or at a NADRA center.")
    print("--- Long-Term Memory ---")
    print(get_user_facts("citizen_1"))
    print("\n--- Short-Term Memory ---")
    print(get_recent_turns("session_demo"))