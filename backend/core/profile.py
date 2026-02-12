# Author: Bradley R. Kinnard
# profile.py - persistent user profile management
# Originally built for personal offline use, now open-sourced for public benefit.

import json
import re
import sqlite3
from pathlib import Path
from ollama import Client
from backend.config import cfg, PROJECT_ROOT

EXPLICIT_PATTERNS = [
    r"^remember this:\s*(.+)$",
    r"^remember that:\s*(.+)$",
    r"^put this in memory:\s*(.+)$",
    r"^save to memory:\s*(.+)$",
    r"^store this:\s*(.+)$",
    r"^memory:\s*(.+)$"
]

QUESTION_PATTERNS = [
    r"^(do|does|did|can|could|should|would|will|is|are|was|were|have|has|had)\s",
    r"^(what|where|when|why|how|who)\s",
    r"\?$"
]

MEMORY_KEYWORDS = ["remember", "i like", "i love", "i hate", "i am", "my name is", "i prefer", "i enjoy"]


def _get_db_path() -> Path:
    """Get the database path, ensuring parent directories exist."""
    db_path = PROJECT_ROOT / cfg["paths"]["db"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_user_profile() -> str:
    """Load and format the user's stored profile facts."""
    db_path = _get_db_path()
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT fact_key, fact_value, category FROM user_profile ORDER BY category, fact_key")
        facts = cursor.fetchall()
        conn.close()

        if not facts:
            return ""

        profile_text = "=== USER PROFILE ===\n"
        current_category = None
        for _, value, category in facts:
            if category != current_category:
                if current_category is not None:
                    profile_text += "\n"
                current_category = category
            profile_text += f"- {value}\n"
        return profile_text + "=================\n"

    except Exception as e:
        # Database may not exist yet - that's fine
        return ""


def update_user_fact(fact_key: str, fact_value: str, category: str = "general"):
    """Store or update a user fact in the database."""
    db_path = _get_db_path()
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                fact_key TEXT PRIMARY KEY,
                fact_value TEXT,
                category TEXT
            )
        """)
        conn.execute("INSERT OR REPLACE INTO user_profile (fact_key, fact_value, category) VALUES (?, ?, ?)",
                    (fact_key, fact_value, category))
        conn.commit()
        conn.close()
    except Exception:
        pass


def detect_and_save_memory(query: str) -> tuple[bool, str | None]:
    query_lower = query.lower().strip()

    # explicit memory commands
    for pattern in EXPLICIT_PATTERNS:
        match = re.match(pattern, query_lower, re.DOTALL)
        if match:
            content = query[match.start(1):match.end(1)].strip()
            fact_key = f"user_note_{hash(content) % 100000}"
            update_user_fact(fact_key, content, category="notes")
            return (True, content)

    # filter out questions
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, query_lower):
            return (False, None)

    # check for memory keywords
    if not any(kw in query_lower for kw in MEMORY_KEYWORDS):
        return (False, None)

    # LLM extraction for implicit memory
    try:
        client = Client(host=cfg["llm"]["host"])
        prompt = f"""Is this a request to remember a fact about the user?
"{query}"

If YES, respond: {{"fact_key": "identifier", "fact_value": "the fact", "category": "preferences|identity|work|personality|general"}}
If NO, respond: {{"memory": false}}
JSON only:"""

        result = client.chat(
            model=cfg["llm"]["model"],
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "max_tokens": 150}
        )

        parsed = json.loads(result["message"]["content"].strip())
        if parsed.get("memory") == False:
            return (False, None)

        if "fact_key" in parsed and "fact_value" in parsed:
            update_user_fact(parsed["fact_key"], parsed["fact_value"], parsed.get("category", "general"))
            return (True, parsed["fact_value"])

        return (False, None)

    except Exception as e:
        print(f"  ⚠️ Memory detection failed: {e}")
        return (False, None)
