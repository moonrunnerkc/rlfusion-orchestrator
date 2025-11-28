# Author: Bradley R. Kinnard
# backend/core/profile.py
"""
User profile management - persistent facts about Brad that should be remembered
across all conversations.
"""

import sqlite3
from pathlib import Path
from backend.config import cfg


def get_user_profile() -> str:
    """
    Retrieve all user profile facts and format them as context.
    This gets injected into every conversation so Brad's info is always available.
    """
    db_path = Path(cfg["paths"]["db"])
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("""
        SELECT fact_key, fact_value, category FROM user_profile
        ORDER BY category, fact_key
        """)

        facts = cursor.fetchall()
        conn.close()

        if not facts:
            return ""

        # Format as context
        profile_text = "=== ABOUT BRAD ===\n"
        current_category = None

        for key, value, category in facts:
            if category != current_category:
                if current_category is not None:
                    profile_text += "\n"
                current_category = category
            profile_text += f"- {value}\n"

        return profile_text + "=================\n"

    except Exception as e:
        print(f"⚠️  Failed to load user profile: {e}")
        return ""


def update_user_fact(fact_key: str, fact_value: str, category: str = "general"):
    """
    Add or update a persistent fact about Brad.
    Called when user says things like "Remember that I..."
    """
    db_path = Path(cfg["paths"]["db"])
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""
        INSERT OR REPLACE INTO user_profile (fact_key, fact_value, category)
        VALUES (?, ?, ?)
        """, (fact_key, fact_value, category))
        conn.commit()
        conn.close()
        print(f"  🧠 Remembered: {fact_value} (category: {category})")
    except Exception as e:
        print(f"  ⚠️  Failed to save fact: {e}")


def detect_and_save_memory(query: str) -> tuple[bool, str | None]:
    """
    Detect if query is asking to remember something and extract the fact.
    Returns (True, memory_content) if a memory was saved, (False, None) otherwise.

    Explicit commands (handles large chunks):
    - "remember this: [any text]"
    - "remember that: [any text]"
    - "put this in memory: [any text]"
    - "save to memory: [any text]"
    - "store this: [any text]"

    Implicit patterns (natural language):
    - "remember that I [fact]"
    - "I like/love/prefer [thing]"
    - "my name is [name]"
    """
    import re
    from ollama import Client

    query_lower = query.lower().strip()

    # 1. Check for EXPLICIT memory commands (handles large text blocks)
    explicit_patterns = [
        r"^remember this:\s*(.+)$",
        r"^remember that:\s*(.+)$",
        r"^put this in memory:\s*(.+)$",
        r"^save to memory:\s*(.+)$",
        r"^store this:\s*(.+)$",
        r"^memory:\s*(.+)$"
    ]

    for pattern in explicit_patterns:
        match = re.match(pattern, query_lower, re.DOTALL)  # DOTALL = match newlines
        if match:
            content = query[match.start(1):match.end(1)].strip()  # Preserve original case
            # Store the entire chunk directly
            fact_key = f"user_note_{hash(content) % 100000}"  # Generate unique key
            update_user_fact(fact_key, content, category="notes")
            print(f"  💾 Stored {len(content)} chars to persistent memory")
            return (True, content)

    # 2. First filter out obvious QUESTIONS (not memory requests)
    question_patterns = [
        r"^(do|does|did|can|could|should|would|will|is|are|was|were|have|has|had)\s",
        r"^(what|where|when|why|how|who)\s",
        r"\?$"  # Ends with question mark
    ]
    for pattern in question_patterns:
        if re.search(pattern, query_lower):
            return (False, None)  # This is a question, not a memory request

    # 3. Check for implicit memory patterns - use LLM extraction
    memory_keywords = ["remember", "i like", "i love", "i hate", "i am", "my name is", "i prefer", "i enjoy"]
    if not any(keyword in query_lower for keyword in memory_keywords):
        return (False, None)

    # Use LLM to extract structured fact
    try:
        client = Client(host=cfg["llm"]["host"])

        extraction_prompt = f"""Analyze this statement and determine if Brad is asking you to remember a persistent fact about himself.

User statement: "{query}"

Context clues for TRUE memory requests:
- Explicitly says "remember that I..."
- States personal preferences: "I like/love/prefer/enjoy..."
- Declares identity: "I am...", "my name is..."
- NOT asking about something else (e.g., "Do you remember that movie?" is NOT a memory request)
- NOT using "remember" in a different context (e.g., "Remember when we talked about X?" is NOT asking to store a fact)

If this IS a memory request about Brad himself, respond with JSON:
{{"fact_key": "short_identifier", "fact_value": "the actual fact", "category": "preferences|identity|work|personality|general"}}

If this is NOT a memory request (asking about something else, or using "remember" in a different way), respond with:
{{"memory": false}}

Examples:
✅ "remember that I like dark humor" → {{"fact_key": "humor_style", "fact_value": "Likes dark humor", "category": "personality"}}
✅ "my name is Brad" → {{"fact_key": "name", "fact_value": "Brad", "category": "identity"}}
✅ "I prefer Python over JavaScript" → {{"fact_key": "language_preference", "fact_value": "Prefers Python over JavaScript", "category": "preferences"}}
❌ "Do you remember that bug we fixed?" → {{"memory": false}}
❌ "Remember when I asked about zebras?" → {{"memory": false}}
❌ "Can you remember the context from earlier?" → {{"memory": false}}

Respond ONLY with JSON:"""

        result = client.chat(
            model=cfg["llm"]["model"],
            messages=[{"role": "user", "content": extraction_prompt}],
            options={"temperature": 0.1, "max_tokens": 150}
        )

        content = result["message"]["content"].strip()

        # Parse JSON response
        import json
        parsed = json.loads(content)

        if parsed.get("memory") == False:
            return (False, None)

        # Save the fact
        if "fact_key" in parsed and "fact_value" in parsed:
            category = parsed.get("category", "general")
            fact_value = parsed["fact_value"]
            update_user_fact(parsed["fact_key"], fact_value, category)
            return (True, fact_value)

        return (False, None)

    except Exception as e:
        print(f"  ⚠️  Memory detection failed: {e}")
        return (False, None)
