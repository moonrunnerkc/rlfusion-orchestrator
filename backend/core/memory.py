# Author: Bradley R. Kinnard
# memory.py - conversation memory with entity tracking and query expansion
# Originally built for personal offline use, now open-sourced for public benefit.

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Set, Tuple

# Entity patterns - business, person, location, product
ENTITY_PATTERNS = [
    (r"(?:restaurant|cafe|bar|store|shop|hotel|place)\s+(?:called|named|is)\s+([A-Z][A-Za-z\s']+)", "business"),
    (r"([A-Z][A-Za-z\s']+(?:Restaurant|Cafe|Bar|Grill|Diner|Caboose|Kitchen|Bistro|Tavern))", "business"),
    (r"(?:person|guy|man|woman|someone)\s+named\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", "person"),
    (r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:who|that|lives|works)", "person"),
    (r"in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:,\s*[A-Z]{2})?)", "location"),
    (r"([A-Z][a-z]+,\s*[A-Z]{2})", "location"),
    (r"(?:the|a|an)\s+([A-Z][A-Za-z\s]+(?:5070|5090|RTX|GPU|Model))", "product"),
]

# Pronouns that need resolution
ANAPHORA_PATTERNS = [
    r"\b(it|its|it's)\b", r"\b(they|them|their|theirs)\b",
    r"\b(he|him|his|she|her|hers)\b", r"\b(this|that|these|those)\b",
    r"\bthe\s+(restaurant|place|business|person|thing|product|item)\b",
]


@dataclass
class ConversationTurn:
    role: str
    content: str
    timestamp: float
    entities: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    topics: Set[str] = field(default_factory=set)


@dataclass
class ConversationState:
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    active_entities: Dict[str, str] = field(default_factory=dict)
    topic_stack: List[str] = field(default_factory=list)
    last_query: str = ""
    last_response_summary: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def get_context_window(self, max_turns: int = 6) -> List[ConversationTurn]:
        return self.turns[-max_turns:] if self.turns else []


class ConversationMemory:
    def __init__(self, max_sessions: int = 1000, session_ttl: int = 3600):
        self._sessions: Dict[str, ConversationState] = {}
        self._lock = Lock()
        self._max_sessions = max_sessions
        self._session_ttl = session_ttl

    def get_or_create_session(self, session_id: str) -> ConversationState:
        with self._lock:
            self._cleanup_old_sessions()
            if session_id not in self._sessions:
                self._sessions[session_id] = ConversationState(session_id=session_id)
            return self._sessions[session_id]

    def _cleanup_old_sessions(self) -> None:
        now = time.time()
        expired = [sid for sid, state in self._sessions.items() if now - state.updated_at > self._session_ttl]
        for sid in expired:
            del self._sessions[sid]

        if len(self._sessions) > self._max_sessions:
            sorted_sessions = sorted(self._sessions.items(), key=lambda x: x[1].updated_at)
            for sid, _ in sorted_sessions[:len(self._sessions) - self._max_sessions]:
                del self._sessions[sid]

    def extract_entities(self, text: str) -> Dict[str, Set[str]]:
        entities: Dict[str, Set[str]] = defaultdict(set)
        for pattern, entity_type in ENTITY_PATTERNS:
            for match in re.findall(pattern, text, re.IGNORECASE):
                clean = match.strip().strip("'\".,;:")
                if len(clean) > 2:
                    entities[entity_type].add(clean)
        return dict(entities)

    def extract_topics(self, text: str) -> Set[str]:
        topics = set()
        for cap in re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text):
            if len(cap) > 3:
                topics.add(cap)
        domain_words = re.findall(r'\b(hours|menu|price|phone|website|address|location|schedule|contact|directions|reviews?|rating)\b', text, re.IGNORECASE)
        topics.update(w.lower() for w in domain_words)
        return topics

    def needs_expansion(self, query: str) -> bool:
        query_lower = query.lower()
        for pattern in ANAPHORA_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        return len(query.split()) <= 4

    def expand_query(self, session_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
        state = self.get_or_create_session(session_id)
        metadata = {"original_query": query, "expanded": False, "entities_used": [], "context_added": ""}

        if not self.needs_expansion(query) or (not state.active_entities and not state.topic_stack):
            return query, metadata

        query_lower = query.lower()
        context_pieces = []

        # business-related queries
        if any(w in query_lower for w in ['hours', 'open', 'close', 'menu', 'phone', 'website', 'address', 'price']):
            if 'business' in state.active_entities:
                context_pieces.append(f"for {state.active_entities['business']}")
                metadata["entities_used"].append(('business', state.active_entities['business']))
            if 'location' in state.active_entities:
                context_pieces.append(f"in {state.active_entities['location']}")
                metadata["entities_used"].append(('location', state.active_entities['location']))

        # person-related queries
        elif any(w in query_lower for w in ['who', 'person', 'contact', 'email', 'social']):
            if 'person' in state.active_entities:
                context_pieces.append(f"about {state.active_entities['person']}")
                metadata["entities_used"].append(('person', state.active_entities['person']))

        # fallback to most recent entity
        if not context_pieces and state.active_entities:
            entity_type, entity_value = list(state.active_entities.items())[-1]
            context_pieces.append(f"regarding {entity_value}")
            metadata["entities_used"].append((entity_type, entity_value))

        if context_pieces:
            context_str = " ".join(context_pieces)
            expanded = query.rstrip()[:-1] + " " + context_str + "?" if query.rstrip().endswith('?') else query + " " + context_str
            metadata["expanded"] = True
            metadata["context_added"] = context_str
            return expanded, metadata

        return query, metadata

    def add_turn(self, session_id: str, role: str, content: str) -> ConversationState:
        state = self.get_or_create_session(session_id)
        entities = self.extract_entities(content)
        topics = self.extract_topics(content)

        turn = ConversationTurn(role=role, content=content, timestamp=time.time(), entities=entities, topics=topics)
        state.turns.append(turn)

        for entity_type, entity_values in entities.items():
            if entity_values:
                state.active_entities[entity_type] = list(entity_values)[0]

        for topic in topics:
            if topic not in state.topic_stack:
                state.topic_stack.append(topic)
        state.topic_stack = state.topic_stack[-10:]

        state.updated_at = time.time()
        if role == 'user':
            state.last_query = content
        elif role == 'assistant':
            state.last_response_summary = content[:200] + "..." if len(content) > 200 else content

        return state

    def get_conversation_context(self, session_id: str, max_chars: int = 2000) -> str:
        state = self.get_or_create_session(session_id)
        if not state.turns:
            return ""

        context_parts = []
        if state.active_entities:
            entities_str = ", ".join(f"{k}: {v}" for k, v in state.active_entities.items())
            context_parts.append(f"[CONVERSATION CONTEXT]\nActive topics: {entities_str}")

        recent = state.get_context_window(4)
        if recent:
            turns_str = []
            for turn in recent:
                role_label = "User" if turn.role == "user" else "Assistant"
                content = turn.content[:300] + "..." if len(turn.content) > 300 else turn.content
                turns_str.append(f"{role_label}: {content}")
            context_parts.append("Recent conversation:\n" + "\n".join(turns_str))

        full_context = "\n\n".join(context_parts)
        return full_context[:max_chars] + "\n[...truncated]" if len(full_context) > max_chars else full_context

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    @classmethod
    def clear_all(cls) -> None:
        """Nuke every in-memory session. Used by /api/reset."""
        conversation_memory._sessions.clear()


# Global instance
conversation_memory = ConversationMemory()


def expand_query_with_context(session_id: str, query: str) -> Tuple[str, Dict[str, Any]]:
    return conversation_memory.expand_query(session_id, query)


def record_turn(session_id: str, role: str, content: str) -> None:
    conversation_memory.add_turn(session_id, role, content)


def get_context_for_prompt(session_id: str) -> str:
    return conversation_memory.get_conversation_context(session_id)


def clear_memory(session_id: str) -> None:
    conversation_memory.clear_session(session_id)
