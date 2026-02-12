// src/components/ChatList.tsx
import { useEffect, useRef } from 'react';
import { ChatMessage } from './ChatMessage';

interface Message {
  id: string;
  text: string;
  role: 'user' | 'rlfusion';
}

interface ChatListProps {
  messages: Message[];
  isLoading?: boolean;
  proactiveHint?: string;
  onSuggestionClick?: (suggestion: string) => void;
}

export default function ChatList({ messages, isLoading, proactiveHint, onSuggestionClick }: ChatListProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, proactiveHint]);

  // Only show suggestion after AI response and not loading
  const showSuggestion = !isLoading &&
    messages.length > 0 &&
    messages[messages.length - 1]?.role === 'rlfusion' &&
    proactiveHint &&
    proactiveHint !== 'Waiting for next query...';

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 pb-32 bg-[#0f0f12]">
      {messages.length === 0 && (
        <div className="max-w-2xl mx-auto mt-16 text-[var(--muted)]">
          <h2 className="text-3xl mb-2 font-bold text-[var(--accent)] text-center">
            RLFusion Orchestrator
          </h2>
          <p className="text-center text-gray-500 mb-8">Local-first AI with adaptive retrieval</p>

          <div className="space-y-5 text-sm leading-relaxed">
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
              <h3 className="text-[var(--accent)] font-semibold mb-2">How it works</h3>
              <p className="text-gray-400">
                Every question you ask gets routed through four retrieval paths — a vector search (RAG),
                a fast cache (CAG), a knowledge graph, and optionally the web. An RL policy decides how
                much to trust each source based on what's worked before. You can see the live weights
                shift in the panel on the right.
              </p>
            </div>

            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
              <h3 className="text-[var(--accent)] font-semibold mb-2">It gets better with use</h3>
              <p className="text-gray-400">
                On a fresh install, the system uses sensible defaults. But it learns from every
                conversation — each response is scored, and those scores train the policy that controls
                retrieval. After <span className="text-gray-300">100–500 interactions</span>, routing
                becomes noticeably tuned to your usage patterns. This is normal. Give it time.
              </p>
            </div>

            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
              <h3 className="text-[var(--accent)] font-semibold mb-2">Tips</h3>
              <ul className="text-gray-400 space-y-1.5 list-none">
                <li>Ask naturally — the system handles follow-ups and remembers context within a session.</li>
                <li>Watch the <span className="text-[var(--accent)]">Fusion Engine</span> panel to see which sources are being used.</li>
                <li>After each response, a 💡 suggestion may appear — click it to keep exploring.</li>
                <li>Say <span className="text-gray-300">"remember this:"</span> followed by a fact to save it to your profile.</li>
                <li>Switch to <span className="text-gray-300">Build</span> mode for multi-step planning tasks.</li>
              </ul>
            </div>

            <p className="text-center text-gray-600 text-xs pt-2">
              Everything runs locally. Nothing leaves your machine.
            </p>
          </div>
        </div>
      )}

      {messages.map((msg) => (
        <ChatMessage key={msg.id} text={msg.text} role={msg.role} />
      ))}

      {isLoading && (
        <div className="flex justify-start mb-6">
          <div className="bg-gray-800 px-5 py-3 rounded-2xl max-w-2xl border border-gray-700">
            <div className="text-xs opacity-70 mb-2 text-[var(--accent)]">RLFusion</div>
            <div className="flex gap-2">
              <span className="w-2 h-2 bg-[var(--accent)] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
              <span className="w-2 h-2 bg-[var(--accent)] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
              <span className="w-2 h-2 bg-[var(--accent)] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
            </div>
          </div>
        </div>
      )}

      {/* Proactive suggestion - appears after AI response */}
      {showSuggestion && (
        <div className="flex justify-start mb-6 mt-2">
          <button
            onClick={() => onSuggestionClick?.(proactiveHint)}
            className="group flex items-center gap-2 px-4 py-2.5 bg-gray-900/60 hover:bg-gray-800/80 border border-gray-700/50 hover:border-cyan-500/30 rounded-xl transition-all duration-200"
          >
            <span className="text-cyan-500/70 group-hover:text-cyan-400 text-sm">💡</span>
            <span className="text-gray-400 group-hover:text-gray-200 text-sm">{proactiveHint}</span>
            <span className="text-gray-600 group-hover:text-cyan-500/50 text-xs ml-1">→</span>
          </button>
        </div>
      )}
    </div>
  );
}
