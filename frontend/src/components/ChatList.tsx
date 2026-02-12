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
        <div className="text-center mt-32 text-[var(--muted)]">
          <h2 className="text-3xl mb-4 font-bold text-[var(--accent)]">
            RLFusion Orchestrator
          </h2>
          <p className="text-lg text-[var(--muted)]">System ready. Ask me anything.</p>
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
