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
}

export default function ChatList({ messages, isLoading }: ChatListProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 pb-32 bg-[#0f0f12]">
      {messages.length === 0 && (
        <div className="text-center mt-32 text-[var(--muted)]">
          <h2 className="text-3xl mb-4 font-bold text-[var(--accent)]">
            RLFusion Orchestrator
          </h2>
          <p className="text-lg text-[var(--muted)]">Your RTX 5070 is alive. Ask me anything.</p>
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
    </div>
  );
}
