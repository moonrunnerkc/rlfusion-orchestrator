// Author: Bradley R. Kinnard
// chat-input: bottom textarea + send button.

import { useEffect, useRef, useState } from 'react';

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled?: boolean;
  // Lets the parent re-focus the textarea after a turn completes so
  // keyboard-only users keep their flow without grabbing focus mid-stream.
  registerFocus?: (focus: () => void) => void;
}

export function ChatInput({ onSend, disabled, registerFocus }: ChatInputProps) {
  const [input, setInput] = useState('');
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (registerFocus) {
      registerFocus(() => {
        ref.current?.focus();
      });
    }
  }, [registerFocus]);

  const handleSend = () => {
    if (input.trim() && !disabled) {
      onSend(input.trim());
      setInput('');
    }
  };

  return (
    <div className="border-t border-gray-800 bg-black p-4">
      <div className="max-w-5xl mx-auto flex gap-3">
        <textarea
          ref={ref}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder="Ask RLFusion anything..."
          className="flex-1 bg-gray-900 text-white rounded-xl px-5 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-cyan-500 transition"
          rows={1}
          disabled={disabled}
          aria-label="Chat message"
        />
        <button
          type="button"
          onClick={handleSend}
          disabled={disabled || !input.trim()}
          className="px-8 py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-xl font-semibold transition shadow-lg"
        >
          Send
        </button>
      </div>
    </div>
  );
}
