// Author: Bradley R. Kinnard
// Renders chat bubbles with full Markdown support.
// After fighting with LLMs spitting walls of text at 3am, we finally gave them proper formatting.
import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type MessageRole = 'user' | 'rlfusion' | 'system';

interface ChatMessageProps {
  text: string;
  role: MessageRole;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ text, role }) => {
  const isUser = role === 'user';
  const label = isUser ? 'You' : 'RLFusion';
  const [copied, setCopied] = useState(false);

  // Copy to clipboard - only for AI messages because who copies their own text?
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // GUNMETAL DAWN: user no longer screams in cyan
  let bubbleClasses = 'max-w-2xl px-5 py-3 rounded-2xl ';
  if (isUser) {
    bubbleClasses += 'bg-[var(--accent)]/10 text-[var(--accent)]';
  } else {
    bubbleClasses += 'bg-gray-900/95 text-[var(--accent)] border border-gray-700';
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      <div className={`${bubbleClasses} relative group`}>
        <div className="text-xs opacity-70 mb-1">{label}</div>

        {/* Copy button - only show for AI responses */}
        {!isUser && (
          <button
            onClick={handleCopy}
            className="absolute -top-2 -right-2 bg-gray-900/90 border border-gray-700 backdrop-blur-sm rounded-lg p-2 opacity-0 group-hover:opacity-100 hover:bg-[var(--accent)]/20 transition-all duration-200"
            title="Copy response"
          >
            <span className="text-xs text-[var(--accent)] font-mono">
              {copied ? '✓' : '⎘'}
            </span>
          </button>
        )}

        {/* prose-invert keeps typography readable on dark bg, max-w-none lets it breathe */}
        <div className="prose prose-invert max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              // Main headers get the cyber cyan glow with left border accent
              h1: ({ node, ...props }) => (
                <h1 className="text-2xl font-bold text-[var(--accent)] mt-8 mb-4 border-l-4 border-[var(--accent)] pl-4" {...props} />
              ),

              // Subheaders slightly dimmer but still pop
              h2: ({ node, ...props }) => (
                <h2 className="text-xl font-semibold text-[var(--accent)] mt-6 mb-3" {...props} />
              ),

              // Bold text gets the cyan treatment for emphasis
              strong: ({ node, ...props }) => (
                <strong className="text-[var(--accent)] font-semibold" {...props} />
              ),

              // Code styling: inline vs block
              code({ node, inline, className, children, ...props }) {
                if (inline) {
                  // Inline code → subtle bg with cyan text for readability
                  return (
                    <code
                      className="bg-gray-900/80 border border-gray-700 px-1.5 py-0.5 rounded text-[var(--accent)] text-sm font-mono"
                      {...props}
                    >
                      {children}
                    </code>
                  );
                }

                // Code blocks → full-width, darker bg, slight shadow for depth
                return (
                  <code
                    className="block bg-gray-900 border border-gray-700 p-4 rounded-lg overflow-x-auto my-5 shadow-lg font-mono text-sm text-[var(--accent)]"
                    {...props}
                  >
                    {children}
                  </code>
                );
              },

              // Lists need proper spacing and bullets
              ul: ({ node, ...props }) => (
                <ul className="list-disc ml-6 my-4 space-y-2" {...props} />
              ),

              ol: ({ node, ...props }) => (
                <ol className="list-decimal ml-6 my-4 space-y-2" {...props} />
              ),

              li: ({ node, ...props }) => (
                <li className="leading-relaxed" {...props} />
              ),

              // Paragraphs get breathing room
              p: ({ node, ...props }) => (
                <p className="mb-4 leading-relaxed" {...props} />
              ),
            }}
          >
            {text}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
};
