// Author: Bradley R. Kinnard
// Renders chat bubbles with full Markdown support.
// After fighting with LLMs spitting walls of text at 3am, we finally gave them proper formatting.
import React, { useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type MessageRole = 'user' | 'rlfusion' | 'system';

interface ChatMessageProps {
  text: string;
  role: MessageRole;
}

// Parse critique block from response text
interface CritiqueData {
  factualAccuracy: number;
  proactivity: number;
  helpfulness: number;
  finalReward: number;
  suggestions: string[];
}

function parseCritique(text: string): { mainText: string; critique: CritiqueData | null } {
  const critiqueMatch = text.match(/<critique>([\s\S]*?)<\/critique>/i);

  if (!critiqueMatch) {
    // Still strip any "Self Critique:" header even if no tags
    const cleanedText = text.replace(/\n*\**\s*Self[- ]?Critique:?\s*\**\n*/gi, '').trim();
    return { mainText: cleanedText, critique: null };
  }

  // Strip critique tags AND any "Self Critique:" header that precedes them
  const mainText = text
    .replace(/\n*\**\s*Self[- ]?Critique:?\s*\**\n*/gi, '')
    .replace(/<critique>[\s\S]*?<\/critique>/gi, '')
    .trim();
  const critiqueText = critiqueMatch[1];

  // Parse individual scores
  const factualMatch = critiqueText.match(/Factual accuracy:\s*([\d.]+)/i);
  const proactiveMatch = critiqueText.match(/Proactivity score:\s*([\d.]+)/i);
  const helpfulMatch = critiqueText.match(/Helpfulness:\s*([\d.]+)/i);
  const rewardMatch = critiqueText.match(/Final reward:\s*([\d.]+)/i);

  // Parse suggestions
  const suggestionsMatch = critiqueText.match(/Proactive suggestions:([\s\S]*?)$/i);
  const suggestions: string[] = [];
  if (suggestionsMatch) {
    const suggestionLines = suggestionsMatch[1].match(/[•\-\*]\s*(?:suggestion \w+:\s*)?(.+)/gi);
    if (suggestionLines) {
      suggestionLines.forEach(line => {
        const cleaned = line.replace(/^[•\-\*]\s*(?:suggestion \w+:\s*)?/i, '').trim();
        if (cleaned) suggestions.push(cleaned);
      });
    }
  }

  return {
    mainText,
    critique: {
      factualAccuracy: factualMatch ? parseFloat(factualMatch[1]) : 0,
      proactivity: proactiveMatch ? parseFloat(proactiveMatch[1]) : 0,
      helpfulness: helpfulMatch ? parseFloat(helpfulMatch[1]) : 0,
      finalReward: rewardMatch ? parseFloat(rewardMatch[1]) : 0,
      suggestions
    }
  };
}

// Styled critique component (retained for future use)
// @ts-ignore unused
function CritiquePanel({ critique }: { critique: CritiqueData }) {
  const [expanded, setExpanded] = useState(false);

  // Color based on score
  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'text-emerald-400';
    if (score >= 0.7) return 'text-cyan-400';
    if (score >= 0.5) return 'text-amber-400';
    return 'text-red-400';
  };

  const getBarColor = (score: number) => {
    if (score >= 0.9) return 'bg-emerald-500';
    if (score >= 0.7) return 'bg-cyan-500';
    if (score >= 0.5) return 'bg-amber-500';
    return 'bg-red-500';
  };

  const rewardColor = critique.finalReward >= 8 ? 'text-emerald-400' :
                      critique.finalReward >= 6 ? 'text-cyan-400' :
                      critique.finalReward >= 4 ? 'text-amber-400' : 'text-red-400';

  return (
    <div className="mt-4 border-t border-gray-700/50 pt-4">
      {/* Collapsed view - just the score */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between text-xs hover:bg-gray-800/50 rounded-lg p-2 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-gray-500">📊</span>
          <span className="text-gray-400 font-medium">Response Quality</span>
        </div>
        <div className="flex items-center gap-3">
          <span className={`font-mono font-bold ${rewardColor}`}>
            {critique.finalReward.toFixed(1)}/10
          </span>
          <span className="text-gray-600 text-[10px]">{expanded ? '▲' : '▼'}</span>
        </div>
      </button>

      {/* Expanded view - full metrics */}
      {expanded && (
        <div className="mt-3 space-y-3 animate-in slide-in-from-top-2 duration-200">
          {/* Score bars */}
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: 'Accuracy', score: critique.factualAccuracy },
              { label: 'Proactive', score: critique.proactivity },
              { label: 'Helpful', score: critique.helpfulness }
            ].map(({ label, score }) => (
              <div key={label} className="space-y-1">
                <div className="flex justify-between text-[10px]">
                  <span className="text-gray-500">{label}</span>
                  <span className={`font-mono ${getScoreColor(score)}`}>
                    {(score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${getBarColor(score)} transition-all duration-500`}
                    style={{ width: `${score * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Suggestions */}
          {critique.suggestions.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-800/50">
              <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">
                💡 Suggestions
              </p>
              <ul className="space-y-1">
                {critique.suggestions.map((s, i) => (
                  <li key={i} className="text-xs text-gray-400 flex items-start gap-2">
                    <span className="text-cyan-500/50 mt-0.5">›</span>
                    <span>{s}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ text, role }) => {
  const isUser = role === 'user';
  const label = isUser ? 'You' : 'RLFusion';
  const [copied, setCopied] = useState(false);

  // Parse critique from message
  const { mainText, critique: _critique } = useMemo(() => parseCritique(text), [text]);

  // Copy to clipboard - only for AI messages because who copies their own text?
  const handleCopy = async () => {
    await navigator.clipboard.writeText(mainText);
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
              h1: ({ node, ...props }: {node?: unknown; [key: string]: unknown}) => (
                <h1 className="text-2xl font-bold text-[var(--accent)] mt-8 mb-4 border-l-4 border-[var(--accent)] pl-4" {...props} />
              ),

              // Subheaders slightly dimmer but still pop
              h2: ({ node, ...props }: {node?: unknown; [key: string]: unknown}) => (
                <h2 className="text-xl font-semibold text-[var(--accent)] mt-6 mb-3" {...props} />
              ),

              // Bold text gets the cyan treatment for emphasis
              strong: ({ node, ...props }: {node?: unknown; [key: string]: unknown}) => (
                <strong className="text-[var(--accent)] font-semibold" {...props} />
              ),

              // Code styling: inline vs block
              code({ inline, className, children, ...props }: {inline?: boolean; className?: string; children?: React.ReactNode; [key: string]: unknown}) {
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
              ul: ({ node, ...props }: {node?: unknown; [key: string]: unknown}) => (
                <ul className="list-disc ml-6 my-4 space-y-2" {...props} />
              ),

              ol: ({ node, ...props }: {node?: unknown; [key: string]: unknown}) => (
                <ol className="list-decimal ml-6 my-4 space-y-2" {...props} />
              ),

              li: ({ node, ...props }: {node?: unknown; [key: string]: unknown}) => (
                <li className="leading-relaxed" {...props} />
              ),

              // Paragraphs get breathing room
              p: ({ node, ...props }: {node?: unknown; [key: string]: unknown}) => (
                <p className="mb-4 leading-relaxed" {...props} />
              ),

              // Phase 7: render image results inline with captions
              img: ({ node, src, alt, ...props }: {node?: unknown; src?: string; alt?: string; [key: string]: unknown}) => {
                // rewrite relative image paths to the backend image endpoint
                const imgSrc = src && !src.startsWith('http') && !src.startsWith('data:')
                  ? `http://localhost:8000/api/images/${src.replace(/^data\/images\//, '')}`
                  : src;
                return (
                  <span className="block my-4">
                    <img
                      src={imgSrc}
                      alt={alt || 'Image result'}
                      className="max-w-full max-h-96 rounded-lg border border-gray-700 shadow-lg"
                      loading="lazy"
                      {...props}
                    />
                    {alt && (
                      <span className="block text-xs text-gray-500 mt-1 italic">{alt}</span>
                    )}
                  </span>
                );
              },
            }}
          >
            {mainText}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
};
