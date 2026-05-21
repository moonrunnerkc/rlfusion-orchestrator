// Author: Bradley R. Kinnard
// app: top-level shell. Heavy logic lives in hooks/, app/, and components/.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AgentPipeline } from './components/agent-pipeline';
import { ChatInput } from './components/chat-input';
import { ChatList } from './components/chat-list';
import { MonitoringPanel } from './components/monitoring-panel';
import { ReindexButton } from './components/reindex-button';
import { createSession, generateTitle, loadChats, saveChats } from './app/chat-store';
import { useWsClient, type WsStatus } from './hooks/use-ws-client';
import type {
  AgentStatus,
  ChatSession,
  Message,
  SystemHealth,
  Weights,
} from './types/contracts';

const INITIAL_PIPELINE: AgentStatus[] = [
  { name: 'safety', status: 'idle' },
  { name: 'retrieval', status: 'idle' },
  { name: 'fusion', status: 'idle' },
  { name: 'generation', status: 'idle' },
];

const PENDING_PIPELINE: AgentStatus[] = [
  { name: 'safety', status: 'pending' },
  { name: 'retrieval', status: 'pending' },
  { name: 'fusion', status: 'pending' },
  { name: 'generation', status: 'pending' },
];

const WS_STATUS_LABEL: Record<WsStatus, string> = {
  connecting: 'Connecting...',
  open: 'Live',
  closed: 'Reconnecting...',
  error: 'Reconnecting...',
};

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function asWeights(raw: unknown): Weights {
  if (raw && typeof raw === 'object' && 'cag' in raw && 'graph' in raw) {
    const obj = raw as { cag: unknown; graph: unknown };
    return { cag: asNumber(obj.cag), graph: asNumber(obj.graph) };
  }
  return { cag: 0, graph: 0 };
}

export function App() {
  const [chats, setChats] = useState<ChatSession[]>(() => loadChats());
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [weights, setWeights] = useState<Weights>({ cag: 0.5, graph: 0.5 });
  const [reward, setReward] = useState<number>(0);
  const [proactiveHint, setProactiveHint] = useState<string>('Waiting for next query...');
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<'chat' | 'build' | 'test'>('chat');
  const [systemInfo, setSystemInfo] = useState<Partial<SystemHealth>>({
    model: '—',
    policy: '—',
    device: 'cpu',
  });
  const [pipelineAgents, setPipelineAgents] = useState<AgentStatus[]>(INITIAL_PIPELINE);
  const [pipelineActive, setPipelineActive] = useState(false);
  const [pipelineElapsed, setPipelineElapsed] = useState(0);
  const pipelineStartRef = useRef(0);
  const textareaFocusRef = useRef<() => void>(() => {});

  // Pull /ping once on mount. Wipe local chats if the server boot id changed
  // (server-side memory is no longer in sync, so the UX should feel fresh).
  useEffect(() => {
    fetch('http://localhost:8000/ping')
      .then((res) => res.json())
      .then((data) => {
        setSystemInfo({
          gpu: data.gpu ?? null,
          model: data.model ?? '—',
          policy: data.policy ?? '—',
          device: data.device ?? 'cpu',
          inferenceEngine: data.inference_engine ?? '',
          engineResolution: data.engine_resolution ?? '',
          policyExists: data.policy_exists ?? false,
          bootId: data.boot_id ?? '',
        });
        const prevBoot = localStorage.getItem('rlfusion_boot_id');
        if (data.boot_id && prevBoot !== data.boot_id) {
          localStorage.setItem('rlfusion_boot_id', data.boot_id);
          localStorage.removeItem('rlfusion_chats');
          setChats([]);
          setCurrentChatId(null);
          setMessages([]);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (currentChatId && messages.length > 0) {
      setChats((prev) => {
        const updated = prev.map((chat) =>
          chat.id === currentChatId
            ? { ...chat, messages, title: generateTitle(messages), updatedAt: Date.now() }
            : chat,
        );
        saveChats(updated);
        return updated;
      });
    }
  }, [messages, currentChatId]);

  const handleWsMessage = useCallback((raw: unknown) => {
    if (!raw || typeof raw !== 'object') return;
    const data = raw as Record<string, unknown>;

    if (typeof data.chunk === 'string') {
      const chunkText = data.chunk;
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === 'rlfusion') {
          return [...prev.slice(0, -1), { ...last, text: last.text + chunkText }];
        }
        return [...prev, { id: Date.now().toString(), text: chunkText, role: 'rlfusion' }];
      });
      if (data.weights !== undefined) setWeights(asWeights(data.weights));
      if (typeof data.reward === 'number') setReward(data.reward);
      if (typeof data.proactive === 'string' && data.proactive) {
        setProactiveHint(data.proactive);
      }
    }

    if (data.type === 'token' && typeof data.token === 'string') {
      const tokenText = data.token;
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last?.role === 'rlfusion') {
          return [...prev.slice(0, -1), { ...last, text: last.text + tokenText }];
        }
        return [...prev, { id: Date.now().toString(), text: tokenText, role: 'rlfusion' }];
      });
    }

    if (data.type === 'done') {
      setIsLoading(false);
      if (typeof data.response === 'string' && data.response) {
        const responseText = data.response;
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.role === 'rlfusion') {
            return [...prev.slice(0, -1), { ...last, text: responseText }];
          }
          return [...prev, { id: Date.now().toString(), text: responseText, role: 'rlfusion' }];
        });
      }
      if (data.fusion_weights !== undefined) setWeights(asWeights(data.fusion_weights));
      if (typeof data.reward === 'number') setReward(data.reward);
      if (Array.isArray(data.proactive_suggestions) && data.proactive_suggestions.length > 0) {
        const first = data.proactive_suggestions[0];
        if (typeof first === 'string') setProactiveHint(first);
      }
      if (pipelineStartRef.current > 0) {
        setPipelineElapsed(Date.now() - pipelineStartRef.current);
      }
      setPipelineActive(false);
      // Refocus the textarea after a turn so keyboard-only users keep flow.
      window.setTimeout(() => textareaFocusRef.current(), 0);
    }

    if (data.type === 'pipeline' && Array.isArray(data.agents)) {
      const agents = data.agents as AgentStatus[];
      setPipelineAgents(agents);
      setPipelineActive(agents.some((a) => a.status === 'running' || a.status === 'pending'));
    }

    if (data.type === 'critique') {
      if (typeof data.reward === 'number') setReward(data.reward);
      if (Array.isArray(data.proactive_suggestions) && data.proactive_suggestions.length > 0) {
        const first = data.proactive_suggestions[0];
        if (typeof first === 'string') setProactiveHint(first);
      }
    }

    if (data.type === 'start') {
      pipelineStartRef.current = Date.now();
      setPipelineElapsed(0);
      setPipelineActive(true);
      setPipelineAgents(PENDING_PIPELINE);
    }
  }, []);

  const ws = useWsClient({ url: 'ws://localhost:8000/ws', onMessage: handleWsMessage });

  const createNewChat = () => {
    ws.send({ clear_memory: true });
    const newChat = createSession();
    setChats((prev) => {
      const updated = [newChat, ...prev];
      saveChats(updated);
      return updated;
    });
    setCurrentChatId(newChat.id);
    setMessages([]);
    setProactiveHint('Waiting for next query...');
    setReward(0);
  };

  const loadChat = (chatId: string) => {
    ws.send({ clear_memory: true });
    const chat = chats.find((c) => c.id === chatId);
    if (chat) {
      setCurrentChatId(chat.id);
      setMessages(chat.messages);
      setProactiveHint('Waiting for next query...');
    }
  };

  const deleteChat = (chatId: string) => {
    setChats((prev) => {
      const updated = prev.filter((c) => c.id !== chatId);
      saveChats(updated);
      return updated;
    });
    if (currentChatId === chatId) {
      setCurrentChatId(null);
      setMessages([]);
    }
  };

  const sendMessage = (text: string) => {
    if (!currentChatId) {
      const newChat = createSession(text.slice(0, 40) + (text.length > 40 ? '...' : ''));
      setChats((prev) => {
        const updated = [newChat, ...prev];
        saveChats(updated);
        return updated;
      });
      setCurrentChatId(newChat.id);
    }
    const userMsg: Message = { id: Date.now().toString(), text, role: 'user' };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
    ws.send({ query: text, mode });
  };

  const sources = useMemo(
    () => [
      { key: 'cag' as const, icon: '💾', label: 'CAG' },
      { key: 'graph' as const, icon: '🕸️', label: 'Graph' },
    ],
    [],
  );

  return (
    <div className="flex h-screen bg-black text-white">
      <div className="scanlines" aria-hidden="true" />
      <aside className="w-80 border-r border-gray-800 flex flex-col">
        <div className="p-6 border-b border-gray-800">
          <h1 className="text-3xl font-black text-[var(--accent)]">RLFusion</h1>
          <p className="text-xs text-gray-500 mt-1">{WS_STATUS_LABEL[ws.status]}</p>
        </div>

        <button
          type="button"
          onClick={createNewChat}
          className="mx-4 mt-4 px-4 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-xl font-semibold transition"
        >
          + New Chat
        </button>

        <div className="flex-1 overflow-y-auto px-2 mt-4">
          <p className="px-2 text-xs text-gray-500 uppercase tracking-wider mb-2">Recent Chats</p>
          {chats.length === 0 ? (
            <p className="px-4 py-2 text-sm text-gray-600">No chats yet</p>
          ) : (
            <div className="space-y-1">
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  className={`group flex items-center justify-between rounded-lg transition-all ${
                    currentChatId === chat.id
                      ? 'bg-cyan-500/20 border border-cyan-500/30'
                      : 'hover:bg-gray-800/50'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => loadChat(chat.id)}
                    className="flex-1 min-w-0 text-left px-3 py-2.5"
                  >
                    <p
                      className={`text-sm truncate ${
                        currentChatId === chat.id ? 'text-cyan-400' : 'text-gray-300'
                      }`}
                    >
                      {chat.title}
                    </p>
                    <p className="text-xs text-gray-600">
                      {new Date(chat.updatedAt).toLocaleDateString()}
                    </p>
                  </button>
                  <button
                    type="button"
                    onClick={() => deleteChat(chat.id)}
                    className="opacity-0 group-hover:opacity-100 px-2 py-2 hover:bg-red-500/20 rounded transition-all"
                    title="Delete chat"
                    aria-label={`Delete chat ${chat.title}`}
                  >
                    <span className="text-red-400 text-xs">✕</span>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="px-6 py-4 border-t border-gray-800">
          <label htmlFor="mode-select" className="text-xs text-gray-400 uppercase tracking-wider">
            Mode
          </label>
          <select
            id="mode-select"
            value={mode}
            onChange={(e) => setMode(e.target.value as 'chat' | 'build' | 'test')}
            className="w-full mt-2 bg-gray-900/80 border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500 transition"
          >
            <option value="chat">🗨️ Chat Mode</option>
            <option value="build">🛠️ Build Assistant</option>
            <option value="test">🧪 Experimental Test</option>
          </select>
        </div>

        <div className="px-4 pb-4">
          <AgentPipeline agents={pipelineAgents} isActive={pipelineActive} elapsedMs={pipelineElapsed} />
        </div>
      </aside>

      <main className="flex-1 flex flex-col max-w-5xl mx-auto w-full">
        <ChatList
          messages={messages}
          isLoading={isLoading}
          proactiveHint={proactiveHint}
          onSuggestionClick={sendMessage}
        />
        <ChatInput
          onSend={sendMessage}
          disabled={isLoading}
          registerFocus={(fn) => {
            textareaFocusRef.current = fn;
          }}
        />
      </main>

      <aside className="w-80 border-l border-gray-800 flex flex-col">
        <div className="p-5 border-b border-gray-800">
          <h2 className="text-lg font-bold text-cyan-400">Fusion Engine</h2>
          <p className="text-xs text-gray-500 mt-1">RL-driven semantic orchestrator</p>
        </div>

        <div className="p-5 space-y-5 flex-1 overflow-y-auto">
          <div className="space-y-4">
            {sources.map(({ key, icon, label }) => (
              <div key={key}>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[var(--accent)] flex items-center gap-2">
                    <span className="text-lg" aria-hidden="true">{icon}</span>
                    {label}
                  </span>
                  <span className="font-mono text-[var(--muted)]">
                    {Number.isNaN(weights[key]) ? '0.0' : (weights[key] * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-[var(--accent)]/50 to-[var(--accent)]/10 h-3 rounded-full transition-all duration-700"
                    style={{ width: `${Number.isNaN(weights[key]) ? 0 : weights[key] * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <ReindexButton />

          <div className="pt-6 border-t border-gray-800">
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="text-[var(--accent)]">Reward</span>
              <span className="font-mono text-lg text-[var(--muted)]">{reward.toFixed(2)}</span>
            </div>
            <div className="text-xs text-[var(--muted)]">RL critique score for last response</div>
          </div>

          <MonitoringPanel weights={weights} reward={reward} isActive={isLoading} />

          <div className="text-xs text-[var(--muted)] space-y-2 pt-6 border-t border-gray-800">
            <div className="flex justify-between items-center">
              <span className="text-gray-500">LLM</span>
              <span className="text-[var(--accent)] font-mono text-[11px]">{systemInfo.model}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Device</span>
              <span className="text-[var(--accent)] font-mono text-[11px]">
                {systemInfo.gpu || (systemInfo.device ?? 'cpu').toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Policy</span>
              <span className="text-[var(--accent)] font-mono text-[11px]">{systemInfo.policy}</span>
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}
