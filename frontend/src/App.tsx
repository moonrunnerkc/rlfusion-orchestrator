// src/App.tsx — FINAL PRODUCTION VERSION (Grok-style cockpit)
import { useEffect, useRef, useState } from 'react';
import ChatInput from './components/ChatInput';
import ChatList from './components/ChatList';

interface Message {
  id: string;
  text: string;
  role: 'user' | 'rlfusion';
}

interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

interface Weights {
  rag: number;
  cag: number;
  graph: number;
  web: number;
}

// Helper to generate chat title from first message
function generateTitle(messages: Message[]): string {
  const firstUserMsg = messages.find(m => m.role === 'user');
  if (!firstUserMsg) return 'New Chat';
  const text = firstUserMsg.text.slice(0, 40);
  return text.length < firstUserMsg.text.length ? `${text}...` : text;
}

// Load chats from localStorage
function loadChats(): ChatSession[] {
  try {
    const stored = localStorage.getItem('rlfusion_chats');
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

// Save chats to localStorage
function saveChats(chats: ChatSession[]) {
  localStorage.setItem('rlfusion_chats', JSON.stringify(chats));
}

// Inline Web Toggle Component
function WebToggle() {
  const [webEnabled, setWebEnabled] = useState(true);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch('http://localhost:8000/api/config')
      .then(res => res.json())
      .then(data => setWebEnabled(data.web?.enabled ?? true))
      .catch(() => {});
  }, []);

  const toggleWeb = async () => {
    setLoading(true);
    try {
      await fetch('http://localhost:8000/api/config', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ web: { enabled: !webEnabled } })
      });
      setWebEnabled(!webEnabled);
    } catch (err) {
      console.error('Failed to toggle web:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="pt-4 border-t border-gray-800">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-lg">🌐</span>
          <span className="text-sm text-[var(--accent)]">Web Search</span>
        </div>
        <button
          onClick={toggleWeb}
          disabled={loading}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            webEnabled ? 'bg-cyan-500' : 'bg-gray-700'
          } ${loading ? 'opacity-50' : ''}`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              webEnabled ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>
      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-400">
          {webEnabled ? 'Online' : 'Offline (100% local)'}
        </span>
      </div>
    </div>
  );
}

export default function App() {
  const [chats, setChats] = useState<ChatSession[]>(() => loadChats());
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [weights, setWeights] = useState<Weights>({ rag: 0.25, cag: 0.25, graph: 0.25, web: 0.25 });
  const [reward, setReward] = useState<number>(0);
  const [proactiveHint, setProactiveHint] = useState<string>('Waiting for next query...');
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<'chat' | 'build' | 'test'>('chat');
  const ws = useRef<WebSocket | null>(null);

  // Save messages to current chat when they change
  useEffect(() => {
    if (currentChatId && messages.length > 0) {
      setChats(prev => {
        const updated = prev.map(chat =>
          chat.id === currentChatId
            ? { ...chat, messages, title: generateTitle(messages), updatedAt: Date.now() }
            : chat
        );
        saveChats(updated);
        return updated;
      });
    }
  }, [messages, currentChatId]);

  // Create new chat - also clears server-side memory
  const createNewChat = () => {
    // Clear server-side conversation memory
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ clear_memory: true }));
    }

    const newChat: ChatSession = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now()
    };
    setChats(prev => {
      const updated = [newChat, ...prev];
      saveChats(updated);
      return updated;
    });
    setCurrentChatId(newChat.id);
    setMessages([]);
    setProactiveHint('Waiting for next query...');
    setReward(0);
  };

  // Load existing chat - clears memory since we're switching context
  const loadChat = (chatId: string) => {
    // Clear server-side memory when switching chats
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ clear_memory: true }));
    }

    const chat = chats.find(c => c.id === chatId);
    if (chat) {
      setCurrentChatId(chat.id);
      setMessages(chat.messages);
      setProactiveHint('Waiting for next query...');
    }
  };

  // Delete a chat
  const deleteChat = (chatId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setChats(prev => {
      const updated = prev.filter(c => c.id !== chatId);
      saveChats(updated);
      return updated;
    });
    if (currentChatId === chatId) {
      setCurrentChatId(null);
      setMessages([]);
    }
  };

  // Real WebSocket connection to your backend
  useEffect(() => {
    // Prevent double-mounting in dev from breaking connection
    if (ws.current?.readyState === WebSocket.OPEN) return;

    ws.current = new WebSocket('ws://localhost:8000/ws');
    ws.current.onmessage = (e) => {
      const data = JSON.parse(e.data);

      // Handle streaming tokens with live weights/reward/proactive
      if (data.chunk) {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.role === 'rlfusion') {
            return [...prev.slice(0, -1), { ...last, text: last.text + data.chunk }];
          }
          return [...prev, { id: Date.now().toString(), text: data.chunk, role: 'rlfusion' }];
        });

        // Update live weights/reward/proactive during streaming
        if (data.weights) {
          setWeights({
            rag: data.weights[0] || 0,
            cag: data.weights[1] || 0,
            graph: data.weights[2] || 0,
            web: data.weights[3] || 0
          });
        }
        if (typeof data.reward === 'number') {
          setReward(data.reward);
        }
        if (data.proactive) {
          setProactiveHint(data.proactive);
        }
      }

      // Handle legacy token format (fallback)
      if (data.type === 'token' && data.token) {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last?.role === 'rlfusion') {
            return [...prev.slice(0, -1), { ...last, text: last.text + data.token }];
          }
          return [...prev, { id: Date.now().toString(), text: data.token, role: 'rlfusion' }];
        });
      }

      // Handle completion
      if (data.type === 'done') {
        console.log('Received done message:', JSON.stringify(data, null, 2));
        setIsLoading(false);
        if (data.fusion_weights) {
          console.log('Updating weights to:', JSON.stringify(data.fusion_weights, null, 2));
          setWeights(data.fusion_weights);
        }
        if (typeof data.reward === 'number') {
          setReward(data.reward);
        }
        if (data.proactive_suggestions && data.proactive_suggestions.length > 0) {
          setProactiveHint(data.proactive_suggestions[0]);
        }
      }

      // Handle start signal
      if (data.type === 'start') {
        // Clear any previous loading state
      }
    };
    return () => ws.current?.close();
  }, []);

  const sendMessage = (text: string) => {
    // Auto-create a new chat if none exists
    if (!currentChatId) {
      const newChat: ChatSession = {
        id: Date.now().toString(),
        title: text.slice(0, 40) + (text.length > 40 ? '...' : ''),
        messages: [],
        createdAt: Date.now(),
        updatedAt: Date.now()
      };
      setChats(prev => {
        const updated = [newChat, ...prev];
        saveChats(updated);
        return updated;
      });
      setCurrentChatId(newChat.id);
    }

    const userMsg: Message = { id: Date.now().toString(), text, role: 'user' };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    // Send via WebSocket
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ query: text, mode }));
    }
  };

  return (
    <div className="flex h-screen bg-black text-white">
      <div className="scanlines" aria-hidden="true" />
      {/* LEFT SIDEBAR — CHAT HISTORY */}
      <div className="w-80 border-r border-gray-800 flex flex-col">
        <div className="p-6 border-b border-gray-800">
          <h1 className="text-3xl font-black text-[var(--accent)]">
            RLFusion
          </h1>
        </div>

        <button
          onClick={createNewChat}
          className="mx-4 mt-4 px-4 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-xl font-semibold transition"
        >
          + New Chat
        </button>

        {/* CHAT HISTORY */}
        <div className="flex-1 overflow-y-auto px-2 mt-4">
          <p className="px-2 text-xs text-gray-500 uppercase tracking-wider mb-2">Recent Chats</p>
          {chats.length === 0 ? (
            <p className="px-4 py-2 text-sm text-gray-600">No chats yet</p>
          ) : (
            <div className="space-y-1">
              {chats.map(chat => (
                <div
                  key={chat.id}
                  onClick={() => loadChat(chat.id)}
                  className={`group flex items-center justify-between px-3 py-2.5 rounded-lg cursor-pointer transition-all ${
                    currentChatId === chat.id
                      ? 'bg-cyan-500/20 border border-cyan-500/30'
                      : 'hover:bg-gray-800/50'
                  }`}
                >
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm truncate ${currentChatId === chat.id ? 'text-cyan-400' : 'text-gray-300'}`}>
                      {chat.title}
                    </p>
                    <p className="text-xs text-gray-600">
                      {new Date(chat.updatedAt).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={(e) => deleteChat(chat.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition-all"
                    title="Delete chat"
                  >
                    <span className="text-red-400 text-xs">✕</span>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* MODE SELECTOR */}
        <div className="px-6 py-4 border-t border-gray-800">
          <label className="text-xs text-gray-400 uppercase tracking-wider">Mode</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as 'chat' | 'build' | 'test')}
            className="w-full mt-2 bg-gray-900/80 border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-cyan-500 transition"
          >
            <option value="chat">🗨️ Chat Mode</option>
            <option value="build">🛠️ Build Assistant</option>
            <option value="test">🧪 Experimental Test</option>
          </select>
        </div>
      </div>

      {/* CENTER: CHAT */}
      <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full">
        <ChatList
          messages={messages}
          isLoading={isLoading}
          proactiveHint={proactiveHint}
          onSuggestionClick={sendMessage}
        />
        <ChatInput onSend={sendMessage} disabled={isLoading} />
      </div>

      {/* RIGHT SIDEBAR — CLEAN & COMPACT */}
      <div className="w-80 border-l border-gray-800 flex flex-col">
        <div className="p-5 border-b border-gray-800">
          <h2 className="text-lg font-bold text-cyan-400">Fusion Engine</h2>
          <p className="text-xs text-gray-500 mt-1">RL-driven semantic orchestrator</p>
        </div>

        <div className="p-5 space-y-5 flex-1 overflow-y-auto">
          {/* All 4 retrieval sources with icons */}
          <div className="space-y-4">
            {([
              { key: 'rag' as const, icon: '📄', label: 'RAG' },
              { key: 'cag' as const, icon: '💾', label: 'CAG' },
              { key: 'graph' as const, icon: '🕸️', label: 'Graph' }
            ]).map(({ key, icon, label }) => (
              <div key={key}>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-[var(--accent)] flex items-center gap-2">
                    <span className="text-lg">{icon}</span>
                    {label}
                  </span>
                  <span className="font-mono text-[var(--muted)]">
                    {isNaN(weights[key]) ? '0.0' : (weights[key] * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-[var(--accent)]/50 to-[var(--accent)]/10 h-3 rounded-full transition-all duration-700"
                    style={{ width: `${isNaN(weights[key]) ? 0 : weights[key] * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Web Toggle with Status */}
          <WebToggle />

          {/* Reward Score */}
          <div className="pt-6 border-t border-gray-800">
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="text-[var(--accent)]">Reward</span>
              <span className="font-mono text-lg text-[var(--muted)]">{reward.toFixed(2)}</span>
            </div>
            <div className="text-xs text-[var(--muted)]">RL critique score for last response</div>
          </div>

          {/* System Info */}
          <div className="text-xs text-[var(--muted)] space-y-1 pt-6 border-t border-gray-800">
            <div className="flex justify-between"><span>Model</span><span className="text-[var(--accent)]">qwen2:7b</span></div>
            <div className="flex justify-between"><span>GPU</span><span className="text-[var(--accent)]">RTX 5070</span></div>
            <div className="flex justify-between"><span>VRAM</span><span className="text-[var(--accent)]">11 GB</span></div>
            <div className="flex justify-between"><span>Policy</span><span className="text-[var(--accent)]">PPO (live)</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}
