// src/App.tsx — PRODUCTION VERSION (2-path CAG+Graph architecture)
import { useEffect, useRef, useState } from 'react';
import type { AgentStatus } from './components/AgentPipeline';
import AgentPipeline from './components/AgentPipeline';
import ChatInput from './components/ChatInput';
import ChatList from './components/ChatList';
import MonitoringPanel from './components/MonitoringPanel';
import type { ChatSession, Message, Weights } from './types/contracts';

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

// RAG Documents panel — upload + reindex
function ReindexButton() {
  const [status, setStatus] = useState<'idle' | 'uploading' | 'indexing' | 'done' | 'empty' | 'error'>('idle');
  const [details, setDetails] = useState<string>('');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadAndIndex = async (files: FileList | File[]) => {
    const allowed = ['.txt', '.md', '.pdf', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'];
    const valid = Array.from(files).filter(f =>
      allowed.some(ext => f.name.toLowerCase().endsWith(ext))
    );
    if (valid.length === 0) {
      setStatus('error');
      setDetails('Only .txt, .md, .pdf, and image files accepted');
      setTimeout(() => { setStatus('idle'); setDetails(''); }, 3000);
      return;
    }

    // Upload
    setStatus('uploading');
    setDetails(`Uploading ${valid.length} file${valid.length > 1 ? 's' : ''}...`);
    const form = new FormData();
    valid.forEach(f => form.append('files', f));

    try {
      const upRes = await fetch('http://localhost:8000/api/upload', { method: 'POST', body: form });
      const upData = await upRes.json();
      if (upData.total_saved === 0) {
        setStatus('error');
        setDetails(upData.message || 'Upload failed');
        setTimeout(() => { setStatus('idle'); setDetails(''); }, 3000);
        return;
      }

      // Auto-reindex after upload
      setStatus('indexing');
      setDetails(`Uploaded ${upData.total_saved} — building index...`);
      const ixRes = await fetch('http://localhost:8000/api/reindex', { method: 'POST' });
      const ixData = await ixRes.json();
      setStatus('done');
      const imgNote = ixData.images_indexed > 0 ? `, ${ixData.images_indexed} images` : '';
      setDetails(`${ixData.files_processed} files, ${ixData.chunks_indexed} chunks${imgNote} (${ixData.elapsed_seconds}s)`);
      setTimeout(() => { setStatus('idle'); setDetails(''); }, 6000);
    } catch {
      setStatus('error');
      setDetails('Backend unreachable');
      setTimeout(() => { setStatus('idle'); setDetails(''); }, 4000);
    }
  };

  const handleReindexOnly = async () => {
    setStatus('indexing');
    setDetails('Rebuilding index...');
    try {
      const res = await fetch('http://localhost:8000/api/reindex', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'empty') {
        setStatus('empty');
        setDetails('No documents in data/docs/');
      } else {
        setStatus('done');
        const imgNote = data.images_indexed > 0 ? `, ${data.images_indexed} images` : '';
        setDetails(`${data.files_processed} files, ${data.chunks_indexed} chunks${imgNote} (${data.elapsed_seconds}s)`);
      }
      setTimeout(() => { setStatus('idle'); setDetails(''); }, 5000);
    } catch {
      setStatus('error');
      setDetails('Backend unreachable');
      setTimeout(() => { setStatus('idle'); setDetails(''); }, 4000);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length > 0) uploadAndIndex(e.dataTransfer.files);
  };

  const onFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      uploadAndIndex(e.target.files);
      e.target.value = '';
    }
  };

  const busy = status === 'uploading' || status === 'indexing';

  return (
    <div className="pt-4 border-t border-gray-800">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-lg">📄</span>
          <span className="text-sm text-[var(--accent)]">RAG Documents</span>
        </div>
        <button
          onClick={handleReindexOnly}
          disabled={busy}
          className="text-xs text-gray-400 hover:text-cyan-400 transition disabled:opacity-40"
          title="Rebuild index without uploading"
        >
          ↻ Reindex
        </button>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !busy && fileInputRef.current?.click()}
        className={`w-full border-2 border-dashed rounded-lg px-3 py-4 text-center cursor-pointer transition ${
          dragOver ? 'border-cyan-400 bg-cyan-500/10' :
          busy ? 'border-gray-700 bg-gray-800/30 cursor-wait' :
          'border-gray-700 hover:border-gray-500 hover:bg-gray-800/30'
        }`}
      >
        {busy ? (
          <p className="text-xs text-gray-400 animate-pulse">{details}</p>
        ) : (
          <>
            <p className="text-sm text-gray-400">Drop files here</p>
            <p className="text-xs text-gray-600 mt-1">.txt  .md  .pdf  .png  .jpg</p>
          </>
        )}
      </div>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".txt,.md,.pdf,.png,.jpg,.jpeg,.webp,.bmp,.tiff"
        onChange={onFileSelect}
        className="hidden"
      />

      {/* Status feedback */}
      {status === 'done' && details && (
        <p className="text-xs text-emerald-400 mt-1">✓ {details}</p>
      )}
      {status === 'empty' && details && (
        <p className="text-xs text-amber-400 mt-1">⚠ {details}</p>
      )}
      {status === 'error' && details && (
        <p className="text-xs text-red-400 mt-1">✕ {details}</p>
      )}
    </div>
  );
}

export default function App() {
  const [chats, setChats] = useState<ChatSession[]>(() => loadChats());
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [weights, setWeights] = useState<Weights>({ cag: 0.5, graph: 0.5 });
  const [reward, setReward] = useState<number>(0);
  const [proactiveHint, setProactiveHint] = useState<string>('Waiting for next query...');
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<'chat' | 'build' | 'test'>('chat');
  const [systemInfo, setSystemInfo] = useState<{gpu: string | null; model: string; policy: string; device: string; stis: {enabled: boolean; model: string | null; status: string; agents: number} | null}>({
    gpu: null, model: '—', policy: '—', device: 'cpu', stis: null
  });
  const [pipelineAgents, setPipelineAgents] = useState<AgentStatus[]>([
    { name: 'safety', status: 'idle' },
    { name: 'retrieval', status: 'idle' },
    { name: 'fusion', status: 'idle' },
    { name: 'generation', status: 'idle' },
  ]);
  const [pipelineActive, setPipelineActive] = useState(false);
  const [pipelineElapsed, setPipelineElapsed] = useState(0);
  const pipelineStartRef = useRef(0);
  const ws = useRef<WebSocket | null>(null);

  // Fetch system info from backend on mount — clear stale chats on new server boot
  useEffect(() => {
    fetch('http://localhost:8000/ping')
      .then(res => res.json())
      .then(data => {
        setSystemInfo({
          gpu: data.gpu || null,
          model: data.model || '—',
          policy: data.policy || '—',
          device: data.device || 'cpu',
          stis: data.stis || null,
        });
        // If the server rebooted, wipe client-side state so it feels fresh
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
            cag: data.weights[0] || 0,
            graph: data.weights[1] || 0,
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

        // Replace streamed text with cleaned response (strips critique block)
        // Also handles blocked queries where no streaming chunks were sent
        if (data.response) {
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last?.role === 'rlfusion') {
              return [...prev.slice(0, -1), { ...last, text: data.response }];
            }
            // no assistant message yet (e.g. safety-blocked before any streaming)
            return [...prev, { id: Date.now().toString(), text: data.response, role: 'rlfusion' }];
          });
        }

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

        // pipeline finished — compute elapsed, agents stay visible until next query
        if (pipelineStartRef.current > 0) {
          setPipelineElapsed(Date.now() - pipelineStartRef.current);
        }
        setPipelineActive(false);
      }

      // Handle pipeline status updates
      if (data.type === 'pipeline' && data.agents) {
        setPipelineAgents(data.agents);
        const hasRunning = data.agents.some((a: AgentStatus) => a.status === 'running' || a.status === 'pending');
        setPipelineActive(hasRunning);
      }

      // Handle async critique result (reward update after generation)
      if (data.type === 'critique') {
        if (typeof data.reward === 'number') {
          setReward(data.reward);
        }
        if (data.proactive_suggestions && data.proactive_suggestions.length > 0) {
          setProactiveHint(data.proactive_suggestions[0]);
        }
      }

      // Handle start signal — set timer imperatively
      if (data.type === 'start') {
        pipelineStartRef.current = Date.now();
        setPipelineElapsed(0);
        setPipelineActive(true);
        setPipelineAgents([
          { name: 'safety', status: 'pending' },
          { name: 'retrieval', status: 'pending' },
          { name: 'fusion', status: 'pending' },
          { name: 'generation', status: 'pending' },
        ]);
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

        {/* AGENT PIPELINE */}
        <div className="px-4 pb-4">
          <AgentPipeline agents={pipelineAgents} isActive={pipelineActive} elapsedMs={pipelineElapsed} />
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
          {/* 2-path retrieval sources with icons */}
          <div className="space-y-4">
            {([
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

          {/* Reindex Documents */}
          <ReindexButton />

          {/* Reward Score */}
          <div className="pt-6 border-t border-gray-800">
            <div className="flex items-center justify-between text-sm mb-2">
              <span className="text-[var(--accent)]">Reward</span>
              <span className="font-mono text-lg text-[var(--muted)]">{reward.toFixed(2)}</span>
            </div>
            <div className="text-xs text-[var(--muted)]">RL critique score for last response</div>
          </div>

          {/* System Monitor (Phase 9) */}
          <MonitoringPanel weights={weights} reward={reward} isActive={isLoading} />

          {/* System Info */}
          <div className="text-xs text-[var(--muted)] space-y-2 pt-6 border-t border-gray-800">
            <div className="flex justify-between items-center">
              <span className="text-gray-500">LLM</span>
              <span className="text-[var(--accent)] font-mono text-[11px]">{systemInfo.model}</span>
            </div>
            {systemInfo.stis?.enabled && (
              <div className="flex justify-between items-center">
                <span className="text-gray-500">STIS</span>
                <span className="flex items-center gap-1.5">
                  <span className={`inline-block w-1.5 h-1.5 rounded-full ${
                    systemInfo.stis.status === 'ready' ? 'bg-emerald-400' :
                    systemInfo.stis.status === 'standby' ? 'bg-amber-400' :
                    'bg-gray-600'
                  }`} />
                  <span className="text-[var(--accent)] font-mono text-[11px]">
                    {systemInfo.stis.model ? systemInfo.stis.model.split('/').pop() : 'Qwen2.5-1.5B'}
                  </span>
                </span>
              </div>
            )}
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Device</span>
              <span className="text-[var(--accent)] font-mono text-[11px]">{systemInfo.gpu || systemInfo.device.toUpperCase()}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-500">Policy</span>
              <span className="text-[var(--accent)] font-mono text-[11px]">{systemInfo.policy}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
