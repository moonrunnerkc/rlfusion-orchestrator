// src/App.tsx — FINAL PRODUCTION VERSION (Grok-style cockpit)
import { useEffect, useRef, useState } from 'react';
import ChatInput from './components/ChatInput';
import ChatList from './components/ChatList';

interface Message {
  id: string;
  text: string;
  role: 'user' | 'rlfusion';
}

interface Weights {
  rag: number;
  cag: number;
  graph: number;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [weights, setWeights] = useState<Weights>({ rag: 0.33, cag: 0.33, graph: 0.34 });
  const [reward, setReward] = useState<number>(0);
  const [proactiveHint, setProactiveHint] = useState<string>('Waiting for next query...');
  const [isLoading, setIsLoading] = useState(false);
  const [mode, setMode] = useState<'chat' | 'build' | 'test'>('chat');
  const ws = useRef<WebSocket | null>(null);

  // Real WebSocket connection to your backend
  useEffect(() => {
    // Prevent double-mounting in dev from breaking connection
    if (ws.current?.readyState === WebSocket.OPEN) return;

    ws.current = new WebSocket('ws://localhost:8000/ws');
    ws.current.onmessage = (e) => {
      const data = JSON.parse(e.data);

      // Handle streaming tokens
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
      {/* LEFT SIDEBAR — NOW WITH REAL MODE SWITCHING */}
      <div className="w-80 border-r border-gray-800 flex flex-col">
        <div className="p-6 border-b border-gray-800">
          {/* Purple dreams finally die here */}
          <h1 className="text-3xl font-black text-[var(--accent)]">
            RLFusion
          </h1>
        </div>

        <button className="mx-4 mt-4 px-4 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-xl font-semibold transition">
          + New Chat
        </button>

        {/* MODE SELECTOR — THIS IS WHAT WAS MISSING */}
        <div className="px-6 mt-6">
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
          <p className="text-xs text-gray-500 mt-2">
            Current: <span className="text-cyan-400 font-mono">{mode}</span>
          </p>
        </div>
      </div>

      {/* CENTER: CHAT */}
      <div className="flex-1 flex flex-col max-w-5xl mx-auto w-full">
        <ChatList messages={messages} isLoading={isLoading} />
        <ChatInput onSend={sendMessage} disabled={isLoading} />
      </div>

      {/* RIGHT SIDEBAR — CLEAN & COMPACT */}
      <div className="w-80 border-l border-gray-800 flex flex-col">
        <div className="p-5 border-b border-gray-800">
          <h2 className="text-lg font-bold text-cyan-400">Fusion Engine</h2>
          <p className="text-xs text-gray-500 mt-1">RL-driven semantic orchestrator</p>
        </div>

        <div className="p-5 space-y-5 flex-1 overflow-y-auto">
          {/* Monochrome bars — because variety is overrated in hell */}
          <div className="space-y-4">
            {(['rag', 'cag', 'graph'] as const).map((key) => (
              <div key={key}>
                <div className="flex items-center justify-between text-sm">
                  {/* Labels use single accent var — erasing the last traces of color chaos */}
                  <span className="text-[var(--accent)]">
                    {key.toUpperCase()}
                  </span>
                  <span className="font-mono text-[var(--muted)]">{(weights[key] * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-3">
                  <div
                    className="bg-gradient-to-r from-[var(--accent)]/50 to-[var(--accent)]/10 h-3 rounded-full transition-all duration-700"
                    style={{ width: `${weights[key] * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

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

          {/* Proactive Hint Panel */}
          <div className="mt-6 p-4 bg-gray-900/50 rounded-lg border border-gray-800 text-xs">
            <p className="font-semibold text-cyan-400 mb-2">Proactive Hint</p>
            <p className="text-gray-300">{proactiveHint}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
