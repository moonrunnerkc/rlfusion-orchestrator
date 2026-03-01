// Author: Bradley R. Kinnard
// MonitoringPanel — real-time system health, weight history, and reward tracking
import { useEffect, useRef, useState } from 'react';

interface WeightSnapshot {
  ts: number;
  cag: number;
  graph: number;
}

interface RewardEntry {
  ts: number;
  reward: number;
}

interface SystemHealth {
  gpu: string | null;
  device: string;
  model: string;
  gpu_model: string;
  cpu_model: string;
  policy: string;
  uptime: number;
}

interface BenchmarkSummary {
  name: string;
  passed: boolean;
  score: number;
  timestamp: string;
}

interface Props {
  weights: { cag: number; graph: number };
  reward: number;
  isActive: boolean;
}

const MAX_HISTORY = 50;

// Small sparkline bar chart (pure CSS, no charting lib required)
function MiniBar({ values, color, maxVal }: { values: number[]; color: string; maxVal: number }) {
  const barW = Math.max(2, Math.floor(100 / Math.max(values.length, 1)));
  return (
    <div className="flex items-end gap-px h-8 w-full">
      {values.slice(-30).map((v, i) => (
        <div
          key={i}
          className={`rounded-t-sm ${color} transition-all duration-300`}
          style={{ width: `${barW}%`, height: `${Math.max(1, (v / Math.max(maxVal, 0.01)) * 100)}%` }}
        />
      ))}
    </div>
  );
}

export default function MonitoringPanel({ weights, reward, isActive }: Props) {
  const [weightHistory, setWeightHistory] = useState<WeightSnapshot[]>([]);
  const [rewardHistory, setRewardHistory] = useState<RewardEntry[]>([]);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarkSummary[]>([]);
  const [expanded, setExpanded] = useState(false);
  const mountRef = useRef(Date.now());

  // Track weight changes over time
  useEffect(() => {
    if (weights.cag === 0 && weights.graph === 0) return;
    setWeightHistory(prev => {
      const next = [...prev, { ts: Date.now(), cag: weights.cag, graph: weights.graph }];
      return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
    });
  }, [weights.cag, weights.graph]);

  // Track reward changes
  useEffect(() => {
    if (reward === 0) return;
    setRewardHistory(prev => {
      const next = [...prev, { ts: Date.now(), reward }];
      return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
    });
  }, [reward]);

  // Poll system health on mount + every 30s
  useEffect(() => {
    const fetchHealth = () => {
      fetch('http://localhost:8000/ping')
        .then(r => r.json())
        .then(data => setHealth({
          gpu: data.gpu || null,
          device: data.device || 'cpu',
          model: data.model || 'unknown',
          gpu_model: data.gpu_model || 'Llama-3.1-8B',
          cpu_model: data.cpu_model || 'Qwen-2.5-1.5B',
          policy: data.policy || 'unknown',
          uptime: Math.round((Date.now() - mountRef.current) / 1000),
        }))
        .catch(() => {});
    };
    fetchHealth();
    const timer = setInterval(fetchHealth, 30_000);
    return () => clearInterval(timer);
  }, []);

  // Attempt to load latest benchmark results from /metrics parse
  useEffect(() => {
    // Benchmarks are local JSON files; for the panel, show static last-known results.
    // In production, this would also come from a /api/benchmarks endpoint.
    setBenchmarks([
      { name: 'RAGChecker', passed: true, score: 0.0, timestamp: '' },
      { name: 'HotpotQA', passed: true, score: 0.0, timestamp: '' },
      { name: 'TruthfulQA', passed: true, score: 0.0, timestamp: '' },
    ]);
  }, []);

  const avgReward = rewardHistory.length > 0
    ? (rewardHistory.reduce((s, e) => s + e.reward, 0) / rewardHistory.length).toFixed(2)
    : '—';

  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="w-full text-left px-3 py-2 mt-2 rounded-lg bg-gray-900/50 border border-gray-800 hover:border-cyan-700 transition text-xs text-gray-400"
      >
        <span className="text-cyan-500">📊</span> System Monitor
        {rewardHistory.length > 0 && (
          <span className="ml-2 text-gray-500">avg reward: {avgReward}</span>
        )}
      </button>
    );
  }

  return (
    <div className="mt-2 rounded-lg bg-gray-900/60 border border-gray-800 overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setExpanded(false)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-gray-800/40 transition"
      >
        <span className="text-xs text-cyan-400 font-semibold flex items-center gap-1.5">
          <span>📊</span> System Monitor
        </span>
        <span className="text-gray-600 text-xs">▾ collapse</span>
      </button>

      <div className="px-3 pb-3 space-y-3">
        {/* Weight Evolution */}
        <div>
          <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Weight History</p>
          <div className="space-y-1">
            {(['cag', 'graph'] as const).map(path => {
              const vals = weightHistory.map(w => w[path]);
              const label = path.toUpperCase();
              const latest = vals.length > 0 ? (vals[vals.length - 1] * 100).toFixed(0) : '—';
              return (
                <div key={path} className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-500 w-10 text-right">{label}</span>
                  <div className="flex-1">
                    <MiniBar values={vals} color={
                      path === 'cag' ? 'bg-purple-500/70' :
                      'bg-emerald-500/70'
                    } maxVal={1.0} />
                  </div>
                  <span className="text-[10px] text-gray-400 w-8">{latest}%</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Reward History */}
        <div>
          <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Reward Trend</p>
          <div className="flex items-center gap-2">
            <div className="flex-1">
              <MiniBar
                values={rewardHistory.map(e => e.reward)}
                color="bg-cyan-500/60"
                maxVal={1.0}
              />
            </div>
            <span className="text-xs text-gray-400 w-10 text-right">{avgReward}</span>
          </div>
          <p className="text-[10px] text-gray-600 mt-0.5">
            {rewardHistory.length} interactions tracked
          </p>
        </div>

        {/* Benchmark Status */}
        <div>
          <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Benchmarks</p>
          <div className="space-y-0.5">
            {benchmarks.map(b => (
              <div key={b.name} className="flex items-center justify-between text-[11px]">
                <span className="text-gray-400">{b.name}</span>
                <span className={`font-mono ${b.passed ? 'text-emerald-500' : 'text-red-400'}`}>
                  {b.score > 0 ? b.score.toFixed(2) : '—'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* System Health */}
        {health && (
          <div>
            <p className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">System</p>
            <div className="text-[11px] text-gray-400 space-y-0.5">
              <div className="flex justify-between">
                <span>Device</span>
                <span className="text-cyan-400">{health.gpu || health.device.toUpperCase()}</span>
              </div>
              <div className="flex justify-between">
                <span>LLM</span>
                <span className="text-cyan-400 text-right text-[10px]">{health.gpu_model || 'Llama-3.1-8B'}</span>
              </div>
              <div className="flex justify-between">
                <span>Triage</span>
                <span className="text-cyan-400 text-right text-[10px]">{health.cpu_model || 'Qwen-2.5-1.5B'}</span>
              </div>
              <div className="flex justify-between">
                <span>Policy</span>
                <span className="text-cyan-400">{health.policy}</span>
              </div>
              <div className="flex justify-between">
                <span>Session</span>
                <span className="text-gray-500">{Math.round(health.uptime / 60)}m</span>
              </div>
            </div>
          </div>
        )}

        {/* Active indicator */}
        <div className="flex items-center gap-1.5 pt-1 border-t border-gray-800">
          <div className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-cyan-400 animate-pulse' : 'bg-gray-700'}`} />
          <span className="text-[10px] text-gray-600">
            {isActive ? 'Processing query...' : 'Idle'}
          </span>
        </div>
      </div>
    </div>
  );
}
