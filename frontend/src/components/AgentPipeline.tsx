// Author: Bradley R. Kinnard
// AgentPipeline — polished multi-agent status with live detail + STIS awareness
import { useEffect, useRef, useState } from 'react';

export interface AgentStatus {
  name: string;
  status: 'idle' | 'pending' | 'running' | 'done' | 'blocked' | 'skipped';
  detail?: string;
}

interface Props {
  agents: AgentStatus[];
  isActive: boolean;
  elapsedMs: number;
}

const AGENT_META: Record<string, { icon: string; label: string }> = {
  safety:     { icon: '🛡️', label: 'Safety Gate' },
  retrieval:  { icon: '🔍', label: 'Retrieval' },
  fusion:     { icon: '⚗️', label: 'RL Fusion' },
  generation: { icon: '🧠', label: 'Generation' },
  critique:   { icon: '📝', label: 'Critique' },
};

function VerticalConnector({ active }: { active: boolean }) {
  return (
    <div className="flex items-center ml-[17px] h-3.5">
      <div className={`w-px h-full transition-colors duration-500 ${
        active ? 'bg-gradient-to-b from-emerald-500/40 to-emerald-500/10' : 'bg-gray-800'
      }`} />
    </div>
  );
}

function AgentRow({ agent, index }: { agent: AgentStatus; index: number }) {
  const meta = AGENT_META[agent.name] || { icon: '⚙️', label: agent.name };
  const isRunning = agent.status === 'running';
  const isDone = agent.status === 'done';
  const isBlocked = agent.status === 'blocked';
  const hasDetail = !!agent.detail;
  const isStis = hasDetail && agent.detail!.includes('STIS');
  const stepNum = index + 1;

  return (
    <div className={`relative rounded-lg transition-all duration-300 ${
      isRunning ? 'bg-cyan-500/[0.06] border border-gray-700/50' :
      isBlocked ? 'bg-red-500/[0.06] border border-red-500/20' :
      'border border-transparent'
    }`}>
      <div className="flex items-center gap-2.5 px-2.5 py-[7px]">
        {/* step circle */}
        <div className="relative flex-shrink-0 w-[26px] h-[26px] flex items-center justify-center">
          <div className={`absolute inset-0 rounded-full border transition-all duration-300 ${
            isRunning ? 'border-cyan-500/40 bg-cyan-500/[0.08]' :
            isDone ? 'border-emerald-500/30 bg-emerald-500/[0.06]' :
            isBlocked ? 'border-red-500/30 bg-red-500/[0.06]' :
            'border-gray-800 bg-gray-900/40'
          }`} />
          {isDone ? (
            <svg className="w-3 h-3 text-emerald-500 relative z-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          ) : isBlocked ? (
            <svg className="w-3 h-3 text-red-400 relative z-10" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <span className={`relative z-10 text-[9px] font-bold transition-colors duration-300 ${
              isRunning ? 'text-cyan-400' : 'text-gray-600'
            }`}>{stepNum}</span>
          )}
          {isRunning && (
            <div className="absolute inset-0 rounded-full border border-cyan-400/30 animate-ping" />
          )}
        </div>

        {/* icon + label */}
        <div className="flex items-center gap-1.5 flex-1 min-w-0">
          <span className={`text-xs transition-opacity duration-300 ${
            isRunning || isDone ? 'opacity-100' : 'opacity-35'
          }`}>{meta.icon}</span>
          <span className={`text-[12px] font-medium tracking-tight transition-all duration-300 ${
            isRunning ? 'text-white' :
            isDone ? 'text-gray-300' :
            isBlocked ? 'text-red-300' :
            'text-gray-500'
          }`}>
            {meta.label}
          </span>
        </div>

        {/* status badge */}
        <div className="flex-shrink-0">
          {isRunning && (
            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/20">
              <span className="relative flex h-1.5 w-1.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-cyan-400" />
              </span>
              <span className="text-[8px] text-cyan-400 font-bold uppercase tracking-wider">Active</span>
            </span>
          )}
          {isDone && (
            <span className="text-[9px] text-emerald-500/50 font-mono">done</span>
          )}
          {isBlocked && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded-full bg-red-500/10 border border-red-500/20">
              <span className="text-[8px] text-red-400 font-bold uppercase tracking-wider">Blocked</span>
            </span>
          )}
        </div>
      </div>

      {/* detail line */}
      {hasDetail && (agent.status === 'running' || agent.status === 'done' || agent.status === 'blocked') && (
        <div className="px-2.5 pb-2 pl-[46px]">
          <p className={`text-[10px] leading-snug font-mono ${
            isStis ? 'text-violet-400/70' :
            isRunning ? 'text-cyan-400/50' :
            isBlocked ? 'text-red-400/50' :
            'text-gray-500/60'
          } ${isRunning ? 'animate-pulse' : ''}`}>
            {isStis && <span className="text-violet-400 font-semibold not-italic">⚡ STIS </span>}
            {isStis ? agent.detail!.replace('STIS ', '').replace('STIS', '') : agent.detail}
          </p>
        </div>
      )}
    </div>
  );
}

function PipelineTimer({ active, elapsedMs }: { active: boolean; elapsedMs: number }) {
  const [liveElapsed, setLiveElapsed] = useState(0);
  const startRef = useRef(0);

  useEffect(() => {
    if (active) {
      startRef.current = Date.now();
      setLiveElapsed(0);
      const interval = setInterval(() => {
        setLiveElapsed(Date.now() - startRef.current);
      }, 100);
      return () => clearInterval(interval);
    }
  }, [active]);

  const displayMs = active ? liveElapsed : elapsedMs;
  if (displayMs === 0 && !active) return null;

  const seconds = (displayMs / 1000).toFixed(1);
  return (
    <div className={`flex items-center justify-between px-3 py-2 mt-2 rounded-lg text-[11px] transition-all duration-300 ${
      active ? 'bg-cyan-500/[0.04] border border-cyan-500/10' : 'bg-gray-800/20 border border-gray-800/30'
    }`}>
      <span className={active ? 'text-cyan-400/70' : 'text-gray-500'}>
        {active ? '● Pipeline running' : '● Completed'}
      </span>
      <span className={`font-mono font-medium ${active ? 'text-cyan-400' : 'text-gray-400'}`}>
        {seconds}s
      </span>
    </div>
  );
}

export default function AgentPipeline({ agents, isActive, elapsedMs }: Props) {
  const allIdle = agents.every(a => a.status === 'idle');

  return (
    <div className="pt-4 border-t border-gray-800/60">
      {/* header */}
      <div className="flex items-center gap-2 mb-3 px-1">
        <div className="flex items-center justify-center w-5 h-5 rounded bg-cyan-500/10 border border-cyan-500/20">
          <span className="text-[10px]">🤖</span>
        </div>
        <span className="text-[11px] text-gray-400 uppercase tracking-widest font-semibold">
          Agent Pipeline
        </span>
        {isActive && (
          <span className="ml-auto flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/20">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-cyan-400" />
            </span>
            <span className="text-[9px] text-cyan-400 font-bold uppercase tracking-widest">Live</span>
          </span>
        )}
      </div>

      {/* agent steps */}
      <div className="space-y-0">
        {agents.map((agent, i) => {
          const prevDone = i > 0 && ['done', 'running', 'blocked'].includes(agents[i - 1].status);
          return (
            <div key={agent.name}>
              {i > 0 && <VerticalConnector active={prevDone} />}
              <AgentRow agent={agent} index={i} />
            </div>
          );
        })}
      </div>

      <PipelineTimer active={isActive} elapsedMs={elapsedMs} />

      {/* idle state */}
      {allIdle && !isActive && elapsedMs === 0 && (
        <div className="flex items-center justify-center gap-2 mt-3 py-3 rounded-lg bg-gray-800/20 border border-gray-800/40">
          <span className="text-gray-700 text-[10px]">●</span>
          <span className="text-[10px] text-gray-600 tracking-wide">Awaiting query</span>
        </div>
      )}
    </div>
  );
}
