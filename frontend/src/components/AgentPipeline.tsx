// Author: Bradley R. Kinnard
// AgentPipeline — live agent status with real-time detail from each agent
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
  fusion:     { icon: '⚗️', label: 'Fusion' },
  generation: { icon: '🧠', label: 'Generation' },
  critique:   { icon: '📝', label: 'Critique' },
};

const STATUS_DOT: Record<AgentStatus['status'], string> = {
  idle:    'bg-gray-700/60',
  pending: 'bg-amber-500/50',
  running: 'bg-cyan-400',
  done:    'bg-emerald-500/70',
  blocked: 'bg-red-500',
  skipped: 'bg-gray-700/30',
};

const STATUS_LABEL: Record<AgentStatus['status'], string> = {
  idle:    '',
  pending: 'queued',
  running: 'working...',
  done:    'complete',
  blocked: 'blocked',
  skipped: 'skipped',
};

function Connector({ active }: { active: boolean }) {
  return (
    <div className="flex justify-center py-0.5">
      <div className={`w-0.5 h-3 rounded-full transition-colors duration-500 ${
        active ? 'bg-emerald-500/30' : 'bg-gray-800/60'
      }`} />
    </div>
  );
}

function AgentRow({ agent }: { agent: AgentStatus }) {
  const meta = AGENT_META[agent.name] || { icon: '⚙️', label: agent.name };
  const isRunning = agent.status === 'running';
  const isDone = agent.status === 'done';
  const isBlocked = agent.status === 'blocked';
  const isPending = agent.status === 'pending';
  const hasDetail = !!agent.detail;

  return (
    <div className={`rounded-lg transition-all duration-300 ${
      isRunning ? 'bg-cyan-500/8 border border-cyan-500/20' :
      isBlocked ? 'bg-red-500/8 border border-red-500/20' :
      'border border-transparent'
    }`}>
      {/* main row */}
      <div className="flex items-center gap-2.5 px-3 py-1.5">
        {/* status dot */}
        <div className="relative flex-shrink-0">
          <div className={`w-2 h-2 rounded-full transition-all duration-300 ${STATUS_DOT[agent.status]}`} />
          {isRunning && (
            <div className="absolute inset-0 w-2 h-2 rounded-full bg-cyan-400 animate-ping opacity-40" />
          )}
        </div>

        {/* icon */}
        <span className={`text-sm flex-shrink-0 transition-opacity duration-300 ${
          isRunning ? 'opacity-100' :
          isDone ? 'opacity-70' :
          isPending ? 'opacity-50' :
          'opacity-30'
        }`}>{meta.icon}</span>

        {/* label */}
        <span className={`text-xs flex-1 transition-all duration-300 ${
          isRunning ? 'text-cyan-300 font-medium' :
          isBlocked ? 'text-red-400' :
          isDone ? 'text-gray-400' :
          isPending ? 'text-gray-500' :
          'text-gray-600'
        }`}>
          {meta.label}
        </span>

        {/* status text */}
        <span className={`text-[10px] font-mono transition-all duration-300 ${
          isRunning ? 'text-cyan-400 font-semibold' :
          isBlocked ? 'text-red-400' :
          isDone ? 'text-emerald-500/70' :
          isPending ? 'text-amber-500/60' :
          'text-gray-700'
        }`}>
          {STATUS_LABEL[agent.status]}
        </span>
      </div>

      {/* detail line — shows what the agent is doing or did */}
      {hasDetail && (agent.status === 'running' || agent.status === 'done' || agent.status === 'blocked') && (
        <div className={`px-3 pb-1.5 pl-[2.1rem] transition-all duration-300`}>
          <p className={`text-[10px] leading-tight ${
            isRunning ? 'text-cyan-400/70 animate-pulse' :
            isBlocked ? 'text-red-400/70' :
            'text-gray-500'
          }`}>
            {agent.detail}
          </p>
        </div>
      )}
    </div>
  );
}

function PipelineTimer({ active, elapsedMs }: { active: boolean; elapsedMs: number }) {
  const [liveElapsed, setLiveElapsed] = useState(0);
  const startRef = useRef(0);

  // tick every 100ms while pipeline is running
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

  // while active, show the live counter; when done, show final elapsed from parent
  const displayMs = active ? liveElapsed : elapsedMs;
  if (displayMs === 0 && !active) return null;

  const seconds = (displayMs / 1000).toFixed(1);
  return (
    <div className="flex items-center justify-between text-[10px] px-1 mt-1.5">
      <span className={active ? 'text-cyan-400/60' : 'text-gray-500'}>
        {active ? 'Pipeline running' : 'Pipeline complete'}
      </span>
      <span className="font-mono text-gray-500">{seconds}s</span>
    </div>
  );
}

export default function AgentPipeline({ agents, isActive, elapsedMs }: Props) {
  const allIdle = agents.every(a => a.status === 'idle');

  return (
    <div className="pt-4 border-t border-gray-800">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm">🤖</span>
        <span className="text-xs text-[var(--accent)] uppercase tracking-wider font-medium">Agent Pipeline</span>
        {isActive && (
          <span className="ml-auto flex items-center gap-1.5">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-cyan-400" />
            </span>
            <span className="text-[9px] text-cyan-400 font-semibold uppercase tracking-widest">Live</span>
          </span>
        )}
      </div>

      <div className="space-y-0">
        {agents.map((agent, i) => {
          const prevDone = i > 0 && ['done', 'running', 'blocked'].includes(agents[i - 1].status);
          return (
            <div key={agent.name}>
              {i > 0 && <Connector active={prevDone} />}
              <AgentRow agent={agent} />
            </div>
          );
        })}
      </div>

      <PipelineTimer active={isActive} elapsedMs={elapsedMs} />

      {allIdle && !isActive && elapsedMs === 0 && (
        <p className="text-[10px] text-gray-700 text-center mt-2">
          Waiting for query
        </p>
      )}
    </div>
  );
}
