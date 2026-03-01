// Author: Bradley R. Kinnard
// Shared type contracts for RLFusion frontend.
// Single source of truth: all data interfaces live here.

export interface Message {
  id: string;
  text: string;
  role: 'user' | 'rlfusion';
}

export interface Weights {
  cag: number;
  graph: number;
}

export interface WeightSnapshot {
  ts: number;
  cag: number;
  graph: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export interface AgentStatus {
  name: string;
  status: 'idle' | 'pending' | 'running' | 'done' | 'blocked' | 'skipped';
  detail?: string;
}

export interface SystemHealth {
  gpu: string | null;
  device: string;
  model: string;
  cpuModel: string;
  gpuModel: string;
  policy: string;
  uptime: number;
}

export interface RewardEntry {
  ts: number;
  reward: number;
}

export interface FusionWeightsResponse {
  cag: number;
  graph: number;
}

export interface ChatResponse {
  response: string;
  fusion_weights: FusionWeightsResponse;
  reward: number;
  proactive_suggestions: string[];
  blocked?: boolean;
  safety_reason?: string;
}

export interface WsDoneMessage {
  type: 'done';
  response: string;
  fusion_weights: FusionWeightsResponse;
  reward: number;
  proactive: string;
  proactive_suggestions: string[];
  query_expanded: boolean;
  expanded_query: string | null;
  web_status: string;
}
