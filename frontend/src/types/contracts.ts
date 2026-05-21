// Author: Bradley R. Kinnard
// Shared type contracts for RLFusion frontend.
// Single source of truth: all data interfaces live here. Backend mirror
// in backend/main.py + backend/api/models.py. Drift caught by
// tests/test_asymmetric.py::TestFrontendContracts.

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

// Mirrors GET /ping. cpuModel / gpuModel / uptime fields used to live here
// but the backend never returned them; F5.4 dropped them to match reality.
export interface SystemHealth {
  status: 'alive';
  gpu: string | null;
  device: 'cuda' | 'cpu';
  model: string;
  inferenceEngine: string;
  engineResolution: string;
  policy: string;
  policyExists: boolean;
  bootId: string;
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

// Discriminated union of every server-pushed frame. The frontend's WS
// handler switches on `type` and TS narrows the body for each variant.
export interface WsStartMessage {
  type: 'start';
}

export interface WsPipelineMessage {
  type: 'pipeline';
  agents: AgentStatus[];
}

// Streaming chunk frame: emitted multiple times before the final `done`.
// `weights` is normalised to an object on the server so the frontend does
// not need a positional/object guard.
export interface WsChunkMessage {
  chunk: string;
  weights: FusionWeightsResponse;
  reward: number;
}

export interface WsTokenMessage {
  type: 'token';
  token: string;
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
  blocked?: boolean;
  safety_reason?: string;
}

export interface WsCritiqueMessage {
  type: 'critique';
  reward: number;
  proactive_suggestions: string[];
  response: string;
}

export interface WsMemoryClearedMessage {
  type: 'memory_cleared';
}

export type WsServerMessage =
  | WsStartMessage
  | WsPipelineMessage
  | WsChunkMessage
  | WsTokenMessage
  | WsDoneMessage
  | WsCritiqueMessage
  | WsMemoryClearedMessage;
