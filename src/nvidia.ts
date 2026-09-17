// Thin OpenAI-compatible REST clients for NVIDIA NIM (embed + chat). The Worker
// talks to NVIDIA directly over plain fetch — no SDK, no Node runtime needed.

import type { Env } from './env';

const NVIDIA_BASE = 'https://integrate.api.nvidia.com/v1';
// nemotron-3-embed-1b is 2048-dimensional. Cloudflare's Workers AI embedding
// models top out at 1024 dims, so an edge-native binding can't fill the
// existing Pinecone index without re-embedding every vector. The latency win is
// instead captured by firing this call in parallel with auth (see chat.ts).
const EMBED_MODEL = 'nvidia/nemotron-3-embed-1b';

// NVIDIA gates models PER REQUEST-SOURCE: a model that answers from one egress
// (e.g. a home/US IP) can return 400/404 from Cloudflare's. The worker used to
// pin one model, so when that model was unavailable the chat had nothing to fall
// back to. This battery is tried in order; the fast 20B model stays primary and
// the rest mirror the Express server's verified-good set. Override at deploy
// time with the NVIDIA_CHAT_MODELS var (comma-separated) without a code change.
const DEFAULT_CHAT_MODELS = [
  'openai/gpt-oss-20b',
  'meta/muse-glimmer-30b',
  'z-ai/glm-5.3-flash',
  'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning',
];

export function chatModels(env: Env): string[] {
  const raw = env.NVIDIA_CHAT_MODELS;
  if (!raw) return DEFAULT_CHAT_MODELS;
  const list = raw
    .split(',')
    .map(m => m.trim())
    .filter(Boolean);
  return list.length > 0 ? list : DEFAULT_CHAT_MODELS;
}

// Hard per-request ceilings. Without these a provider that accepts the
// connection but never streams leaves the chat stuck on "AI is thinking..."
// forever (the UI has no client-side timeout either). The embedding is fast, so
// it gets a short leash; chat attempts are longer but still bounded, and the
// caller additionally enforces an overall wall-clock deadline.
const EMBED_TIMEOUT_MS = 20_000;
export const CHAT_ATTEMPT_TIMEOUT_MS = 45_000;

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

// Combine the caller's abort signal (client disconnect) with a hard timeout so
// the request — INCLUDING body streaming — is always torn down.
function requestSignal(external: AbortSignal | undefined, ms: number): AbortSignal {
  const timeout = AbortSignal.timeout(ms);
  return external ? AbortSignal.any([external, timeout]) : timeout;
}

export async function embed(
  apiKey: string,
  input: string,
  timeoutMs = EMBED_TIMEOUT_MS
): Promise<number[]> {
  const res = await fetch(`${NVIDIA_BASE}/embeddings`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({ model: EMBED_MODEL, input }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(`Embedding request failed (${res.status}).`);
  const data = (await res.json()) as { data?: Array<{ embedding?: number[] }> };
  const vec = data.data?.[0]?.embedding;
  if (!vec || vec.length === 0) throw new Error('Embedding response was empty.');
  return vec;
}

const chatPayload = (model: string, messages: ChatMessage[], stream: boolean) => ({
  model,
  messages,
  temperature: 0.6,
  top_p: 0.95,
  max_tokens: 500,
  stream,
});

// Streaming chat: returns the upstream Response. The caller pipes its body to
// the client, translating NVIDIA SSE frames into the app's delta/done frames.
export function streamChat(
  apiKey: string,
  model: string,
  messages: ChatMessage[],
  signal?: AbortSignal,
  timeoutMs = CHAT_ATTEMPT_TIMEOUT_MS
): Promise<Response> {
  return fetch(`${NVIDIA_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(chatPayload(model, messages, true)),
    signal: requestSignal(signal, timeoutMs),
  });
}

// Non-streaming chat (used for the JSON output mode and error fallback).
export async function chatOnce(
  apiKey: string,
  model: string,
  messages: ChatMessage[],
  timeoutMs = CHAT_ATTEMPT_TIMEOUT_MS
): Promise<string> {
  const res = await fetch(`${NVIDIA_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(chatPayload(model, messages, false)),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(`Chat request failed (${res.status}).`);
  const data = (await res.json()) as { choices?: Array<{ message?: { content?: string | null } }> };
  return (data.choices?.[0]?.message?.content || '').trim();
}
