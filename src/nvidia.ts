// Thin OpenAI-compatible REST clients for NVIDIA NIM (embed + chat). The Worker
// talks to NVIDIA directly over plain fetch — no SDK, no Node runtime needed.

const NVIDIA_BASE = 'https://integrate.api.nvidia.com/v1';
const EMBED_MODEL = 'nvidia/nemotron-3-embed-1b';
const CHAT_MODEL = 'openai/gpt-oss-20b';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export async function embed(apiKey: string, input: string): Promise<number[]> {
  const res = await fetch(`${NVIDIA_BASE}/embeddings`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({ model: EMBED_MODEL, input }),
  });
  if (!res.ok) throw new Error(`Embedding request failed (${res.status}).`);
  const data = (await res.json()) as { data?: Array<{ embedding?: number[] }> };
  const vec = data.data?.[0]?.embedding;
  if (!vec || vec.length === 0) throw new Error('Embedding response was empty.');
  return vec;
}

const chatPayload = (messages: ChatMessage[], stream: boolean) => ({
  model: CHAT_MODEL,
  messages,
  temperature: 0.6,
  top_p: 0.95,
  max_tokens: 700,
  stream,
});

// Streaming chat: returns the upstream Response. The caller pipes its body to
// the client, translating NVIDIA SSE frames into the app's delta/done frames.
export function streamChat(apiKey: string, messages: ChatMessage[], signal?: AbortSignal): Promise<Response> {
  return fetch(`${NVIDIA_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(chatPayload(messages, true)),
    signal,
  });
}

// Non-streaming chat (used for the JSON output mode and error fallback).
export async function chatOnce(apiKey: string, messages: ChatMessage[]): Promise<string> {
  const res = await fetch(`${NVIDIA_BASE}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(chatPayload(messages, false)),
  });
  if (!res.ok) throw new Error(`Chat request failed (${res.status}).`);
  const data = (await res.json()) as { choices?: Array<{ message?: { content?: string | null } }> };
  return (data.choices?.[0]?.message?.content || '').trim();
}