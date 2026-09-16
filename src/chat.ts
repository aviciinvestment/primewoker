import type { Env } from './env';
import { verifyFirebaseToken } from './auth';
import { embed, streamChat, chatOnce, type ChatMessage } from './nvidia';
import { queryIndex } from './pinecone';
import { cacheGet, cacheSet, embedCacheGet, embedCacheSet, checkUserRateLimit } from './cache';
import { fetchOpportunityContext, recordComplaint } from './backend';
import {
  OFF_TOPIC_REFUSAL,
  isOffTopic,
  hasMentorshipIntent,
  hasMentorshipComplaint,
  buildMentorshipReply,
  buildComplaintReply,
  makeTicket,
  buildSystemPrompt,
} from '../../shared/chatPolicies';

// Each user's CV vectors live in their OWN namespace (cvs-<userId>) so a chat
// query can never retrieve another user's CV data. Mirrors the server.
const CV_NAMESPACE_PREFIX = 'cvs-';

const SSE_HEADERS: Record<string, string> = {
  'Content-Type': 'text/event-stream',
  'Cache-Control': 'no-cache',
  'Connection': 'keep-alive',
  'X-Accel-Buffering': 'no',
};

const json = (status: number, data: unknown): Response =>
  new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });

const frame = (obj: unknown): string => `data: ${JSON.stringify(obj)}\n\n`;

// A single 'done' frame (cached reply, guardrail, mentorship intent, ...).
function doneSseResponse(reply: string, action: { type: 'mentorship' } | null): Response {
  return new Response(frame({ type: 'done', reply, action: action || null }), {
    headers: SSE_HEADERS,
  });
}

// Pipe NVIDIA's streaming chat through while translating SSE frames into the
// app's delta/done protocol. The full reply is assembled so it can be cached.
async function streamAssistantReply(env: Env, messages: ChatMessage[], cacheKey: string): Promise<Response> {
  const abort = new AbortController();
  const upstream = await streamChat(env.NVIDIA_API_KEY, messages, abort.signal);

  if (!upstream.ok || !upstream.body) {
    const reply = await chatOnce(env.NVIDIA_API_KEY, messages);
    await cacheSet(env, cacheKey, reply);
    return doneSseResponse(reply, null);
  }

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let reply = '';

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        let buffer = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let idx: number;
          while ((idx = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (!line.startsWith('data:')) continue;
            const payload = line.slice(5).trim();
            if (!payload || payload === '[DONE]') continue;

            let chunk: any;
            try {
              chunk = JSON.parse(payload);
            } catch {
              continue;
            }
            const delta = chunk?.choices?.[0]?.delta?.content;
            if (typeof delta === 'string' && delta.length > 0) {
              reply += delta;
              controller.enqueue(encoder.encode(frame({ type: 'delta', text: delta })));
            }
          }
        }

        if (!reply.trim()) reply = 'Sorry, I could not generate a response. Please try again.';
        await cacheSet(env, cacheKey, reply);
        controller.enqueue(encoder.encode(frame({ type: 'done', reply, action: null })));
        controller.close();
      } catch (err) {
        if (!abort.signal.aborted) {
          controller.enqueue(
            encoder.encode(frame({ type: 'error', message: 'Failed to process chat message.' }))
          );
        }
        controller.close();
      }
    },
    cancel() {
      // Client disconnected mid-stream: stop paying for the upstream LLM call.
      abort.abort();
      reader.cancel().catch(() => {});
    },
  });

  return new Response(stream, { headers: SSE_HEADERS });
}

// Bound cost + keep raw conversation text out of the KV cache: the cache key
// is a SHA-256 hash of uid|message, and long messages are capped before use.
const MAX_MESSAGE_LENGTH = 2000;

async function sha256Hex(input: string): Promise<string> {
  const data = new TextEncoder().encode(input);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

const hashCacheKey = (uid: string, message: string) => sha256Hex(`${uid}|${message}`);

// Embed with a local, hash-keyed in-memory cache: repeat text never hits the
// NVIDIA paid endpoint. The key is sha256(text) — no PII retained, and shared
// across users (embeddings are content-derived, not user-scoped).
async function embedCached(apiKey: string, text: string): Promise<number[]> {
  const key = await sha256Hex(text);
  const cached = embedCacheGet(key);
  if (cached) return cached;
  const vector = await embed(apiKey, text);
  embedCacheSet(key, vector);
  return vector;
}

// The chat pipeline. Mirrors server/src/controllers/aiController.ts chatWithAI
// with the Mongo steps replaced by backend() calls.
export async function handleChat(request: Request, env: Env, authHeader: string | null, clientIp: string | null): Promise<Response> {
  let body: any;
  try {
    body = await request.json();
  } catch {
    return json(400, { success: false, error: 'Invalid JSON body.' });
  }

  const message = String(body?.message || '').trim();
  if (!message) return json(400, { success: false, error: 'Message is required.' });
  // Strict cap (S14): reject over-limit input outright — fail closed rather than
  // truncating silently — to bound embed + LLM token cost per request.
  if (message.length > MAX_MESSAGE_LENGTH) {
    return json(400, {
      success: false,
      error: `Message exceeds the ${MAX_MESSAGE_LENGTH} character limit.`,
    });
  }

  const stream = body?.stream === true;
  const rawHistory = Array.isArray(body?.history) ? body.history : [];
  const history: ChatMessage[] = rawHistory
    .filter(
      (m: any) =>
        m &&
        (m.role === 'user' || m.role === 'assistant') &&
        typeof m.content === 'string'
    )
    .slice(-10)
    .map((m: any) => ({ role: m.role, content: m.content }));

  // Identity comes from the verified Firebase token (never the request body).
  let user;
  try {
    user = await verifyFirebaseToken(env, authHeader);
  } catch {
    return json(401, { success: false, error: 'Invalid or expired session.' });
  }
  const uid = user.uid;

  // Per-user ceiling (20 req/min) enforced in KV by verified uid. Fail-open on
  // KV errors; when exceeded, reject with 429 before any paid NVIDIA call.
  const rl = await checkUserRateLimit(env, uid);
  if (!rl.allowed) {
    return json(429, {
      success: false,
      error: 'Rate limit exceeded. Please try again in a moment.',
      retryAfterMs: rl.retryAfterMs,
    });
  }

  const cacheKey = await hashCacheKey(uid, message);
  const cached = await cacheGet(env, cacheKey);
  if (cached) return stream ? doneSseResponse(cached, null) : json(200, { success: true, reply: cached });

  // Hard guardrail: refuse clearly out-of-scope questions without an LLM call.
  if (isOffTopic(message)) {
    const reply = OFF_TOPIC_REFUSAL;
    return stream ? doneSseResponse(reply, null) : json(200, { success: true, reply });
  }

  // Paid-but-no-mentor escalation: write the complaint server-side (Mongo) and
  // stream back the reply with the ticket number.
  if (hasMentorshipComplaint(message)) {
    try {
      const reply = await recordComplaint(env, authHeader, message, clientIp);
      return stream ? doneSseResponse(reply, null) : json(200, { success: true, reply });
    } catch {
      const fallback = buildComplaintReply(user.email || '', makeTicket());
      return stream ? doneSseResponse(fallback, null) : json(200, { success: true, reply: fallback });
    }
  }

  // Mentorship request -> deterministic reply with an in-chat link to the
  // purchase/guidance page (client renders the clickable action button).
  if (hasMentorshipIntent(message)) {
    const fee = env.MENTORSHIP_FEE || '20000';
    const currency = env.MENTORSHIP_CURRENCY || 'NGN';
    const reply = buildMentorshipReply(fee, currency);
    if (stream) return doneSseResponse(reply, { type: 'mentorship' });
    return json(200, { success: true, reply, action: { type: 'mentorship' } });
  }

  // RAG pipeline: embed -> Pinecone (public + user's private CV namespace) ->
  // full opportunity context from Render -> LLM.
  try {
    const messageVector = await embedCached(env.NVIDIA_EMBED_API_KEY, message);

    const matches = await queryIndex(env, messageVector, 5);
    const matchIds = matches.map(m => m.id);

    let retrievedContext =
      'No specific opportunities were retrieved for this question. Answer generally using your knowledge.';
    if (matchIds.length > 0) {
      const ctx = await fetchOpportunityContext(env, authHeader, matchIds, clientIp);
      if (ctx) retrievedContext = ctx;
    }

    let userCvContext = '';
    try {
      const cvMatches = await queryIndex(env, messageVector, 4, `${CV_NAMESPACE_PREFIX}${uid}`);
      const chunks = cvMatches
        .filter(m => m.metadata && typeof m.metadata.text === 'string')
        .map(m => m.metadata!.text as string);
      if (chunks.length > 0) userCvContext = chunks.join('\n\n');
    } catch {
      /* user has no CV vectors — assistant answers generally */
    }

    const system = buildSystemPrompt({ userCvContext, retrievedContext });
    const messages: ChatMessage[] = [
      { role: 'system', content: system },
      ...history,
      { role: 'user', content: message },
    ];

    if (stream) return streamAssistantReply(env, messages, cacheKey);

    const reply = await chatOnce(env.NVIDIA_API_KEY, messages);
    await cacheSet(env, cacheKey, reply);
    return json(200, { success: true, reply });
  } catch (err) {
    console.error('AI chat error:', err);
    if (stream) {
      return new Response(frame({ type: 'error', message: 'Failed to process chat message.' }), {
        headers: SSE_HEADERS,
      });
    }
    return json(500, { success: false, error: 'Failed to process chat message.' });
  }
}