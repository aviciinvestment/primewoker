import type { Env } from './env';
import { verifyFirebaseToken } from './auth';
import { embed, streamChat, chatOnce, chatModels, CHAT_ATTEMPT_TIMEOUT_MS, type ChatMessage } from './nvidia';
import { queryIndex, type PineconeMatch } from './pinecone';
import { cacheGet, cacheSet, embedCacheGet, embedCacheSet, checkUserRateLimit } from './cache';
import { recordComplaint } from './backend';
import {
  OFF_TOPIC_REFUSAL,
  isOffTopic,
  hasMentorshipIntent,
  hasMentorshipComplaint,
  buildMentorshipReply,
  buildComplaintReply,
  makeTicket,
  serializeOpportunities,
  buildSystemPrompt,
  type OpportunityLite,
} from '../../shared/chatPolicies';

// Each user's CV vectors live in their OWN namespace (cvs-<userId>) so a chat
// query can never retrieve another user's CV data. Mirrors the server.
const CV_NAMESPACE_PREFIX = 'cvs-';

const SSE_HEADERS: Record<string, string> = {
  'Content-Type': 'text/event-stream',
  // no-store + no-transform stop proxies (incl. Cloudflare) from buffering or
  // compressing the stream, which would otherwise hold every frame back until
  // the whole reply is done and the browser would get one big burst.
  'Cache-Control': 'no-cache, no-store, no-transform, max-age=0',
  'Connection': 'keep-alive',
  'X-Accel-Buffering': 'no',
  'X-Content-Type-Options': 'nosniff',
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

// Visible step labels sent as { type:'status' } frames while the pipeline runs,
// so the UI can show the user exactly what the assistant is doing.
const STATUS_EMBEDDING = 'Embedding your message...';
const STATUS_SEARCHING = 'Searching opportunities and your CV...';
const STATUS_THINKING = 'AI is thinking...';

// Adaptive rank pruning (prompt payload minimization): only high-confidence
// matches earn a slot in the system prompt, and the totals stay capped so the
// LLM's pre-read — and therefore every first token — finish faster. Matches the
// serialized server-side format byte-for-byte via serializeOpportunities.
const MIN_MATCH_SCORE = 0.75;
const MAX_PUBLIC_MATCHES = 3;
const MAX_CV_CHUNKS = 2;
const CV_CHUNK_MAX_CHARS = 600;

// Every RAG field the prompt needs is denormalized inside Pinecone metadata, so
// context never waits on a backend round-trip. These helpers pull the values
// out, preferring the canonical Mongo names with the denormalized aliases
// (org / link / full_description / eligibility) as fallbacks for old vectors.
function mdStr(m: PineconeMatch, key: string): string {
  const v = m.metadata?.[key];
  return typeof v === 'string' ? v : '';
}

function mdStrList(m: PineconeMatch, key: string): string[] {
  const v = m.metadata?.[key];
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

// Reconstruct an opportunity from its vector's metadata so serializeOpportunities
// can render the exact same context block the Express server used to build.
function matchToOpportunity(m: PineconeMatch): OpportunityLite {
  return {
    _id: m.id,
    title: mdStr(m, 'title'),
    organization: mdStr(m, 'organization') || mdStr(m, 'org'),
    opportunityType: mdStr(m, 'opportunityType'),
    category: mdStr(m, 'category'),
    location: mdStr(m, 'location'),
    eligibleFields: mdStrList(m, 'eligibleFields'),
    eligibleEducationLevels: mdStrList(m, 'eligibleEducationLevels'),
    targetAudience: mdStrList(m, 'targetAudience'),
    deadline: mdStr(m, 'deadline') || null,
    status: mdStr(m, 'status'),
    tags: mdStrList(m, 'tags'),
    description: mdStr(m, 'description') || mdStr(m, 'full_description'),
    officialUrl: mdStr(m, 'officialUrl') || mdStr(m, 'link'),
  };
}

async function buildRagMessages(
  env: Env,
  uid: string,
  message: string,
  history: ChatMessage[],
  preEmbed: Promise<number[]> | null,
  emit?: (obj: unknown) => void
): Promise<ChatMessage[]> {
  // RAG pipeline: embed -> Pinecone (public + user's private CV namespace) ->
  // full opportunity context from Pinecone metadata -> LLM answers. No backend
  // round-trip: every field the prompt needs is already in the vector payload.
  emit?.({ type: 'status', step: STATUS_EMBEDDING });
  const messageVector = await (preEmbed ?? embedCached(env.NVIDIA_EMBED_API_KEY, message));

  // 'Searching...' is flushed right before the queries resolve on the wire, so
  // the browser shows the step while the vector RTTs are still in flight.
  emit?.({ type: 'status', step: STATUS_SEARCHING });
  const [matches, cvMatches] = await Promise.all([
    queryIndex(env, messageVector, 5),
    queryIndex(env, messageVector, 4, `${CV_NAMESPACE_PREFIX}${uid}`).catch(() => []),
  ]);

  // Adaptive rank pruning: drop low-confidence matches and cap at the tightest
  // set that still answers the question. Shorter prompt = faster first token.
  const highlyRelevant = matches
    .filter((m): m is PineconeMatch & { score: number } => typeof m.score === 'number' && m.score >= MIN_MATCH_SCORE)
    .slice(0, MAX_PUBLIC_MATCHES);

  const retrievedContext = highlyRelevant.length
    ? serializeOpportunities(highlyRelevant.map(matchToOpportunity))
    : 'No specific opportunities were retrieved for this question. Answer generally using your knowledge.';

  // Personalisation from the user's private CV namespace. Threshold-pruned like
  // the public set, with the single best chunk as a floor so a user with a real
  // CV never loses personalised context when every chunk scores just below the
  // bar. Each chunk is capped so the prefill stays small.
  const cvAbove = cvMatches.filter(
    (m): m is PineconeMatch & { score: number } => typeof m.score === 'number' && m.score >= MIN_MATCH_SCORE
  );
  let userCvContext = '';
  const cvChunks = (cvAbove.length > 0 ? cvAbove : cvMatches.slice(0, 1))
    .filter((m): m is PineconeMatch & { metadata: Record<string, string> } =>
      !!m.metadata && typeof (m.metadata as Record<string, string>).text === 'string'
    )
    .map(m => m.metadata.text as string)
    .slice(0, MAX_CV_CHUNKS)
    .map(c => c.slice(0, CV_CHUNK_MAX_CHARS));
  if (cvChunks.length > 0) userCvContext = cvChunks.join('\n\n');

  const system = buildSystemPrompt({ userCvContext, retrievedContext });
  return [
    { role: 'system', content: system },
    ...history,
    { role: 'user', content: message },
  ];
}

// Streaming response that runs the whole pipeline lazily, emitting a visible
// {type:'status'} frame at each real stage, then the model's tokens as
// {type:'delta'} frames. The first status frame is enqueued synchronously when
// the stream starts, so headers + the loading animation flush to the browser
// immediately while the async RAG work continues over the wire. The user is
// carried along instead of staring at a frozen "Thinking..." bubble.
function ragStreamResponse(
  env: Env,
  uid: string,
  message: string,
  history: ChatMessage[],
  preEmbed: Promise<number[]> | null
): Response {
  const encoder = new TextEncoder();
  const abort = new AbortController();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const emit = (obj: unknown) => controller.enqueue(encoder.encode(frame(obj)));
      try {
        const messages = await buildRagMessages(env, uid, message, history, preEmbed, emit);
        emit({ type: 'status', step: STATUS_THINKING });
        await emitAssistantReply(env, messages, emit, abort.signal);
      } catch (err) {
        console.error('AI chat error:', err);
        if (!abort.signal.aborted) {
          emit({ type: 'error', message: 'Failed to process chat message.' });
        }
      } finally {
        controller.close();
      }
    },
    cancel() {
      // Client disconnected mid-stream: stop paying for the upstream LLM call.
      abort.abort();
    },
  });

  return new Response(stream, { headers: SSE_HEADERS });
}

// Overall wall-clock budget for a full reply. Per-attempt timeouts bound each
// call, but the battery + retries could still sum to minutes, so this caps the
// whole thing and guarantees the UI always gets a terminal frame.
const CHAT_DEADLINE_MS = 90_000;

// Stream NVIDIA's chat while translating frames into the app's delta/done
// protocol, emitting each frame via `emit`. Walks a model battery (NVIDIA gates
// models per request-source, so a single pinned model is a single point of
// failure), retries each once, and falls back to a bounded one-shot call that is
// re-chunked so the UI keeps the streaming feel. Every path emits exactly one
// terminal frame — 'done' or 'error' — so the chat can never hang on
// "AI is thinking...".
async function emitAssistantReply(
  env: Env,
  messages: ChatMessage[],
  emit: (obj: unknown) => void,
  signal: AbortSignal
): Promise<void> {
  const startedAt = Date.now();
  const remainingMs = () => CHAT_DEADLINE_MS - (Date.now() - startedAt);
  const models = chatModels(env);
  let reply = '';
  let lastError = '';

  for (const model of models) {
    if (signal.aborted) return;

    for (let attempt = 0; attempt < 2; attempt++) {
      if (signal.aborted) return;

      const budget = Math.min(CHAT_ATTEMPT_TIMEOUT_MS, remainingMs());
      if (budget < 3000) break;

      let timedOut = false;
      try {
        const upstream = await streamChat(env.NVIDIA_API_KEY, model, messages, signal, budget);
        if (!upstream.ok || !upstream.body) {
          lastError = `${model} HTTP ${upstream.status}`;
          try {
            await upstream.body?.cancel();
          } catch {
            /* ignore */
          }
          continue;
        }

        const reader = (upstream.body as ReadableStream<Uint8Array>).getReader();
        const decoder = new TextDecoder();
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
              emit({ type: 'delta', text: delta });
            }
          }
        }
      } catch (err) {
        const detail = `${(err as Error)?.name || 'Error'}: ${(err as Error)?.message || String(err)}`;
        // A timeout/abort means this model accepted the connection but never
        // streamed (the gpt-oss/glm failure mode). Retrying the SAME model just
        // burns another full timeout and can eat the whole deadline before a
        // healthy model is ever tried, so skip straight to the next model.
        timedOut = /abort|timeout/i.test(detail);
        lastError = `${model}: ${detail}`;
      }

      if (signal.aborted) return;
      if (reply.trim()) {
        // Content already reached the user (even if the socket died mid-stream):
        // commit it rather than restarting and duplicating text.
        emit({ type: 'done', reply, action: null });
        return;
      }
      if (timedOut) break;
    }

    if (remainingMs() < 3000) break;
  }

  // Streaming unavailable across the battery → one bounded one-shot attempt per
  // model, re-chunked so the UI still renders progressively.
  for (const model of models) {
    const budget = Math.min(CHAT_ATTEMPT_TIMEOUT_MS, remainingMs());
    if (budget < 3000) break;
    try {
      const oneShot = await chatOnce(env.NVIDIA_API_KEY, model, messages, budget);
      if (oneShot.trim()) {
        const chunkSize = 60;
        for (let i = 0; i < oneShot.length; i += chunkSize) {
          emit({ type: 'delta', text: oneShot.slice(i, i + chunkSize) });
          await new Promise(r => setTimeout(r, 40));
        }
        emit({ type: 'done', reply: oneShot, action: null });
        return;
      }
      lastError = `${model} produced no content`;
    } catch (err) {
      lastError = `${model}: ${(err as Error)?.message || String(err)}`;
    }
  }

  console.error('All AI providers failed:', lastError);
  if (!signal.aborted) {
    emit({
      type: 'error',
      message: 'The assistant is taking too long right now. Please try again in a moment.',
    });
  }
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
  // Keep the prompt lean: last few messages only, each capped, so the LLM's
  // first token isn't delayed by a huge prefill.
  const history: ChatMessage[] = rawHistory
    .filter(
      (m: any) =>
        m &&
        (m.role === 'user' || m.role === 'assistant') &&
        typeof m.content === 'string'
    )
    .slice(-6)
    .map((m: any) => ({ role: m.role, content: m.content.slice(0, 400) }));

  // Guardrail classification is pure regex — compute it up front so guardrail
  // messages (which never need the LLM) skip the paid embedding entirely.
  const offTopic = isOffTopic(message);
  const complaint = hasMentorshipComplaint(message);
  const mentorship = hasMentorshipIntent(message);
  const needsRag = !offTopic && !complaint && !mentorship;

  // Kick the embedding off IMMEDIATELY — in parallel with Firebase verification
  // and the KV rate-limit/cache reads — instead of after them. The auth
  // round-trip now hides the embed's network RTT. Only RAG-needing messages
  // fire a paid embed. Early-return branches that never consume this promise
  // attach a no-op catch so a failed embed can't surface as an unhandled
  // rejection; the RAG path awaits the ORIGINAL promise to observe real errors.
  const embedPromise = needsRag ? embedCached(env.NVIDIA_EMBED_API_KEY, message) : null;
  if (embedPromise) embedPromise.catch(() => {});

  // Identity comes from the verified Firebase token (never the request body).
  let user;
  try {
    user = await verifyFirebaseToken(env, authHeader);
  } catch {
    return json(401, { success: false, error: 'Invalid or expired session.' });
  }
  const uid = user.uid;

  const cacheKey = await hashCacheKey(uid, message);

  // Fire the two independent KV latencies in parallel: rate-limit + cache.
  const [rl, cached] = await Promise.all([
    checkUserRateLimit(env, uid),
    cacheGet(env, cacheKey),
  ]);

  // Per-user ceiling (20 req/min) enforced in KV by verified uid. Fail-open on
  // KV errors; when exceeded, reject with 429 before any paid NVIDIA call.
  if (!rl.allowed) {
    return json(429, {
      success: false,
      error: 'Rate limit exceeded. Please try again in a moment.',
      retryAfterMs: rl.retryAfterMs,
    });
  }
  if (cached) return stream ? doneSseResponse(cached, null) : json(200, { success: true, reply: cached });

  // Hard guardrail: refuse clearly out-of-scope questions without an LLM call.
  if (offTopic) {
    const reply = OFF_TOPIC_REFUSAL;
    return stream ? doneSseResponse(reply, null) : json(200, { success: true, reply });
  }

  // Paid-but-no-mentor escalation: write the complaint server-side (Mongo) and
  // stream back the reply with the ticket number.
  if (complaint) {
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
  if (mentorship) {
    const fee = env.MENTORSHIP_FEE || '20000';
    const currency = env.MENTORSHIP_CURRENCY || 'NGN';
    const reply = buildMentorshipReply(fee, currency);
    if (stream) return doneSseResponse(reply, { type: 'mentorship' });
    return json(200, { success: true, reply, action: { type: 'mentorship' } });
  }

  // RAG path: streaming mode returns a lazy response that emits live step-by-step
  // status frames (embedding -> searching -> AI is thinking) before the actual
  // token stream begins; the user sees each real stage as it happens. The embed
  // fired above is already in flight, so its RTT is spent by the time the stream
  // starts consuming it.
  if (stream) {
    return ragStreamResponse(env, uid, message, history, embedPromise);
  }

  // Non-stream (JSON) path: run eagerly and return the full reply at once.
  try {
    const messages = await buildRagMessages(env, uid, message, history, embedPromise);
    let reply = '';
    let lastError = '';
    for (const model of chatModels(env)) {
      try {
        reply = await chatOnce(env.NVIDIA_API_KEY, model, messages);
        if (reply.trim()) break;
      } catch (err) {
        lastError = (err as Error)?.message || String(err);
      }
    }
    if (!reply.trim()) throw new Error(lastError || 'No AI model returned a reply.');
    await cacheSet(env, cacheKey, reply);
    return json(200, { success: true, reply });
  } catch (err) {
    console.error('AI chat error:', err);
    return json(500, { success: false, error: 'Failed to process chat message.' });
  }
}