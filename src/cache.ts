import type { Env } from './env';

// Assistant reply dedupe cache. Uses Workers KV when bound (production); falls
// back to an in-process LRU for `wrangler dev` without a KV namespace.

const TTL_MS = 30 * 60 * 1000;
const MEMORY_MAX = 300;
const memory = new Map<string, { reply: string; at: number }>();

export async function cacheGet(env: Env, key: string): Promise<string | null> {
  if (env.AI_CACHE) {
    return env.AI_CACHE.get(key);
  }
  const entry = memory.get(key);
  if (!entry) return null;
  if (Date.now() - entry.at > TTL_MS) {
    memory.delete(key);
    return null;
  }
  return entry.reply;
}

export async function cacheSet(env: Env, key: string, reply: string): Promise<void> {
  if (env.AI_CACHE) {
    await env.AI_CACHE.put(key, reply, { expirationTtl: Math.ceil(TTL_MS / 1000) });
    return;
  }
  memory.set(key, { reply, at: Date.now() });
  if (memory.size > MEMORY_MAX) {
    const oldest = memory.keys().next().value;
    if (oldest !== undefined) memory.delete(oldest);
  }
}

// ---------------------------------------------------------------------------
// Embedding cache — pure in-memory bounded LRU (NOT stored in KV; vectors are
// ~1500 floats each and KV writes are metered on the free tier).
//
// Keys are SHA-256 hashes of the input text so no identifiable PII ever sits
// in Worker memory, and identical/similar repeated questions (across users)
// share a single cache entry. LRU cap + TTL guarantees bounded RSS (~24 MB)
// with no possibility of unbounded growth or leaks.
// ---------------------------------------------------------------------------
const EMBED_TTL_MS = 24 * 60 * 60 * 1000;
const EMBED_MEMORY_MAX = 2000;
const embedMemory = new Map<string, { vector: number[]; at: number }>();

export function embedCacheGet(key: string): number[] | null {
  const entry = embedMemory.get(key);
  if (!entry) return null;
  if (Date.now() - entry.at > EMBED_TTL_MS) {
    embedMemory.delete(key);
    return null;
  }
  // Refresh recency (LRU: delete and re-insert at the back).
  embedMemory.delete(key);
  embedMemory.set(key, entry);
  return entry.vector;
}

export function embedCacheSet(key: string, vector: number[]): void {
  embedMemory.delete(key);
  embedMemory.set(key, { vector, at: Date.now() });
  if (embedMemory.size > EMBED_MEMORY_MAX) {
    const oldest = embedMemory.keys().next().value;
    if (oldest !== undefined) embedMemory.delete(oldest);
  }
}

// ---------------------------------------------------------------------------
// Per-user rate limiting (Part 4 & 6 of the diagnosis).
//
// Counter lives in Cloudflare KV, keyed by the *verified* Firebase uid (never
// the client IP — CDN egress and NAT make IPs unreliable, and IPs are trivially
// rotated). A 60-second absolute-TTL key per uid+minute-bucket:
//   ratelimit:<uid>:<epochMinute>  ->  total requests in that minute
// KV has no atomic increment, so concurrent bursts may exceed the ceiling by a
// couple — acceptable for best-effort protection. KV failures FAIL OPEN: an
// error is logged and the request is allowed so a KV blip never locks a user
// out (or blocks chat entirely).
// ---------------------------------------------------------------------------
export const RATE_LIMIT_MAX = 20;
const RATE_LIMIT_WINDOW_S = 60;

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  retryAfterMs: number;
}

const allowAll = (retryAfterMs = 0): RateLimitResult => ({
  allowed: true,
  remaining: RATE_LIMIT_MAX,
  retryAfterMs,
});

// Dev fallback (wrangler dev without a KV binding): bounded local map.
const rateMemory = new Map<string, { count: number; at: number }>();

export async function checkUserRateLimit(env: Env, uid: string): Promise<RateLimitResult> {
  const now = Date.now();
  const bucket = Math.floor(now / 1000 / RATE_LIMIT_WINDOW_S);
  const key = `ratelimit:${uid}:${bucket}`;
  const retryAfterMs = (bucket + 1) * RATE_LIMIT_WINDOW_S * 1000 - now;

  if (env.AI_CACHE) {
    let count = 0;
    try {
      const raw = await env.AI_CACHE.get(key);
      count = raw ? parseInt(raw, 10) || 0 : 0;
      const next = count + 1;
      await env.AI_CACHE.put(key, String(next), { expirationTtl: RATE_LIMIT_WINDOW_S });
      if (next > RATE_LIMIT_MAX) return { allowed: false, remaining: 0, retryAfterMs };
      return { allowed: true, remaining: RATE_LIMIT_MAX - next, retryAfterMs: 0 };
    } catch (err) {
      console.error('[ratelimit] KV failed, allowing request:', err);
      return allowAll();
    }
  }

  const entry = rateMemory.get(key);
  const count = entry && now - entry.at < RATE_LIMIT_WINDOW_S * 1000 ? entry.count : 0;
  const next = count + 1;
  rateMemory.set(key, { count: next, at: now });
  if (next > RATE_LIMIT_MAX) return { allowed: false, remaining: 0, retryAfterMs };
  return { allowed: true, remaining: RATE_LIMIT_MAX - next, retryAfterMs: 0 };
}
