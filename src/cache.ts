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