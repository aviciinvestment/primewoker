import type { Env } from './env';

// Calls back into the Express server (Render) for the DB-backed pieces of the
// assistant pipeline. The Worker forwards the user's Firebase Authorization
// header so the server's requireAuth can verify it (Mongo is not reachable
// from the Worker).

// Hard ceiling on every backend round-trip. Render free tier sleeps after idle;
// without this a cold start (20-90s) would leave the chat stuck "thinking" with
// the upstream fetch hanging. Callers fall back to a generic answer on timeout.
const BACKEND_TIMEOUT_MS = 7000;

async function backendFetch(
  env: Env,
  path: string,
  auth: string | null,
  body: unknown,
  clientIp: string | null,
  timeoutMs = BACKEND_TIMEOUT_MS
): Promise<any> {
  const base = env.BACKEND_ORIGIN;
  if (!base) throw new Error('BACKEND_ORIGIN is not set in the Worker.');

  // Forward the client's true IP using standard Cloudflare headers so the
  // backend's trust-proxy + rate limiting sees the real address (cf-connecting-ip
  // is set by Cloudflare itself on ingress and cannot be spoofed by the caller).
  const ipHeaders: Record<string, string> = clientIp
    ? { 'CF-Connecting-IP': clientIp, 'X-Forwarded-For': clientIp, 'X-Real-IP': clientIp }
    : {};

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${base.replace(/\/+$/, '')}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(auth ? { Authorization: auth } : {}),
        ...ipHeaders,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    if (controller.signal.aborted) throw new Error(`Backend ${path} timed out after ${timeoutMs}ms.`);
    throw err;
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) throw new Error(`Backend ${path} failed (${res.status}).`);
  return res.json();
}

// Write a mentorship complaint ticket (with the user's paid records) and get
// the exact assistant reply to stream back.
export async function recordComplaint(
  env: Env,
  auth: string | null,
  message: string,
  clientIp: string | null = null
): Promise<string> {
  const data = await backendFetch(env, '/api/ai/mentorship-complaint', auth, { message }, clientIp);
  if (typeof data?.reply !== 'string' || !data.reply) {
    throw new Error('Backend returned no complaint reply.');
  }
  return data.reply;
}

// Persist a finished WORKER-produced exchange for the admin Chat Activity feed.
// The reply was already streamed to the user when this runs, so it is pure
// fire-and-forget: the backend identity is re-verified via the forwarded auth
// header, and any failure is logged (never thrown) so chat is never slowed.
export async function logChat(
  env: Env,
  auth: string | null,
  payload: { message: string; reply: string; userName?: string },
  clientIp: string | null = null
): Promise<void> {
  try {
    await backendFetch(env, '/api/ai/chat-log', auth, payload, clientIp);
  } catch (err) {
    console.error('Failed to log chat:', (err as Error)?.message || err);
  }
}