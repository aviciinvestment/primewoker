import type { Env } from './env';
import { handleChat } from './chat';

const json = (status: number, data: unknown, extraHeaders: Record<string, string> = {}): Response =>
  new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
  });

// CORS restored to the original open behavior: echo back whatever Origin the
// browser sends (no allowlist, no env variable to manage). The chat endpoint is
// still gated by Firebase auth + per-user rate limiting, so permissiveness here
// grants no data access — it only lets the browser read the response.
function corsHeaders(request: Request): Record<string, string> {
  return {
    'Access-Control-Allow-Origin': request.headers.get('Origin') || '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400',
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const cors = corsHeaders(request);

    // Preflight for the browser client.
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: cors });
    }

    if (request.method !== 'POST' || url.pathname !== '/api/ai/chat') {
      return json(404, { success: false, error: 'Not found.' }, cors);
    }

    // Standard Cloudflare header carrying the client's true IP (set on ingress,
    // cannot be spoofed by the caller). Forwarded to the backend per request.
    const res = await handleChat(
      request,
      env,
      request.headers.get('Authorization'),
      request.headers.get('CF-Connecting-IP') || null
    );

    // Always re-derive headers with the matched origin so the browser sees
    // the correct single-origin header on every response path.
    const resCors = corsHeaders(request);
    const headers = new Headers(res.headers);
    for (const [k, v] of Object.entries(resCors)) headers.set(k, v);
    return new Response(res.body, { status: res.status, headers });
  },

  // Keep the free-tier Render backend warm between visits so scripted escalations
  // (mentorship complaints) never wait on a 30-90s cold start. Wired via
  // [triggers] in wrangler.toml. The ping itself is ZERO-QUERY: /healthz only
  // reads a mongoose readyState property, so waking Render never piles cold-start
  // overhead onto MongoDB. A heartbeat timestamp is then written to KV (best
  // effort) to keep the Worker's KV binding warm and give ops a last-seen marker.
  async scheduled(_controller: ScheduledController, env: Env): Promise<void> {
    const base = env.BACKEND_ORIGIN;
    if (!base) return;
    try {
      await fetch(`${base.replace(/\/+$/, '')}/healthz`, { method: 'GET' });
    } catch {
      /* a failed keep-warm ping is harmless */
    }
    try {
      await env.AI_CACHE?.put('last-health-heartbeat', new Date().toISOString());
    } catch {
      /* a KV heartbeat write is best-effort */
    }
  },
};