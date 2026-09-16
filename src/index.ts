import type { Env } from './env';
import { handleChat } from './chat';

const json = (status: number, data: unknown, extraHeaders: Record<string, string> = {}): Response =>
  new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
  });

function corsHeaders(env: Env, request: Request): Record<string, string> {
  const origin = request.headers.get('Origin');
  let allowed = '';

  // Split comma-separated allowlist, strip whitespace, reject any origin not
  // explicitly listed. Fail closed — a missing/empty env denies all cross-origin
  // traffic (no wildcard); same-origin requests are unaffected because browsers
  // only enforce CORS when an Origin header is actually sent.
  const allowlist = (env.ALLOWED_ORIGINS || '')
    .split(',')
    .map(o => o.trim())
    .filter(Boolean);

  // Echo back the exact matching origin (no wildcards, no comma-joined list).
  if (origin && allowlist.includes(origin)) {
    allowed = origin;
  }

  return {
    'Access-Control-Allow-Origin': allowed,
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400',
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const cors = corsHeaders(env, request);

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
    const resCors = corsHeaders(env, request);
    const headers = new Headers(res.headers);
    for (const [k, v] of Object.entries(resCors)) headers.set(k, v);
    return new Response(res.body, { status: res.status, headers });
  },
};