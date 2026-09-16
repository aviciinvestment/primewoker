import type { Env } from './env';
import { handleChat } from './chat';

const json = (status: number, data: unknown, extraHeaders: Record<string, string> = {}): Response =>
  new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', ...extraHeaders },
  });

function corsHeaders(env: Env): Record<string, string> {
  const allowed = env.ALLOWED_ORIGINS || '*';
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
    const cors = corsHeaders(env);

    // Preflight for the browser client.
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: cors });
    }

    if (request.method !== 'POST' || url.pathname !== '/api/ai/chat') {
      return json(404, { success: false, error: 'Not found.' }, cors);
    }

    const res = await handleChat(request, env, request.headers.get('Authorization'));

    const headers = new Headers(res.headers);
    for (const [k, v] of Object.entries(cors)) headers.set(k, v);
    return new Response(res.body, { status: res.status, headers });
  },
};