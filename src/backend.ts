import type { Env } from './env';

// Calls back into the Express server (Render) for the DB-backed pieces of the
// assistant pipeline. The Worker forwards the user's Firebase Authorization
// header so the server's requireAuth can verify it (Mongo is not reachable
// from the Worker).

async function backendFetch(env: Env, path: string, auth: string | null, body: unknown): Promise<any> {
  const base = env.BACKEND_ORIGIN;
  if (!base) throw new Error('BACKEND_ORIGIN is not set in the Worker.');
  const res = await fetch(`${base.replace(/\/$/, '')}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(auth ? { Authorization: auth } : {}),
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`Backend ${path} failed (${res.status}).`);
  return res.json();
}

// Given the Pinecone-matched opportunity ids, get the serialized RAG context
// block (titles, deadlines, links, ...) built server-side from Mongo.
export async function fetchOpportunityContext(
  env: Env,
  auth: string | null,
  ids: string[]
): Promise<string | null> {
  const data = await backendFetch(env, '/api/ai/opportunity-context', auth, { ids });
  return typeof data?.context === 'string' ? data.context : null;
}

// Write a mentorship complaint ticket (with the user's paid records) and get
// the exact assistant reply to stream back.
export async function recordComplaint(
  env: Env,
  auth: string | null,
  message: string
): Promise<string> {
  const data = await backendFetch(env, '/api/ai/mentorship-complaint', auth, { message });
  if (typeof data?.reply !== 'string' || !data.reply) {
    throw new Error('Backend returned no complaint reply.');
  }
  return data.reply;
}