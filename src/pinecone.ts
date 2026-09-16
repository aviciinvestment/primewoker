import type { Env } from './env';

export interface PineconeMatch {
  id: string;
  score?: number;
  metadata?: Record<string, unknown> | null;
}

// Data-plane query against a Pinecone serverless index over REST.
export async function queryIndex(
  env: Env,
  vector: number[],
  topK: number,
  namespace?: string
): Promise<PineconeMatch[]> {
  const host = env.PINECONE_HOST;
  if (!host) throw new Error('PINECONE_HOST is not set.');

  const res = await fetch(`https://${host}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Api-Key': env.PINECONE_API_KEY,
    },
    body: JSON.stringify({
      namespace,
      topK,
      vector,
      includeMetadata: true,
      includeValues: false,
    }),
  });
  if (!res.ok) throw new Error(`Pinecone query failed (${res.status}).`);
  const data = (await res.json()) as { matches?: PineconeMatch[] };
  return data.matches || [];
}