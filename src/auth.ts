import type { Env } from './env';

export interface VerifiedUser {
  uid: string;
  email: string | null;
  emailVerified: boolean;
}

// Firebase publishes its signing certs here (no private credential needed).
const CERTS_URL =
  'https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com';

let certCache: { keys: Record<string, string>; exp: number } | null = null;

async function getPublicKeys(env: Env): Promise<Record<string, string>> {
  const now = Date.now();
  if (certCache && certCache.exp > now + 60_000) return certCache.keys;

  const res = await fetch(CERTS_URL);
  if (!res.ok) throw new Error(`Failed to fetch Firebase certs (${res.status}).`);
  const keys = (await res.json()) as Record<string, string>;

  let ttlMs = 60 * 60 * 1000;
  const m = /max-age=(\d+)/.exec(res.headers.get('cache-control') || '');
  if (m) ttlMs = parseInt(m[1], 10) * 1000;
  certCache = { keys, exp: now + ttlMs };
  return keys;
}

function b64urlDecode(s: string): string {
  const pad = s.length % 4 === 0 ? '' : '='.repeat(4 - (s.length % 4));
  return atob(s.replace(/-/g, '+').replace(/_/g, '/') + pad);
}

async function importPublicKey(pem: string): Promise<CryptoKey> {
  const der = b64urlDecode(
    pem
      .replace(/-----BEGIN CERTIFICATE-----/g, '')
      .replace(/-----END CERTIFICATE-----/g, '')
      .replace(/\s+/g, '')
  );
  const buf = new Uint8Array(der.length);
  for (let i = 0; i < der.length; i++) buf[i] = der.charCodeAt(i);
  return crypto.subtle.importKey(
    'spki',
    buf.buffer as ArrayBuffer,
    { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' },
    false,
    ['verify']
  );
}

/**
 * Verify a Firebase ID token exactly like `firebase-admin.verifyIdToken`:
 * RS256 signature, audience = project id, issuer = securetoken.google.com,
 * and expiry/issued-at checks with a small leeway.
 */
export async function verifyFirebaseToken(env: Env, authHeader: string | null): Promise<VerifiedUser> {
  if (!authHeader || !/^Bearer\s+.+/i.test(authHeader)) {
    throw new Error('UNAUTHORIZED');
  }
  const token = authHeader.replace(/^Bearer\s+/i, '').trim();
  const projectId = env.FIREBASE_PROJECT_ID || 'primeopportunity-18381';

  const parts = token.split('.');
  if (parts.length !== 3) throw new Error('INVALID_TOKEN');

  let header: any;
  let payload: any;
  try {
    header = JSON.parse(b64urlDecode(parts[0]));
    payload = JSON.parse(b64urlDecode(parts[1]));
  } catch {
    throw new Error('INVALID_TOKEN');
  }

  if (header.alg !== 'RS256') throw new Error('INVALID_TOKEN');
  if (payload.aud !== projectId) throw new Error('INVALID_TOKEN');
  if (payload.iss !== `https://securetoken.google.com/${projectId}`) throw new Error('INVALID_TOKEN');

  const now = Math.floor(Date.now() / 1000);
  if (payload.exp && payload.exp < now - 300) throw new Error('EXPIRED');
  if (payload.iat && payload.iat > now + 300) throw new Error('INVALID_TOKEN');

  const keys = await getPublicKeys(env);
  const pem = keys[header.kid];
  if (!pem) throw new Error('INVALID_TOKEN');

  const publicKey = await importPublicKey(pem);
  const sigBytes = Uint8Array.from(b64urlDecode(parts[2]), c => c.charCodeAt(0));
  const data = new TextEncoder().encode(`${parts[0]}.${parts[1]}`);

  const ok = await crypto.subtle.verify('RSASSA-PKCS1-v1_5', publicKey, sigBytes, data);
  if (!ok) throw new Error('INVALID_TOKEN');

  return {
    uid: payload.sub as string,
    email: (payload.email as string) || null,
    emailVerified: !!payload.email_verified,
  };
}