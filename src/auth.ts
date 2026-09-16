import type { Env } from './env';

export interface VerifiedUser {
  uid: string;
  email: string | null;
  emailVerified: boolean;
}

// Firebase publishes its signing keys as a JWK set here (no private credential
// needed). Keys are matched to the token's `kid` header.
const JWK_URL =
  'https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com';

// Minimal JWK shape used by WebCrypto importKey('jwk', ...).
interface Jwk {
  kid: string;
  kty: string;
  n: string;
  e: string;
  alg?: string;
  use?: string;
}

let certCache: { keys: Record<string, Jwk>; exp: number } | null = null;

async function getPublicKeys(env: Env): Promise<Record<string, Jwk>> {
  const now = Date.now();
  if (certCache && certCache.exp > now + 60_000) return certCache.keys;

  const res = await fetch(JWK_URL);
  if (!res.ok) throw new Error(`Failed to fetch Firebase public keys (${res.status}).`);
  const data = (await res.json()) as { keys?: Jwk[] };

  const keys: Record<string, Jwk> = {};
  for (const k of data.keys || []) {
    // Only RSA signing keys are valid for RS256 (the only alg we accept).
    if (k && k.kid && k.kty === 'RSA' && k.n && k.e) keys[k.kid] = k;
  }

  let ttlMs = 5 * 60 * 1000;
  const m = /max-age=(\d+)/.exec(res.headers.get('cache-control') || '');
  if (m) ttlMs = parseInt(m[1], 10) * 1000;
  certCache = { keys, exp: now + ttlMs };
  return keys;
}

function b64urlDecode(s: string): string {
  const pad = s.length % 4 === 0 ? '' : '='.repeat(4 - (s.length % 4));
  return atob(s.replace(/-/g, '+').replace(/_/g, '/') + pad);
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
  const jwk = keys[header.kid];
  if (!jwk) throw new Error('INVALID_TOKEN');

  const publicKey = await crypto.subtle.importKey(
    'jwk',
    { kty: 'RSA', alg: 'RS256', use: 'sig', n: jwk.n, e: jwk.e },
    { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' },
    false,
    ['verify']
  );
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