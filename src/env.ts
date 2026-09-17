// Runtime environment available to the Worker (vars + secrets + bindings).
export interface Env {
  // --- secrets (wrangler secret put ...) ---
  NVIDIA_API_KEY: string;
  NVIDIA_EMBED_API_KEY: string;
  PINECONE_API_KEY: string;

  // --- plain vars ([vars] in wrangler.toml) ---
  FIREBASE_PROJECT_ID?: string;
  PINECONE_INDEX?: string;
  PINECONE_HOST?: string;
  BACKEND_ORIGIN?: string;
  // Comma-separated NVIDIA chat model battery (optional; overrides the default
  // list in nvidia.ts). Useful because NVIDIA gates models per request-source.
  NVIDIA_CHAT_MODELS?: string;
  MENTORSHIP_FEE?: string;
  MENTORSHIP_CURRENCY?: string;
  ALLOWED_ORIGINS?: string;

  // --- KV binding for the assistant reply cache (optional in dev) ---
  AI_CACHE?: KVNamespace;
}