/**
 * Idempotency manager and resilient fetch client with retry + exponential backoff.
 * Prevents double-clicks or transient network retries from executing duplicate mutations.
 */

export function generateUUID(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// In-flight intent set to prevent UI race conditions
const pendingIntents = new Set<string>();

export interface IdempotentRequestOptions extends RequestInit {
  idempotencyKey?: string;
  maxRetries?: number;
  initialDelayMs?: number;
}

export async function idempotentFetch<T = any>(
  url: string,
  options: IdempotentRequestOptions = {}
): Promise<T> {
  const {
    idempotencyKey = generateUUID(),
    maxRetries = 2,
    initialDelayMs = 400,
    headers = {},
    ...rest
  } = options;

  // Prevent concurrent execution of identical intent
  if (pendingIntents.has(idempotencyKey)) {
    throw new Error(`Duplicate request in progress for key: ${idempotencyKey}`);
  }
  pendingIntents.add(idempotencyKey);

  const requestHeaders = new Headers(headers);
  requestHeaders.set('Idempotency-Key', idempotencyKey);
  requestHeaders.set('Content-Type', 'application/json');

  let attempt = 0;
  let delay = initialDelayMs;

  try {
    while (attempt <= maxRetries) {
      try {
        const resp = await fetch(url, {
          ...rest,
          headers: requestHeaders,
        });

        if (!resp.ok && resp.status >= 500 && attempt < maxRetries) {
          throw new Error(`Server error HTTP ${resp.status}`);
        }

        const data = await resp.json().catch(() => ({}));
        return data as T;
      } catch (err) {
        attempt++;
        if (attempt > maxRetries) throw err;
        await new Promise((r) => setTimeout(r, delay));
        delay *= 2; // exponential backoff
      }
    }
    throw new Error('Max retries exceeded');
  } finally {
    pendingIntents.delete(idempotencyKey);
  }
}
