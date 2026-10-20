/**
 * Token-bucket pacer for the batch scanner. The production proxy lets one client make 20 upload/detect requests a
 * minute (burst 5, see nginx.prod.conf), so a big batch has to take its requests one at a time, not all at once:
 * `acquire()` resolves immediately while tokens are left (the burst) and then once per `refillMs`.
 * refillMs <= 0 turns pacing off (local dev has no proxy limit).
 */
export function createPacer({ capacity = 5, refillMs = 3400, now = () => Date.now(),
                              sleep = (ms) => new Promise((r) => setTimeout(r, ms)) } = {}) {
  let tokens = capacity
  let last = now()
  return async function acquire() {
    if (refillMs <= 0) return
    for (;;) {
      const t = now()
      const gained = Math.floor((t - last) / refillMs)
      if (gained > 0) {
        tokens = Math.min(capacity, tokens + gained)
        last += gained * refillMs
      }
      if (tokens >= capacity) last = t           // a full bucket does not bank time
      if (tokens > 0) { tokens -= 1; return }
      await sleep(Math.max(1, last + refillMs - t))
    }
  }
}

/** Runs fn(), and when the server answers 429 (too many requests) waits and tries again - a few times, with a
 * growing pause - instead of failing the file. Any other error is thrown straight away. */
export async function withRetryOn429(fn, { tries = 6, baseMs = 6000, sleep = (ms) => new Promise((r) => setTimeout(r, ms)) } = {}) {
  for (let attempt = 1; ; attempt++) {
    try {
      return await fn()
    } catch (err) {
      if (err?.status !== 429 || attempt >= tries) throw err
      await sleep(baseMs * attempt)
    }
  }
}
