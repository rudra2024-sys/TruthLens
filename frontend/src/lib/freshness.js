/**
 * Whether a History/Reports entry was uploaded recently enough to flag as
 * "just completed" — purely derived from the existing `uploaded_at`
 * timestamp the backend already returns, no new backend behavior needed.
 */
const FRESH_WINDOW_MS = 5 * 60 * 1000

export function isFresh(item) {
  const t = new Date(item.uploaded_at || 0).getTime()
  return Number.isFinite(t) && Date.now() - t < FRESH_WINDOW_MS
}
