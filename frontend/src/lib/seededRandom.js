/**
 * Deterministic pseudo-random points from a string seed (filename+size),
 * so scan-visualization markers are stable per file rather than jittering
 * on every re-render, without claiming to be actual detection findings.
 */
export function seededPoints(seed, count) {
  let h = 0
  const str = String(seed)
  for (let i = 0; i < str.length; i++) {
    h = (Math.imul(31, h) + str.charCodeAt(i)) | 0
  }
  let state = h >>> 0
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    return state / 4294967296
  }
  return Array.from({ length: count }, () => ({ x: 8 + next() * 84, y: 8 + next() * 84 }))
}
