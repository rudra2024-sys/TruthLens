import { useState, useEffect } from 'react'

/**
 * Respects the visitor's OS/browser "reduce motion" preference. Every
 * animated component in this app imports `useReducedMotion` from here
 * instead of directly from framer-motion, so this is the single place that
 * decision lives.
 *
 * This does NOT change TruthLens' default motion design: visitors who have
 * not requested reduced motion get the full LensNarrative/VerdictSeal/Reveal
 * experience exactly as before. Only visitors who have explicitly turned on
 * "reduce motion" at the OS/browser level get the reduced-motion fallback
 * paths those components already had built (see e.g. LensNarrative's
 * StaticLensExperience) — this hook is what makes that reachable again.
 */
export default function useReducedMotion() {
  const [reduced, setReduced] = useState(
    typeof window !== 'undefined'
      ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
      : false
  )

  useEffect(() => {
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)')
    const handler = (e) => setReduced(e.matches)
    mql.addEventListener('change', handler)
    setReduced(mql.matches)
    return () => mql.removeEventListener('change', handler)
  }, [])

  return reduced
}
