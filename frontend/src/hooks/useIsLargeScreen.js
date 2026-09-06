import { useState, useEffect } from 'react'

/**
 * True at the Tailwind `lg` breakpoint (1024px) and up.
 *
 * Pinned/scroll-scrubbed sections need their content to fit within one
 * screen height, since the whole point is that it stays put while scroll
 * drives it. On mobile, Home's hero and pipeline content stack into a
 * single tall column that's routinely taller than the viewport — forcing
 * it into a pinned box just clips/overlaps content (confirmed: it pushed
 * the hero heading up behind the nav on a 375x812 viewport). Below `lg`,
 * these sections fall back to a normal flowing layout with simple
 * scroll-triggered entrance animation instead of pinning.
 */
export default function useIsLargeScreen() {
  const [isLarge, setIsLarge] = useState(
    typeof window !== 'undefined' ? window.innerWidth >= 1024 : true
  )

  useEffect(() => {
    const mql = window.matchMedia('(min-width: 1024px)')
    const handler = (e) => setIsLarge(e.matches)
    mql.addEventListener('change', handler)
    setIsLarge(mql.matches)
    return () => mql.removeEventListener('change', handler)
  }, [])

  return isLarge
}
