import React from 'react'
import { motion } from 'framer-motion'
import useReducedMotion from '../hooks/useReducedMotion'

/**
 * Shared scroll-reveal primitive. Triggers once when scrolled into view,
 * skips the transform entirely under prefers-reduced-motion (fades only,
 * no translation), and never re-triggers on scroll-back — this is the one
 * mechanism the whole app uses for "content arrives as you scroll to it",
 * rather than every page inventing its own.
 */
export default function Reveal({ children, delay = 0, y = 20, duration = 0.7, className = '', once = true, margin = '-60px' }) {
  const reduced = useReducedMotion()

  return (
    <motion.div
      className={className}
      initial={reduced ? { opacity: 0 } : { opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once, margin }}
      transition={{ duration: reduced ? 0.3 : duration, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  )
}
