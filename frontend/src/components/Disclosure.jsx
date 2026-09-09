import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import useReducedMotion from '../hooks/useReducedMotion'

/**
 * Shared "progressive disclosure" content wrapper — purely presentational,
 * no trigger UI of its own. Every call site renders its own trigger button
 * (wiring its own aria-expanded/aria-controls) and passes `open` down, so
 * this stays reusable across very differently-laid-out trigger clusters
 * (a single header-style toggle in ReportDetail, a multi-button
 * "Examination Controls" row in UploadFlow) without forcing one visual
 * shape on both.
 *
 * Height-animates on open/close (the same technique already used for the
 * Login/SignUp error banner), which is measured rather than purely
 * compositor-friendly — an accepted, already-used cost here because it's
 * one-shot and user-triggered, not continuous. Reduced-motion drops the
 * height animation and just fades.
 */
export default function Disclosure({ open, children, className = '', id }) {
  const reduced = useReducedMotion()

  return (
    <AnimatePresence initial={false}>
      {open && (
        <motion.div
          id={id}
          initial={reduced ? { opacity: 0 } : { opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={reduced ? { opacity: 0 } : { opacity: 0, height: 0 }}
          transition={{ duration: reduced ? 0.15 : 0.35, ease: [0.22, 1, 0.36, 1] }}
          className={`overflow-hidden ${className}`}
        >
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
