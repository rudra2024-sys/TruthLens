import React from 'react'
import { motion } from 'framer-motion'
import useReducedMotion from '../hooks/useReducedMotion'
import { getVerdictInfo } from '../lib/verdict'

/**
 * The "moment of reveal" verdict presentation — used once per result (Verify
 * completion, Report detail), not in lists. Each verdict has a distinct
 * motion signature so the state is legible without relying on color alone:
 *
 *   REAL        — a single, calm settle. Nothing further moves. Resolved.
 *   MANIPULATED — a sharper arrival with one brief alert ring flash.
 *   UNCERTAIN   — a brief left-right waver before settling. Ambiguous, not broken.
 *
 * All three converge on the same rest state; reduced-motion users land there
 * immediately with no transform, only a plain fade.
 */
export default function VerdictSeal({ verdict, size = 96 }) {
  const info = getVerdictInfo(verdict)
  const { Icon } = info
  const reduced = useReducedMotion()

  const ringTransition =
    info.key === 'manipulated'
      ? { duration: 0.9, ease: [0.16, 1, 0.3, 1], delay: 0.15 }
      : info.key === 'uncertain'
        ? { duration: 1.1, ease: [0.65, 0, 0.35, 1], delay: 0.15 }
        : { duration: 1.2, ease: [0.22, 1, 0.36, 1], delay: 0.1 }

  const sealVariants = reduced
    ? { hidden: { opacity: 0 }, shown: { opacity: 1 } }
    : {
        hidden: { opacity: 0, scale: 0.85, rotate: 0, x: 0 },
        shown:
          info.key === 'uncertain'
            ? {
                opacity: 1,
                scale: 1,
                x: [0, -6, 5, -3, 0],
                transition: { x: { duration: 0.7, ease: 'easeInOut', delay: 0.25 }, default: { duration: 0.4 } },
              }
            : info.key === 'manipulated'
              ? { opacity: 1, scale: [0.85, 1.06, 1], transition: { duration: 0.55, ease: [0.34, 1.56, 0.64, 1] } }
              : { opacity: 1, scale: 1, transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] } },
      }

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      {!reduced && (
        <motion.span
          className="absolute inset-0 rounded-full border"
          style={{ borderColor: info.accent }}
          initial={{ opacity: 0.6, scale: 0.7 }}
          animate={{ opacity: 0, scale: 1.5 }}
          transition={ringTransition}
        />
      )}
      <motion.div
        className={`relative flex items-center justify-center rounded-full border ${info.bg} ${info.border}`}
        style={{ width: size, height: size }}
        variants={sealVariants}
        initial="hidden"
        animate="shown"
      >
        <Icon size={size * 0.4} strokeWidth={1.5} color={info.accent} />
      </motion.div>
    </div>
  )
}
