import React from 'react'
import { motion } from 'framer-motion'
import useReducedMotion from '../hooks/useReducedMotion'

/**
 * Honest "actively examining" indicator for the analysis stage. It never
 * implies a percentage or step count the backend doesn't actually report —
 * it communicates "in progress" only, for exactly as long as the real
 * request is in flight. Reduced-motion gets a static pulsing marker instead
 * of the sweeping line.
 */
export default function ScanSweep() {
  const reduced = useReducedMotion()

  if (reduced) {
    return (
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <span className="w-2 h-2 rounded-full bg-[#A67B5B] animate-pulse" />
      </div>
    )
  }

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none rounded-[inherit]">
      <motion.div
        className="absolute left-0 right-0 h-px"
        style={{
          background: 'linear-gradient(90deg, transparent, #A67B5B 50%, transparent)',
          boxShadow: '0 0 14px 1px rgba(166,123,91,0.45)',
        }}
        initial={{ top: '0%' }}
        animate={{ top: ['0%', '100%', '0%'] }}
        transition={{ duration: 2.6, repeat: Infinity, ease: 'easeInOut' }}
      />
    </div>
  )
}
