import React, { useEffect, useState } from 'react'
import { motion, animate } from 'framer-motion'
import useReducedMotion from '../hooks/useReducedMotion'

/**
 * Arc gauge for the backend's real confidence_score (0-1). The animated
 * sweep and count-up are purely presentational — the endpoint value is
 * always the authoritative number the API returned, never invented.
 */
export default function ConfidenceGauge({ value, accent = '#C89361', size = 148, label = 'Confidence' }) {
  const reduced = useReducedMotion()
  const pct = Math.max(0, Math.min(100, Math.round((value ?? 0) * 100)))
  const [display, setDisplay] = useState(reduced ? pct : 0)

  const stroke = 6
  const r = size / 2 - stroke
  const cx = size / 2
  const cy = size / 2
  // 270-degree sweep starting at -225deg (bottom-left) for a technical
  // "instrument dial" feel rather than a plain 360 ring.
  const sweepDeg = 270
  const startDeg = -225
  const circumference = 2 * Math.PI * r
  const arcLength = (sweepDeg / 360) * circumference

  useEffect(() => {
    if (reduced) {
      setDisplay(pct)
      return
    }
    const controls = animate(0, pct, {
      duration: 1.3,
      delay: 0.2,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: (v) => setDisplay(Math.round(v)),
    })
    return () => controls.stop()
  }, [pct, reduced])

  const dashOffset = arcLength - (display / 100) * arcLength

  const polarToCartesian = (angleDeg) => {
    const a = ((angleDeg - 90) * Math.PI) / 180
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) }
  }
  const start = polarToCartesian(startDeg)
  const end = polarToCartesian(startDeg + sweepDeg)
  const largeArc = sweepDeg > 180 ? 1 : 0
  const trackPath = `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${end.x} ${end.y}`

  return (
    <div className="relative inline-flex flex-col items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-0">
        <path d={trackPath} fill="none" stroke="rgba(237,234,227,0.1)" strokeWidth={stroke} strokeLinecap="round" />
        <motion.path
          d={trackPath}
          fill="none"
          stroke={accent}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={arcLength}
          style={{ strokeDashoffset: reduced ? arcLength - (pct / 100) * arcLength : dashOffset }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-[28px] leading-none text-bone" aria-hidden="true">
          {display}
          <span className="text-[14px] text-bone-dim">%</span>
        </span>
        <span className="mt-2 text-[9px] tracking-[0.18em] uppercase text-bone-dim">{label}</span>
      </div>
      <span className="sr-only">{`${label}: ${pct}%`}</span>
    </div>
  )
}
