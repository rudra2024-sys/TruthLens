import React from 'react'
import { getVerdictInfo } from '../lib/verdict'

/**
 * Compact inline verdict indicator — used in list rows (History, Reports)
 * where dozens may render at once, so it stays a plain styled element with
 * no per-instance animation.
 */
export default function VerdictBadge({ verdict, className = '' }) {
  const info = getVerdictInfo(verdict)
  const { Icon } = info

  return (
    <span
      className={`inline-flex items-center gap-1.5 text-[10px] font-medium tracking-[0.08em] uppercase px-3 py-1 rounded-full border whitespace-nowrap ${info.bg} ${info.border} ${info.textClass} ${className}`}
    >
      <Icon size={11} strokeWidth={2} />
      {info.label}
    </span>
  )
}
