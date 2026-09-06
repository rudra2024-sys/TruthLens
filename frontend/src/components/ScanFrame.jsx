import React from 'react'

/**
 * Inspection-boundary corner brackets — the recurring visual signature that
 * marks "this region is under examination" (upload zone, evidence cards,
 * report figures). Purely decorative, purely CSS, no motion cost.
 */
export default function ScanFrame({ color = 'rgba(200,147,97,0.3)', armSize = 16, gap = 24 }) {
  const base = 'absolute w-4 h-4'
  const style = { borderColor: color, width: armSize, height: armSize }
  return (
    <div className="pointer-events-none absolute" style={{ inset: gap }} aria-hidden="true">
      <span className={`${base} top-0 left-0 border-l border-t`} style={style} />
      <span className={`${base} top-0 right-0 border-r border-t`} style={style} />
      <span className={`${base} bottom-0 left-0 border-l border-b`} style={style} />
      <span className={`${base} bottom-0 right-0 border-r border-b`} style={style} />
    </div>
  )
}
