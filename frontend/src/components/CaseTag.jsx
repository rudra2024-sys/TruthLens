import React from 'react'

/**
 * A stable "case identifier" derived from the upload's real UUID — never a
 * fabricated number. Same upload always renders the same tag.
 */
export default function CaseTag({ id, className = '' }) {
  const short = (id || '').replace(/-/g, '').slice(0, 8).toUpperCase()
  return (
    <span className={`tl-figure text-[10px] tracking-[0.12em] text-bone-faint ${className}`}>
      CASE {short || '————————'}
    </span>
  )
}
