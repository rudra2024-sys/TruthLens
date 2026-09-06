import React from 'react'

/**
 * Shared loading indicator — previously copy-pasted (with minor drift) across
 * ProtectedRoute, History, Reports, and ReportDetail. `layout` controls how
 * much of the viewport it claims:
 *   'screen'        — full viewport, no nav offset (nav isn't rendered yet,
 *                      e.g. ProtectedRoute's pre-auth-check state)
 *   'screen-offset' — full viewport below the fixed 72px nav (a whole page's
 *                      only content, e.g. ReportDetail while it fetches)
 *   'inline'        — sits within an already-rendered page section
 *                      (e.g. History/Reports' list area)
 */
export default function LoadingState({ label = 'Loading…', layout = 'inline', size = 'md' }) {
  const spinnerSize = size === 'sm' ? 'w-8 h-8' : 'w-10 h-10'
  const wrapperClass =
    layout === 'screen'
      ? 'min-h-screen bg-ground flex flex-col items-center justify-center gap-4'
      : layout === 'screen-offset'
        ? 'min-h-screen bg-ground pt-[72px] flex flex-col items-center justify-center gap-4'
        : 'flex flex-col items-center justify-center gap-4 py-24'

  return (
    <div className={wrapperClass} role="status" aria-live="polite">
      <div className={`${spinnerSize} rounded-full border-2 border-line-strong border-t-brass animate-spin`} />
      <p className="tl-hud-label">{label}</p>
    </div>
  )
}
