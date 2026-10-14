import React from 'react'
import { AlertCircle } from 'lucide-react'

/**
 * Shown when the backend withheld a video verdict because the footage is too small to judge (CLAUDE.md
 * section 26): Video Model v1's accuracy was measured to collapse below 360 px on the short side, so such
 * videos are always UNCERTAIN. Explains the UNCERTAIN rather than leaving it unexplained.
 */
export default function LowResolutionNote({ analysis, className = '' }) {
  if (!analysis?.low_resolution) return null
  return (
    <div role="note" className={`flex gap-3 border border-verdictCaution/60 bg-verdictCaution/10 rounded-[4px] p-4 ${className}`}>
      <AlertCircle size={18} strokeWidth={1.75} className="text-verdictCaution shrink-0 mt-[2px]" aria-hidden="true" />
      <div>
        <p className="text-[13px] font-medium text-bone mb-1">Verdict withheld: resolution too low</p>
        <p className="text-[13px] text-bone-dim">
          This video is only {analysis.short_side_px} px on its short side. The video model is unreliable on
          footage this small, so the result is reported as uncertain whatever the score. Try a higher-resolution
          copy of the video if you have one.
        </p>
      </div>
    </div>
  )
}
