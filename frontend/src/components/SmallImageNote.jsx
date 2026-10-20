import React from 'react'
import { AlertCircle } from 'lucide-react'

// Images under this many pixels on the short side get a caution. A judgement call, not a measured cut-off: in a
// local check every real test photo (178-299 px) was tiny, and 9 of 26 were not called REAL, but there were no large
// real photos to compare against.
export const SMALL_IMAGE_PX = 300

/** Shown next to the verdict for very small images, which carry little detail for the models to work with. */
export default function SmallImageNote({ width, height, className = '' }) {
  if (!width || !height || Math.min(width, height) >= SMALL_IMAGE_PX) return null
  return (
    <div role="note" className={`flex gap-3 border border-verdictCaution/60 bg-verdictCaution/10 rounded-[4px] p-4 ${className}`}>
      <AlertCircle size={18} strokeWidth={1.75} className="text-verdictCaution shrink-0 mt-[2px]" aria-hidden="true" />
      <div>
        <p className="text-[13px] font-medium text-bone mb-1">Small image: result may be less reliable</p>
        <p className="text-[13px] text-bone-dim">
          This image is only {width} x {height} px. Small images carry little detail, and in our own tests genuine photos this
          small were sometimes flagged. Treat the verdict with extra caution, and try a larger copy if you have one.
        </p>
      </div>
    </div>
  )
}
