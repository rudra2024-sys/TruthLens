import React, { useRef, useState } from 'react'
import { Upload as UploadIcon, Camera } from 'lucide-react'
import WebcamCapture from './WebcamCapture'

const ID_PHOTO_ACCEPT = 'image/jpeg,image/png,image/webp'

/**
 * Extracted from IdentityMatch.jsx (2026-10-14) so the same "photograph or upload your ID, with
 * explicit consent" step can be reused by the per-exam identity check in ExamAttempt.jsx without
 * duplicating this UI. Pure capture component -- it has no idea what happens to the resulting
 * file; the caller's onFile decides (enroll once, or verify fresh for this exam attempt).
 */
export default function GovernmentIdCapture({ busy, error, consented, onConsentChange, onFile }) {
  const [mode, setMode] = useState(null) // null=choice, 'upload', 'camera'
  const inputRef = useRef(null)
  const disabled = busy || !consented

  if (mode === 'camera') {
    return (
      <WebcamCapture
        onCapture={onFile}
        onCancel={() => setMode(null)}
        facingMode="environment"
        mirror={false}
        className="w-full"
      />
    )
  }

  return (
    <div className="flex flex-col items-center gap-5 w-full">
      <label className="flex items-start gap-3 max-w-[480px] text-left cursor-pointer select-none">
        <input
          type="checkbox"
          checked={consented}
          onChange={(e) => onConsentChange(e.target.checked)}
          disabled={busy}
          className="mt-1 shrink-0 accent-brass w-[18px] h-[18px]"
        />
        <span className="text-[15px] text-bone-dim leading-relaxed">
          I consent to TruthLens using this ID photo momentarily to verify my identity. The photo
          itself is never stored — only a resulting face-match embedding is kept, exactly like a
          plain reference photo.
        </span>
      </label>
      <div className="flex items-center gap-4 w-full max-w-[480px]">
        <button
          type="button"
          disabled={disabled}
          onClick={() => setMode('camera')}
          className="flex-1 flex items-center justify-center gap-2 bg-brass text-ground px-5 py-3.5 rounded-[3px] text-[14px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-40 disabled:pointer-events-none"
        >
          <Camera size={16} strokeWidth={1.75} /> Photograph ID
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          className="flex-1 flex items-center justify-center gap-2 border border-line-strong text-bone-dim px-5 py-3.5 rounded-[3px] text-[14px] font-medium tracking-[0.08em] uppercase hover:border-brass hover:text-brass transition-colors duration-300 disabled:opacity-40 disabled:pointer-events-none"
        >
          <UploadIcon size={16} strokeWidth={1.75} /> Upload Photo
        </button>
        <input
          ref={inputRef}
          type="file"
          accept={ID_PHOTO_ACCEPT}
          className="hidden"
          disabled={disabled}
          onChange={(e) => { const f = e.target.files[0]; if (f) onFile(f) }}
        />
      </div>
      {!consented && <p className="text-[14px] text-bone-faint">Check the box above to continue</p>}
      {error && <p className="text-[15px] text-verdictDanger max-w-[400px] text-center">{error}</p>}
    </div>
  )
}
