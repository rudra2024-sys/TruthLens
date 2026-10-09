import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Upload as UploadIcon, ShieldQuestion } from 'lucide-react'
import {
  deleteIdentityReference,
  enrollIdentityReference,
  getIdentityReference,
  matchIdentity,
  uploadMedia,
} from './api/client'
import ConfidenceGauge from './components/ConfidenceGauge'
import WebcamCapture from './components/WebcamCapture'
import { getIdentityMatchInfo } from './lib/identityVerdict'

const REFERENCE_ACCEPT = 'image/jpeg,image/png,image/webp'

const heading = (
  <div className="text-center mb-12">
    <p className="tl-hud-label !text-brass mb-4">Identity Verification</p>
    <h1 className="font-serif text-display-l text-bone mb-6">
      Confirm it's <span className="italic text-brass">you.</span>
    </h1>
    <p className="text-[16px] text-bone-dim max-w-[460px] mx-auto">
      Enroll a reference photo once, then compare a live camera capture against it at any time.
    </p>
  </div>
)

function EnrollDropzone({ busy, error, onFile }) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef(null)

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault()
      setIsDragging(false)
      const f = e.dataTransfer.files[0]
      if (f) onFile(f)
    },
    [onFile]
  )

  return (
    <div className="flex flex-col items-center gap-5">
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={(e) => { e.preventDefault(); setIsDragging(false) }}
        onDrop={handleDrop}
        onClick={() => !busy && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && !busy) inputRef.current?.click() }}
        aria-label="Upload a reference photo"
        className={`relative bg-panel tl-ticks rounded-[4px] w-full max-w-[420px] aspect-[4/3] flex flex-col items-center justify-center cursor-pointer transition-colors duration-500 ${
          isDragging ? 'border-2 border-dashed border-brass bg-[rgba(200,147,97,0.04)]' : 'border border-dashed border-line-strong'
        } ${busy ? 'cursor-default opacity-60' : ''}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={REFERENCE_ACCEPT}
          className="hidden"
          onChange={(e) => { const f = e.target.files[0]; if (f) onFile(f) }}
        />
        <div className="w-16 h-16 rounded-full bg-[rgba(200,147,97,0.06)] border border-line flex items-center justify-center mb-5">
          <UploadIcon size={28} strokeWidth={1.25} className="text-bone-faint" />
        </div>
        <p className="font-serif text-[20px] text-bone mb-2">
          {busy ? 'Enrolling…' : 'Place your reference photo'}
        </p>
        <p className="text-[13px] text-bone-dim">or click to browse — JPG, PNG, WEBP</p>
      </div>
      {error && <p className="text-[13px] text-verdictDanger max-w-[360px] text-center">{error}</p>}
    </div>
  )
}

function MatchResultCard({ result, onRetry }) {
  const info = getIdentityMatchInfo(result.verdict)
  const Icon = info.Icon
  return (
    <div className="flex flex-col items-center gap-6 text-center">
      <div className="flex flex-col md:flex-row items-center gap-10">
        <div className={`w-20 h-20 rounded-full ${info.bg} border ${info.border} flex items-center justify-center`}>
          <Icon size={30} strokeWidth={1.5} style={{ color: info.text }} />
        </div>
        <ConfidenceGauge value={result.similarity_score} accent={info.accent} size={128} label="Similarity" />
      </div>
      <div>
        <p className={`text-[20px] font-serif mb-1 ${info.textClass}`}>{info.label}</p>
        <p className="text-[13px] text-bone-dim">{info.caption}</p>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="text-[12px] font-medium tracking-[0.06em] text-brass hover:text-bone transition-colors duration-300 link-underline"
      >
        Verify again
      </button>
    </div>
  )
}

export default function IdentityMatch() {
  const [reference, setReference] = useState(undefined) // undefined=loading, null=not enrolled, object=enrolled
  const [enrollBusy, setEnrollBusy] = useState(false)
  const [enrollError, setEnrollError] = useState(null)
  const [showCamera, setShowCamera] = useState(false)
  const [matchBusy, setMatchBusy] = useState(false)
  const [matchError, setMatchError] = useState(null)
  const [matchResult, setMatchResult] = useState(null)

  useEffect(() => {
    let cancelled = false
    getIdentityReference()
      .then((ref) => { if (!cancelled) setReference(ref) })
      .catch(() => { if (!cancelled) setReference(null) })
    return () => { cancelled = true }
  }, [])

  const enroll = async (file) => {
    setEnrollBusy(true)
    setEnrollError(null)
    try {
      const uploadRes = await uploadMedia(file)
      const ref = await enrollIdentityReference(uploadRes.data.upload_id)
      setReference(ref)
      setMatchResult(null)
    } catch (err) {
      setEnrollError(err.message)
    } finally {
      setEnrollBusy(false)
    }
  }

  const handleCapture = async (file) => {
    setShowCamera(false)
    setMatchBusy(true)
    setMatchError(null)
    try {
      const uploadRes = await uploadMedia(file)
      const result = await matchIdentity(uploadRes.data.upload_id)
      setMatchResult(result)
    } catch (err) {
      setMatchError(err.message)
    } finally {
      setMatchBusy(false)
    }
  }

  const handleDeleteReference = async () => {
    try {
      await deleteIdentityReference()
    } finally {
      setReference(null)
      setMatchResult(null)
      setMatchError(null)
    }
  }

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />
      <section className="relative min-h-[calc(100vh-72px)] flex flex-col items-center justify-center py-16 md:py-24 tl-inspection-grid">
        <div className="absolute inset-0 bg-gradient-to-b from-ground via-ground to-panel/40" />

        <div className="relative z-10 max-w-[600px] mx-auto px-6 w-full animate-fade-in-up animate-delay-1">
          {heading}

          <div className="bg-panel border border-line rounded-[6px] p-10 md:p-12 flex flex-col items-center">
            {reference === undefined && (
              <p className="text-[13px] text-bone-dim">Loading…</p>
            )}

            {reference === null && (
              <>
                <div className="flex items-start gap-3 mb-8 max-w-[420px] text-left">
                  <ShieldQuestion size={18} strokeWidth={1.5} className="text-bone-faint shrink-0 mt-0.5" />
                  <p className="text-[12px] text-bone-dim leading-relaxed">
                    No reference photo enrolled yet. Upload a clear, front-facing photo once — it's stored
                    the same way as any other upload.
                  </p>
                </div>
                <EnrollDropzone busy={enrollBusy} error={enrollError} onFile={enroll} />
              </>
            )}

            {reference && !showCamera && !matchBusy && !matchResult && !matchError && (
              <div className="flex flex-col items-center gap-6 text-center">
                <p className="tl-hud-label !text-[9px]">Reference enrolled</p>
                <p className="text-[13px] text-bone-dim max-w-[380px]">
                  This will access your camera to verify your identity. Captured frames are stored the
                  same way as any other upload. Camera access requires a secure connection.
                </p>
                <button
                  type="button"
                  onClick={() => setShowCamera(true)}
                  className="bg-brass text-ground px-6 py-3 rounded-[4px] text-[12px] font-medium tracking-[0.08em] uppercase btn-lift"
                >
                  Verify Identity
                </button>
                <Link
                  to="/identity/monitor"
                  className="text-[12px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
                >
                  Start Monitoring Session
                </Link>
                <button
                  type="button"
                  onClick={handleDeleteReference}
                  className="text-[11px] tracking-[0.06em] text-bone-faint hover:text-bone link-underline"
                >
                  Remove enrolled reference
                </button>
              </div>
            )}

            {reference && showCamera && (
              <WebcamCapture onCapture={handleCapture} onCancel={() => setShowCamera(false)} className="w-full" />
            )}

            {matchBusy && (
              <p className="text-[13px] text-bone-dim">Comparing against your enrolled reference…</p>
            )}

            {matchError && !matchBusy && (
              <div className="flex flex-col items-center gap-4 text-center">
                <p className="text-[14px] text-verdictDanger max-w-[360px]">{matchError}</p>
                <button
                  type="button"
                  onClick={() => { setMatchError(null); setShowCamera(true) }}
                  className="text-[12px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
                >
                  Try again
                </button>
              </div>
            )}

            {matchResult && !matchBusy && (
              <MatchResultCard result={matchResult} onRetry={() => { setMatchResult(null); setShowCamera(true) }} />
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
