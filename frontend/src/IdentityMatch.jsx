import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Upload as UploadIcon, ShieldQuestion, Building2, Download, CreditCard, Camera } from 'lucide-react'
import {
  deleteIdentityReference,
  downloadSessionReport,
  enrollIdentityReference,
  getIdentityReference,
  getMyOrganization,
  getMySessions,
  joinOrganization,
  matchIdentity,
  uploadMedia,
  verifyGovernmentId,
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

function EnrollDropzone({ busy, error, consented, onConsentChange, onFile }) {
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef(null)
  const disabled = busy || !consented

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault()
      setIsDragging(false)
      if (disabled) return
      const f = e.dataTransfer.files[0]
      if (f) onFile(f)
    },
    [onFile, disabled]
  )

  return (
    <div className="flex flex-col items-center gap-5">
      <label className="flex items-start gap-3 max-w-[420px] text-left cursor-pointer select-none">
        <input
          type="checkbox"
          checked={consented}
          onChange={(e) => onConsentChange(e.target.checked)}
          disabled={busy}
          className="mt-0.5 shrink-0 accent-brass"
        />
        <span className="text-[12px] text-bone-dim leading-relaxed">
          I consent to TruthLens storing a face embedding derived from this photo, to verify my
          identity and, if I start one, during monitoring sessions. It is kept until I delete it
          below or delete my account.
        </span>
      </label>
      <div
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setIsDragging(true) }}
        onDragLeave={(e) => { e.preventDefault(); setIsDragging(false) }}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && !disabled) inputRef.current?.click() }}
        aria-label="Upload a reference photo"
        aria-disabled={disabled}
        className={`relative bg-panel tl-ticks rounded-[4px] w-full max-w-[420px] aspect-[4/3] flex flex-col items-center justify-center transition-colors duration-500 ${
          disabled ? 'cursor-default opacity-60' : 'cursor-pointer'
        } ${
          isDragging ? 'border-2 border-dashed border-brass bg-[rgba(200,147,97,0.04)]' : 'border border-dashed border-line-strong'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={REFERENCE_ACCEPT}
          className="hidden"
          disabled={disabled}
          onChange={(e) => { const f = e.target.files[0]; if (f) onFile(f) }}
        />
        <div className="w-16 h-16 rounded-full bg-[rgba(200,147,97,0.06)] border border-line flex items-center justify-center mb-5">
          <UploadIcon size={28} strokeWidth={1.25} className="text-bone-faint" />
        </div>
        <p className="font-serif text-[20px] text-bone mb-2">
          {busy ? 'Enrolling…' : 'Place your reference photo'}
        </p>
        <p className="text-[13px] text-bone-dim">
          {consented ? 'or click to browse — JPG, PNG, WEBP' : 'Check the box above to continue'}
        </p>
      </div>
      {error && <p className="text-[13px] text-verdictDanger max-w-[360px] text-center">{error}</p>}
    </div>
  )
}

function JoinOrganizationForm({ busy, error, onJoin, onCancel }) {
  const [code, setCode] = useState('')

  const submit = (e) => {
    e.preventDefault()
    if (!code.trim() || busy) return
    onJoin(code.trim())
  }

  return (
    <form onSubmit={submit} className="flex flex-col items-center gap-5 max-w-[380px] w-full">
      <div className="flex items-start gap-3 text-left">
        <Building2 size={18} strokeWidth={1.5} className="text-bone-faint shrink-0 mt-0.5" />
        <p className="text-[12px] text-bone-dim leading-relaxed">
          Enter the private join code your organisation gave you. Identity enrollment opens once
          you're linked — this ties any monitoring sessions you start to them.
        </p>
      </div>
      <input
        type="text"
        value={code}
        onChange={(e) => setCode(e.target.value)}
        disabled={busy}
        placeholder="A3F9-K7M2"
        autoCapitalize="characters"
        className="w-full bg-ground border border-line rounded-[3px] px-4 py-3 text-[13px] tl-figure tracking-[0.1em] text-bone text-center placeholder:text-bone-faint focus:outline-none focus:border-brass transition-colors duration-300"
      />
      {error && <p className="text-[13px] text-verdictDanger text-center">{error}</p>}
      <div className="flex items-center gap-4">
        <button
          type="submit"
          disabled={busy || !code.trim()}
          className="bg-brass text-ground px-6 py-3 rounded-[4px] text-[12px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-50 disabled:pointer-events-none"
        >
          {busy ? 'Linking…' : 'Link Organisation'}
        </button>
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="text-[12px] text-bone-faint hover:text-bone link-underline"
          >
            Cancel
          </button>
        )}
      </div>
    </form>
  )
}

function EnrollmentChoice({ onChooseId, onChooseSelfie }) {
  return (
    <div className="flex flex-col items-center gap-6 max-w-[420px]">
      <p className="text-[13px] text-bone-dim text-center leading-relaxed">
        Choose how to prove it's you. Government-ID verification is required before taking a
        proctored exam; a plain photo is enough for ad-hoc identity checks only.
      </p>
      <button
        type="button"
        onClick={onChooseId}
        className="w-full flex items-center gap-4 text-left bg-[rgba(200,147,97,0.06)] border border-brass/40 rounded-[4px] p-5 hover:border-brass transition-colors duration-300"
      >
        <CreditCard size={22} strokeWidth={1.5} className="text-brass shrink-0" />
        <div>
          <p className="text-[14px] text-bone font-medium">Verify with a government ID</p>
          <p className="text-[12px] text-bone-dim mt-0.5">
            Aadhaar, PAN, or similar — matched live against your camera. Required for exams.
          </p>
        </div>
      </button>
      <button
        type="button"
        onClick={onChooseSelfie}
        className="text-[12px] text-bone-faint hover:text-bone link-underline"
      >
        Use a plain photo instead (no exam access)
      </button>
    </div>
  )
}

function GovernmentIdCapture({ busy, error, consented, onConsentChange, onFile }) {
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
      <label className="flex items-start gap-3 max-w-[420px] text-left cursor-pointer select-none">
        <input
          type="checkbox"
          checked={consented}
          onChange={(e) => onConsentChange(e.target.checked)}
          disabled={busy}
          className="mt-0.5 shrink-0 accent-brass"
        />
        <span className="text-[12px] text-bone-dim leading-relaxed">
          I consent to TruthLens using this ID photo momentarily to verify my identity. The photo
          itself is never stored — only a resulting face-match embedding is kept, exactly like a
          plain reference photo.
        </span>
      </label>
      <div className="flex items-center gap-4 w-full max-w-[420px]">
        <button
          type="button"
          disabled={disabled}
          onClick={() => setMode('camera')}
          className="flex-1 flex items-center justify-center gap-2 bg-brass text-ground px-5 py-3 rounded-[3px] text-[12px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-40 disabled:pointer-events-none"
        >
          <Camera size={14} strokeWidth={1.75} /> Photograph ID
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
          className="flex-1 flex items-center justify-center gap-2 border border-line-strong text-bone-dim px-5 py-3 rounded-[3px] text-[12px] font-medium tracking-[0.08em] uppercase hover:border-brass hover:text-brass transition-colors duration-300 disabled:opacity-40 disabled:pointer-events-none"
        >
          <UploadIcon size={14} strokeWidth={1.75} /> Upload Photo
        </button>
        <input
          ref={inputRef}
          type="file"
          accept={REFERENCE_ACCEPT}
          className="hidden"
          disabled={disabled}
          onChange={(e) => { const f = e.target.files[0]; if (f) onFile(f) }}
        />
      </div>
      {!consented && <p className="text-[12px] text-bone-faint">Check the box above to continue</p>}
      {error && <p className="text-[13px] text-verdictDanger max-w-[360px] text-center">{error}</p>}
    </div>
  )
}

function GovernmentIdFlow({ onVerified, onCancel }) {
  const [step, setStep] = useState('capture-id') // 'capture-id' | 'verify-face'
  const [idConsent, setIdConsent] = useState(false)
  const [idBusy, setIdBusy] = useState(false)
  const [idError, setIdError] = useState(null)
  const [idReference, setIdReference] = useState(null)
  const [matchBusy, setMatchBusy] = useState(false)
  const [matchError, setMatchError] = useState(null)
  const [matchResult, setMatchResult] = useState(null)
  const [showCamera, setShowCamera] = useState(true)

  const handleIdFile = async (file) => {
    if (!idConsent) return
    setIdBusy(true)
    setIdError(null)
    try {
      const uploadRes = await uploadMedia(file)
      const ref = await verifyGovernmentId(uploadRes.data.upload_id, true)
      setIdReference(ref)
      setStep('verify-face')
    } catch (err) {
      setIdError(err.message)
    } finally {
      setIdBusy(false)
    }
  }

  const handleFaceCapture = async (file) => {
    setShowCamera(false)
    setMatchBusy(true)
    setMatchError(null)
    try {
      const uploadRes = await uploadMedia(file)
      const result = await matchIdentity(uploadRes.data.upload_id)
      setMatchResult(result)
      if (result.verdict === 'MATCH') {
        onVerified(idReference)
      }
    } catch (err) {
      setMatchError(err.message)
    } finally {
      setMatchBusy(false)
    }
  }

  if (step === 'capture-id') {
    return (
      <div className="flex flex-col items-center gap-6 w-full">
        <p className="tl-hud-label !text-[9px]">Step 1 of 2 — Photograph Your ID</p>
        <GovernmentIdCapture
          busy={idBusy}
          error={idError}
          consented={idConsent}
          onConsentChange={setIdConsent}
          onFile={handleIdFile}
        />
        <button type="button" onClick={onCancel} className="text-[12px] text-bone-faint hover:text-bone link-underline">
          Cancel
        </button>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center gap-6 w-full">
      <p className="tl-hud-label !text-[9px]">Step 2 of 2 — Confirm It's You</p>
      <p className="text-[12px] text-bone-dim text-center max-w-[380px]">
        Look at the camera so we can confirm your live face matches the ID photo.
      </p>
      {showCamera && (
        <WebcamCapture onCapture={handleFaceCapture} onCancel={onCancel} className="w-full" />
      )}
      {matchBusy && <p className="text-[13px] text-bone-dim">Comparing against your ID photo…</p>}
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
      {matchResult && matchResult.verdict !== 'MATCH' && !matchBusy && (
        <div className="flex flex-col items-center gap-4 text-center">
          <p className="text-[14px] text-verdictDanger max-w-[360px]">
            That doesn't look like a match. Make sure you're well-lit and facing the camera.
          </p>
          <button
            type="button"
            onClick={() => { setMatchResult(null); setShowCamera(true) }}
            className="text-[12px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
          >
            Try again
          </button>
        </div>
      )}
    </div>
  )
}

function SessionHistory({ sessions }) {
  const [downloadingId, setDownloadingId] = useState(null)

  if (sessions === undefined) return null
  if (sessions.length === 0) return null

  const download = async (sessionId) => {
    setDownloadingId(sessionId)
    try {
      await downloadSessionReport(sessionId)
    } finally {
      setDownloadingId(null)
    }
  }

  return (
    <div className="w-full max-w-[420px] mt-10 pt-8 border-t border-line">
      <p className="tl-hud-label !text-[9px] mb-4">Your Past Sessions</p>
      <div className="flex flex-col gap-2">
        {sessions.slice(0, 10).map((s) => (
          <div key={s.session_id} className="flex items-center justify-between gap-3 py-1.5 border-b border-line last:border-b-0">
            <div className="min-w-0">
              <p className="text-[12px] text-bone-dim">
                {new Date(s.started_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
              </p>
              <p className="text-[11px] text-bone-faint">
                {s.total_checks} check{s.total_checks === 1 ? '' : 's'}
                {s.flagged_checks > 0 && <span className="text-verdictDanger"> · {s.flagged_checks} flagged</span>}
              </p>
            </div>
            <button
              type="button"
              onClick={() => download(s.session_id)}
              disabled={downloadingId === s.session_id}
              className="flex items-center gap-1 text-[11px] text-brass hover:text-bone link-underline shrink-0 disabled:opacity-50"
            >
              <Download size={12} strokeWidth={1.75} /> Report
            </button>
          </div>
        ))}
      </div>
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
  const [consented, setConsented] = useState(false)
  const [org, setOrg] = useState(undefined) // undefined=loading, null=not joined, object=joined
  const [switchingOrg, setSwitchingOrg] = useState(false)
  const [joinBusy, setJoinBusy] = useState(false)
  const [joinError, setJoinError] = useState(null)
  const [mySessions, setMySessions] = useState(undefined)
  const [enrollMode, setEnrollMode] = useState(null) // null=choice, 'id'=government ID flow, 'selfie'=plain flow

  useEffect(() => {
    let cancelled = false
    getIdentityReference()
      .then((ref) => { if (!cancelled) setReference(ref) })
      .catch(() => { if (!cancelled) setReference(null) })
    getMyOrganization()
      .then((o) => { if (!cancelled) setOrg(o) })
      .catch(() => { if (!cancelled) setOrg(null) })
    getMySessions()
      .then((rows) => { if (!cancelled) setMySessions(rows) })
      .catch(() => { if (!cancelled) setMySessions([]) })
    return () => { cancelled = true }
  }, [])

  const handleJoin = async (code) => {
    setJoinBusy(true)
    setJoinError(null)
    try {
      const result = await joinOrganization(code)
      setOrg(result)
      setSwitchingOrg(false)
    } catch (err) {
      setJoinError(err.message)
    } finally {
      setJoinBusy(false)
    }
  }

  const handleGovernmentIdVerified = (ref) => {
    setReference(ref)
    setEnrollMode(null)
  }

  const enroll = async (file) => {
    if (!consented) return
    setEnrollBusy(true)
    setEnrollError(null)
    try {
      const uploadRes = await uploadMedia(file)
      const ref = await enrollIdentityReference(uploadRes.data.upload_id, true)
      setReference(ref)
      setMatchResult(null)
      setConsented(false)
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

            {reference === null && enrollMode === null && (
              <EnrollmentChoice
                onChooseId={() => setEnrollMode('id')}
                onChooseSelfie={() => setEnrollMode('selfie')}
              />
            )}

            {reference === null && enrollMode === 'id' && (
              <GovernmentIdFlow onVerified={handleGovernmentIdVerified} onCancel={() => setEnrollMode(null)} />
            )}

            {reference === null && enrollMode === 'selfie' && (
              <>
                <div className="flex items-start gap-3 mb-8 max-w-[420px] text-left">
                  <ShieldQuestion size={18} strokeWidth={1.5} className="text-bone-faint shrink-0 mt-0.5" />
                  <p className="text-[12px] text-bone-dim leading-relaxed">
                    No reference photo enrolled yet. Upload a clear, front-facing, unobstructed
                    photo of just you once — it needs to be sharp and well-framed. This path does
                    not unlock exam access.
                  </p>
                </div>
                <EnrollDropzone
                  busy={enrollBusy}
                  error={enrollError}
                  consented={consented}
                  onConsentChange={setConsented}
                  onFile={enroll}
                />
                <button
                  type="button"
                  onClick={() => setEnrollMode(null)}
                  className="mt-5 text-[12px] text-bone-faint hover:text-bone link-underline"
                >
                  Back
                </button>
              </>
            )}

            {reference && org === undefined && (
              <p className="text-[13px] text-bone-dim">Loading…</p>
            )}

            {reference && (org === null || switchingOrg) && (
              <>
                <p className="tl-hud-label !text-[9px] mb-5">Your Organisation</p>
                <JoinOrganizationForm
                  busy={joinBusy}
                  error={joinError}
                  onJoin={handleJoin}
                  onCancel={switchingOrg ? () => setSwitchingOrg(false) : null}
                />
              </>
            )}

            {reference && org && !switchingOrg && !showCamera && !matchBusy && !matchResult && !matchError && (
              <div className="flex flex-col items-center gap-6 text-center">
                <p className="tl-hud-label !text-[9px]">Reference enrolled</p>
                <p className="text-[12px] text-bone-dim">
                  Linked to <span className="text-brass">{org.organization_name}</span> ·{' '}
                  <button
                    type="button"
                    onClick={() => setSwitchingOrg(true)}
                    className="text-brass hover:text-bone link-underline"
                  >
                    Switch
                  </button>
                </p>
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
                <SessionHistory sessions={mySessions} />
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
