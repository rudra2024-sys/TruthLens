import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Download, ShieldAlert, ShieldCheck } from 'lucide-react'
import FaceOverlay from './components/FaceOverlay'
import LiveFaceMesh from './components/LiveFaceMesh'
import useMonitoringSession from './hooks/useMonitoringSession'
import { downloadSessionReport, getIdentityReference, getMyOrganization } from './api/client'

const ArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 8H13" /><path d="M9 4L13 8L9 12" />
  </svg>
)

const REASON_LABELS = {
  no_face: 'No face detected',
  multiple_faces: 'Multiple faces detected',
  identity_mismatch: 'Identity mismatch',
  identity_uncertain: 'Identity uncertain',
  deepfake_signal: 'Possible AI manipulation',
  deepfake_uncertain: 'Manipulation uncertain',
  looking_away: 'Looking away',
  talking_detected: 'Talking detected',
  object_detected: 'Unauthorized object detected',
  device_mismatch_suspected: 'Possible device change',
  static_image_suspected: 'Possible static photo detected',
}

const EVENT_LABELS = {
  tab_hidden: 'Tab hidden',
  window_blurred: 'Window lost focus',
  clipboard_paste: 'Clipboard paste',
  devtools_suspected: 'Devtools suspected',
  camera_interrupted: 'Camera interrupted',
  live_no_face: 'Live: no face detected',
  live_looking_away: 'Live: looking away',
  live_multiple_faces: 'Live: second person detected',
}

function CheckRow({ check }) {
  const time = new Date(check.checked_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  return (
    <div className="flex items-start gap-3 py-3 border-b border-line last:border-b-0">
      <div
        className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${check.flagged ? 'bg-verdictDanger' : 'bg-verdictReal'}`}
        aria-hidden="true"
      />
      <div className="flex-1">
        <div className="flex items-center justify-between gap-3">
          <span className="tl-figure text-[12px] text-bone-dim">{time}</span>
          <span className="tl-figure text-[11px] text-bone-faint">
            {check.similarity_score != null ? `similarity ${Math.round(check.similarity_score * 100)}%` : '—'}
            {' · '}
            fake {Math.round(check.fake_probability * 100)}%
            {check.object_detections?.length > 0 ? ` · ${check.object_detections.join(', ')} detected` : ''}
          </span>
        </div>
        {check.flagged ? (
          <div className="flex flex-wrap gap-2 mt-1.5">
            {check.flag_reasons.map((reason) => (
              <span
                key={reason}
                className="text-[10px] font-medium tracking-[0.06em] uppercase text-verdictDanger bg-[rgba(209,101,101,0.1)] border border-[rgba(209,101,101,0.3)] rounded-[3px] px-2 py-1"
              >
                {REASON_LABELS[reason] || reason}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-[12px] text-bone-dim mt-1">Normal</p>
        )}
      </div>
    </div>
  )
}

function EventRow({ event }) {
  const time = new Date(event.occurred_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  return (
    <div className="flex items-start gap-3 py-3 border-b border-line last:border-b-0">
      <div className="w-2 h-2 rounded-full mt-1.5 shrink-0 bg-verdictCaution" aria-hidden="true" />
      <div className="flex-1">
        <div className="flex items-center justify-between gap-3">
          <span className="tl-figure text-[12px] text-bone-dim">{time}</span>
        </div>
        <p className="text-[12px] text-verdictCaution mt-1">
          {EVENT_LABELS[event.event_type] || event.event_type}
          {event.detail ? ` — ${/^[\d.]+$/.test(event.detail) ? `${event.detail}s` : event.detail}` : ''}
        </p>
      </div>
    </div>
  )
}

export default function MonitoringSession() {
  const [reference, setReference] = useState(undefined) // undefined=loading, null=not enrolled, object=enrolled
  const [acknowledged, setAcknowledged] = useState(false) // instructions gate -- re-shown every visit, not persisted
  const [orgName, setOrgName] = useState(null) // display-only, for the instructions card's org-name line

  const {
    videoRef, session, checks, events, error, setError, starting, lastSessionId,
    startSession, stopSession, handleSustainedIssue,
    lastCheck, flaggedCount, guidanceText,
  } = useMonitoringSession()

  useEffect(() => {
    let cancelled = false
    getIdentityReference()
      .then((ref) => { if (!cancelled) setReference(ref) })
      .catch(() => { if (!cancelled) setReference(null) })
    getMyOrganization()
      .then((org) => { if (!cancelled && org) setOrgName(org.organization_name) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  const timeline = [
    ...checks.map((c) => ({ kind: 'check', ts: c.checked_at, key: `check-${c.check_id}`, data: c })),
    ...events.map((e) => ({ kind: 'event', ts: e.occurred_at, key: `event-${e.event_id}`, data: e })),
  ].sort((a, b) => (a.ts < b.ts ? 1 : -1))

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />
      <section className="relative min-h-[calc(100vh-72px)] flex flex-col items-center py-16 md:py-24 tl-inspection-grid">
        <div className="absolute inset-0 bg-gradient-to-b from-ground via-ground to-panel/40" />

        <div className="relative z-10 max-w-[840px] mx-auto px-6 w-full animate-fade-in-up animate-delay-1">
          <div className="text-center mb-12">
            <p className="tl-hud-label !text-brass mb-4">Continuous Monitoring</p>
            <h1 className="font-serif text-display-l text-bone mb-6">
              Stay under <span className="italic text-brass">watch.</span>
            </h1>
            <p className="text-[16px] text-bone-dim max-w-[560px] mx-auto">
              Requires camera and microphone access. Review what's tracked before you begin.
            </p>
          </div>

          <div className="bg-panel border border-line rounded-[6px] p-10 md:p-12">
            {reference === undefined && <p className="text-[13px] text-bone-dim text-center">Loading…</p>}

            {reference === null && (
              <p className="text-[13px] text-bone-dim text-center">
                No reference photo enrolled yet.{' '}
                <Link to="/identity" className="text-brass hover:text-bone link-underline">Enroll one first</Link>.
              </p>
            )}

            {reference && !acknowledged && (
              <div className="max-w-[520px] mx-auto">
                <p className="tl-hud-label !text-[11px] mb-4">Before you begin</p>
                <ul className="space-y-3 text-[14px] text-bone-dim leading-relaxed mb-8">
                  <li className="flex gap-3">
                    <ShieldAlert size={16} strokeWidth={1.5} className="text-brass shrink-0 mt-0.5" />
                    <span>Your camera and microphone will be used for the entire session.</span>
                  </li>
                  <li className="flex gap-3">
                    <ShieldAlert size={16} strokeWidth={1.5} className="text-brass shrink-0 mt-0.5" />
                    <span>
                      Each check compares your face against your enrolled reference, screens for
                      signs of manipulation, and watches gaze, talking, and phones/books in frame
                      — every 20s normally, every 5s for a minute after anything looks off.
                    </span>
                  </li>
                  <li className="flex gap-3">
                    <ShieldAlert size={16} strokeWidth={1.5} className="text-brass shrink-0 mt-0.5" />
                    <span>
                      Tab switches, window focus loss, clipboard pastes, and devtools use in this
                      browser tab are also logged. Activity outside this tab is not visible to it.
                    </span>
                  </li>
                  <li className="flex gap-3">
                    <ShieldCheck size={16} strokeWidth={1.5} className="text-brass shrink-0 mt-0.5" />
                    <span>
                      A session report is generated for <strong className="text-brass">{orgName || 'your organisation'}</strong> to
                      review — nothing here is proof of misconduct on its own, it's a record for a
                      human to judge.
                    </span>
                  </li>
                </ul>
                <div className="flex justify-center">
                  <button
                    type="button"
                    onClick={() => setAcknowledged(true)}
                    className="group flex items-center gap-3 bg-brass text-ground px-8 py-4 rounded-[4px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift"
                  >
                    I Understand — Continue
                    <span className="transition-transform duration-300 group-hover:translate-x-1"><ArrowRight /></span>
                  </button>
                </div>
              </div>
            )}

            {reference && acknowledged && (
              <>
                <div className="relative w-full max-w-[640px] mx-auto aspect-[4/3] bg-ground border border-line rounded-[4px] overflow-hidden mb-6">
                  {/* Mirrored together so the overlay's coordinates (in the camera's own,
                      unmirrored pixel space) line up with the mirrored on-screen preview --
                      mirroring only the <video> via CSS would leave the net flipped relative
                      to the face. */}
                  <div className="relative w-full h-full scale-x-[-1]">
                    <video ref={videoRef} muted playsInline className="w-full h-full object-cover" />
                    <LiveFaceMesh videoRef={videoRef} active={!!session} onSustainedIssue={handleSustainedIssue} />
                    {session && lastCheck?.image_width && (
                      <FaceOverlay
                        box={lastCheck.face_box}
                        landmarks={lastCheck.landmarks}
                        imageWidth={lastCheck.image_width}
                        imageHeight={lastCheck.image_height}
                        identityVerdict={lastCheck.identity_verdict}
                      />
                    )}
                  </div>
                  {!session && (
                    <div className="absolute inset-0 flex items-center justify-center text-bone-faint text-[13px]">
                      Camera off
                    </div>
                  )}
                  {session && guidanceText && (
                    <div className="absolute inset-x-0 bottom-0 bg-ground/85 border-t border-line px-4 py-2.5 text-center">
                      <p className="text-[12px] text-bone leading-snug">{guidanceText}</p>
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-center gap-4 mb-2">
                  {!session ? (
                    <button
                      type="button"
                      disabled={starting}
                      onClick={startSession}
                      className="group flex items-center gap-3 bg-brass text-ground px-8 py-4 rounded-[4px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-50"
                    >
                      {starting ? 'Starting…' : 'Start Session'}
                      <span className="transition-transform duration-300 group-hover:translate-x-1"><ArrowRight /></span>
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={stopSession}
                      className="bg-panel-raised border border-line-strong text-bone px-6 py-3 rounded-[4px] text-[12px] font-medium tracking-[0.08em] uppercase btn-lift"
                    >
                      Stop Session
                    </button>
                  )}
                  {lastSessionId && (
                    <button
                      type="button"
                      onClick={() => downloadSessionReport(lastSessionId).catch((err) => setError(err.message))}
                      className="flex items-center gap-2 text-[12px] font-medium tracking-[0.06em] text-bone-dim hover:text-bone transition-colors duration-300"
                    >
                      <Download size={14} strokeWidth={1.75} /> Download Report
                    </button>
                  )}
                </div>

                {error && <p className="text-[13px] text-verdictDanger text-center mt-2">{error}</p>}

                {session && (
                  <div className="flex items-center justify-center gap-2 mt-4 mb-2 text-[12px] text-bone-dim">
                    {flaggedCount > 0 ? (
                      <>
                        <ShieldAlert size={14} strokeWidth={1.5} className="text-verdictDanger" />
                        <span>{flaggedCount} flagged item{flaggedCount === 1 ? '' : 's'}</span>
                      </>
                    ) : (
                      <>
                        <ShieldCheck size={14} strokeWidth={1.5} className="text-verdictReal" />
                        <span>No flags yet</span>
                      </>
                    )}
                  </div>
                )}

                {timeline.length > 0 && (
                  <div className="mt-6 pt-6 border-t border-line">
                    <p className="tl-hud-label !text-[9px] mb-2">Timeline</p>
                    {timeline.map((item) => (
                      item.kind === 'check'
                        ? <CheckRow key={item.key} check={item.data} />
                        : <EventRow key={item.key} event={item.data} />
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
