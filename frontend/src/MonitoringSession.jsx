import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Download, ShieldAlert, ShieldCheck } from 'lucide-react'
import {
  downloadSessionReport,
  endMonitoringSession,
  getIdentityReference,
  postMonitoringEvent,
  startMonitoringSession,
  submitMonitoringCheck,
  uploadMedia,
} from './api/client'

const NORMAL_INTERVAL_MS = 20000
const HEIGHTENED_INTERVAL_MS = 5000
const HEIGHTENED_WINDOW_MS = 60000

// Don't bother reporting a tab/window switch shorter than this -- avoids spamming an event for
// every incidental alt-tab flicker.
const AWAY_THRESHOLD_S = 2
const DEVTOOLS_CHECK_MS = 2000
const DEVTOOLS_SIZE_THRESHOLD_PX = 160

const REASON_LABELS = {
  no_face: 'No face detected',
  multiple_faces: 'Multiple faces detected',
  identity_mismatch: 'Identity mismatch',
  identity_uncertain: 'Identity uncertain',
  deepfake_signal: 'Possible AI manipulation',
  deepfake_uncertain: 'Manipulation uncertain',
}

const EVENT_LABELS = {
  tab_hidden: 'Tab hidden',
  window_blurred: 'Window lost focus',
  clipboard_paste: 'Clipboard paste',
  devtools_suspected: 'Devtools suspected',
  camera_interrupted: 'Camera interrupted',
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
  const [session, setSession] = useState(null)
  const [checks, setChecks] = useState([])
  const [events, setEvents] = useState([])
  const [error, setError] = useState(null)
  const [starting, setStarting] = useState(false)
  const [lastSessionId, setLastSessionId] = useState(null) // stays set after Stop, so the report stays downloadable

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const checkTimeoutRef = useRef(null)
  const sessionRef = useRef(null) // mirrors `session` for use inside async/interval closures
  const lastFlagAtRef = useRef(0) // drives the adaptive check interval
  const deliberateStopRef = useRef(false) // suppresses a spurious camera_interrupted on our own Stop click
  const hiddenAtRef = useRef(null)
  const blurredAtRef = useRef(null)
  const devtoolsOpenRef = useRef(false)

  useEffect(() => {
    let cancelled = false
    getIdentityReference()
      .then((ref) => { if (!cancelled) setReference(ref) })
      .catch(() => { if (!cancelled) setReference(null) })
    return () => { cancelled = true }
  }, [])

  const stopStream = () => {
    deliberateStopRef.current = true
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
  }

  const stopScheduledCheck = () => {
    if (checkTimeoutRef.current) {
      clearTimeout(checkTimeoutRef.current)
      checkTimeoutRef.current = null
    }
  }

  // Releases the camera and stops any scheduled check if the user navigates away mid-session,
  // same cleanup discipline the stream itself already had.
  useEffect(() => () => { stopScheduledCheck(); stopStream() }, [])

  const postEvent = useCallback((eventType, detail) => {
    const activeSession = sessionRef.current
    if (!activeSession) return
    postMonitoringEvent(activeSession.session_id, eventType, detail != null ? String(detail) : undefined)
      .then((event) => setEvents((prev) => [event, ...prev]))
      .catch(() => { /* best-effort -- a missed behavioral event shouldn't surface as a user-facing error */ })
  }, [])

  // Tab/window-focus, clipboard-paste and devtools-heuristic listeners -- only wired up while a
  // session is active, torn down on stop/unmount. These only ever see activity inside this
  // browser tab (stated in the page copy below, not just here).
  useEffect(() => {
    if (!session) return undefined

    const onVisibilityChange = () => {
      if (document.hidden) {
        hiddenAtRef.current = Date.now()
      } else if (hiddenAtRef.current) {
        const awaySeconds = (Date.now() - hiddenAtRef.current) / 1000
        hiddenAtRef.current = null
        if (awaySeconds >= AWAY_THRESHOLD_S) {
          lastFlagAtRef.current = Date.now()
          postEvent('tab_hidden', awaySeconds.toFixed(1))
        }
      }
    }
    const onBlur = () => { blurredAtRef.current = Date.now() }
    const onFocus = () => {
      if (blurredAtRef.current) {
        const awaySeconds = (Date.now() - blurredAtRef.current) / 1000
        blurredAtRef.current = null
        if (awaySeconds >= AWAY_THRESHOLD_S) {
          lastFlagAtRef.current = Date.now()
          postEvent('window_blurred', awaySeconds.toFixed(1))
        }
      }
    }
    const onPaste = () => {
      lastFlagAtRef.current = Date.now()
      postEvent('clipboard_paste')
    }
    const devtoolsInterval = setInterval(() => {
      const widthDiff = window.outerWidth - window.innerWidth
      const heightDiff = window.outerHeight - window.innerHeight
      const isOpen = widthDiff > DEVTOOLS_SIZE_THRESHOLD_PX || heightDiff > DEVTOOLS_SIZE_THRESHOLD_PX
      if (isOpen && !devtoolsOpenRef.current) {
        lastFlagAtRef.current = Date.now()
        postEvent('devtools_suspected')
      }
      devtoolsOpenRef.current = isOpen
    }, DEVTOOLS_CHECK_MS)

    document.addEventListener('visibilitychange', onVisibilityChange)
    window.addEventListener('blur', onBlur)
    window.addEventListener('focus', onFocus)
    document.addEventListener('paste', onPaste)

    return () => {
      document.removeEventListener('visibilitychange', onVisibilityChange)
      window.removeEventListener('blur', onBlur)
      window.removeEventListener('focus', onFocus)
      document.removeEventListener('paste', onPaste)
      clearInterval(devtoolsInterval)
    }
  }, [session, postEvent])

  const runCheck = useCallback(() => {
    const video = videoRef.current
    const activeSession = sessionRef.current
    const track = streamRef.current?.getVideoTracks()[0]
    if (!video || !video.videoWidth || !activeSession || !track || track.readyState !== 'live') return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    canvas.toBlob(async (blob) => {
      if (!blob) return
      try {
        const file = new File([blob], 'monitor-frame.jpg', { type: 'image/jpeg' })
        const uploadRes = await uploadMedia(file)
        const check = await submitMonitoringCheck(activeSession.session_id, uploadRes.data.upload_id)
        setChecks((prev) => [check, ...prev])
        if (check.flagged) lastFlagAtRef.current = Date.now()
      } catch (err) {
        if (err.status === 400) {
          // The session ended server-side (e.g. a previous /end call) -- stop quietly.
          stopScheduledCheck()
          stopStream()
          return
        }
        setError(err.message)
      }
    }, 'image/jpeg', 0.92)
  }, [])

  // Self-rescheduling instead of a fixed interval: right after a flagged check (or a flagged
  // behavioral event -- a tab-hidden moment deserves denser camera evidence too), checks run
  // every 5s for a minute, then relax back to the normal 20s cadence.
  //
  // Re-checks the track's own readyState before each scheduling decision (not just inside
  // runCheck) -- caught by testing: without this, a camera that dies outside of our own Stop
  // click (unplugged, OS permission revoked) left the loop silently re-submitting black frames
  // every cycle instead of stopping, flooding the timeline with meaningless "no face"/"possible
  // AI manipulation" flags from empty frames.
  const scheduleNextCheck = useCallback(() => {
    const track = streamRef.current?.getVideoTracks()[0]
    if (!track || track.readyState !== 'live') {
      stopScheduledCheck()
      return
    }
    const heightened = Date.now() - lastFlagAtRef.current < HEIGHTENED_WINDOW_MS
    const delay = heightened ? HEIGHTENED_INTERVAL_MS : NORMAL_INTERVAL_MS
    checkTimeoutRef.current = setTimeout(async () => {
      await runCheck()
      scheduleNextCheck()
    }, delay)
  }, [runCheck])

  const startSession = async () => {
    setStarting(true)
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
      streamRef.current = stream
      deliberateStopRef.current = false
      // Per spec, track.stop() called by the page itself never fires 'ended' -- it's reserved
      // for interruptions outside the app's control (device unplugged, OS permission revoked,
      // a virtual-camera driver dropping the feed), so deliberateStopRef is defense-in-depth
      // for engine differences, not something that's actually load-bearing against our own
      // Stop button (confirmed directly: our own stopStream() never triggers this handler).
      const [track] = stream.getVideoTracks()
      if (track) {
        track.onended = () => {
          if (deliberateStopRef.current) return
          postEvent('camera_interrupted', 'track ended')
        }
        track.onmute = () => {
          if (deliberateStopRef.current) return
          postEvent('camera_interrupted', 'muted')
        }
      }
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      const started = await startMonitoringSession()
      setSession(started)
      sessionRef.current = started
      setLastSessionId(started.session_id)
      setChecks([])
      setEvents([])
      lastFlagAtRef.current = 0
      scheduleNextCheck()
    } catch (err) {
      stopStream()
      setError(err?.message || 'Could not access the camera.')
    } finally {
      setStarting(false)
    }
  }

  const stopSession = async () => {
    stopScheduledCheck()
    stopStream()
    const activeSession = sessionRef.current
    sessionRef.current = null
    setSession(null)
    if (activeSession) {
      try { await endMonitoringSession(activeSession.session_id) } catch { /* best-effort */ }
    }
  }

  const flaggedCount = checks.filter((c) => c.flagged).length + events.length // every event is inherently a flag
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
              Checks your face, identity match, and manipulation signal — every 20s normally,
              every 5s for a minute after anything looks off. Also watches for tab switches,
              clipboard pastes and devtools. All of this only sees activity in this browser tab.
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

            {reference && (
              <>
                <div className="relative w-full max-w-[640px] mx-auto aspect-[4/3] bg-ground border border-line rounded-[4px] overflow-hidden mb-6">
                  <video ref={videoRef} muted playsInline className="w-full h-full object-cover scale-x-[-1]" />
                  {!session && (
                    <div className="absolute inset-0 flex items-center justify-center text-bone-faint text-[13px]">
                      Camera off
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-center gap-4 mb-2">
                  {!session ? (
                    <button
                      type="button"
                      disabled={starting}
                      onClick={startSession}
                      className="bg-brass text-ground px-6 py-3 rounded-[4px] text-[12px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-50"
                    >
                      {starting ? 'Starting…' : 'Start Session'}
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
