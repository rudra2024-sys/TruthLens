import { useCallback, useEffect, useRef, useState } from 'react'
import {
  endMonitoringSession,
  getMonitoringCheckJob,
  postMonitoringEvent,
  startMonitoringCheckJob,
  startMonitoringSession,
  uploadMedia,
} from '../api/client'

/**
 * Extracted from MonitoringSession.jsx (added 2026-10-12) so the exam-taking page can reuse the
 * entire camera lifecycle / adaptive check loop / behavioral-event tracking unchanged, without
 * MonitoringSession.jsx's own page structure (instructions gate, timeline rendering) leaking
 * into a page that needs a materially different shape (questions, answer state, a lock screen).
 * MonitoringSession.jsx itself is now a thin consumer of this hook -- same rendered output as
 * before the extraction.
 *
 * `startSessionFn` defaults to the plain startMonitoringSession() call; the exam-taking page
 * passes a function that starts an exam attempt instead (which creates the linked
 * MonitoringSession server-side as part of the same transaction) -- the hook only ever needs
 * the resulting object's `.session_id`, so it doesn't care which shape (MonitoringSessionOut vs
 * ExamAttemptOut) the caller's start function returns.
 *
 * `onCheck(check)` fires after every completed check (flagged or not) -- the exam page uses it
 * to trigger its own attempt-status poll, since server-side flag-threshold enforcement runs
 * inside that same check request and the exam page needs to notice promptly if it crossed the
 * line.
 */

const CHECK_POLL_INTERVAL_MS = 300
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

const NORMAL_INTERVAL_MS = 20000
const HEIGHTENED_INTERVAL_MS = 5000
const HEIGHTENED_WINDOW_MS = 60000

const AWAY_THRESHOLD_S = 2
const DEVTOOLS_CHECK_MS = 2000
const DEVTOOLS_SIZE_THRESHOLD_PX = 160
const AUDIO_CLIP_MS = 3000

const canvasToBlob = (canvas) => new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92))

const recordAudioClip = (stream, durationMs = AUDIO_CLIP_MS) => new Promise((resolve) => {
  const audioTrack = stream?.getAudioTracks()[0]
  if (!audioTrack || audioTrack.readyState !== 'live') { resolve(null); return }
  let recorder
  try {
    recorder = new MediaRecorder(new MediaStream([audioTrack]), { mimeType: 'audio/webm' })
  } catch {
    resolve(null)
    return
  }
  const chunks = []
  recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data) }
  recorder.onstop = () => resolve(chunks.length ? new Blob(chunks, { type: 'audio/webm' }) : null)
  recorder.onerror = () => resolve(null)
  recorder.start()
  setTimeout(() => { if (recorder.state !== 'inactive') recorder.stop() }, durationMs)
})

const REASON_GUIDANCE = {
  no_face: "We can't see your face. Center it in frame, forehead to chin, with nothing blocking it.",
  multiple_faces: 'More than one face is in frame. Make sure only you are visible.',
  identity_uncertain: "We're having trouble matching you to your reference photo. Face the camera directly in good light.",
  identity_mismatch: "This doesn't look like a match to your reference photo. Face the camera directly in good light.",
  static_image_suspected: 'This looks like it might be a still photo rather than a live feed. Move naturally in frame.',
}
const REASON_GUIDANCE_PRIORITY = ['no_face', 'multiple_faces', 'identity_mismatch', 'identity_uncertain', 'static_image_suspected']

export default function useMonitoringSession({ onCheck, startSessionFn = startMonitoringSession } = {}) {
  const [session, setSession] = useState(null)
  const [checks, setChecks] = useState([])
  const [events, setEvents] = useState([])
  const [error, setError] = useState(null)
  const [starting, setStarting] = useState(false)
  const [lastSessionId, setLastSessionId] = useState(null)

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const checkTimeoutRef = useRef(null)
  const sessionRef = useRef(null)
  const lastFlagAtRef = useRef(0)
  const deliberateStopRef = useRef(false)
  const hiddenAtRef = useRef(null)
  const blurredAtRef = useRef(null)
  const devtoolsOpenRef = useRef(false)
  const onCheckRef = useRef(onCheck)
  onCheckRef.current = onCheck

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

  useEffect(() => () => { stopScheduledCheck(); stopStream() }, [])

  const postEvent = useCallback((eventType, detail) => {
    const activeSession = sessionRef.current
    if (!activeSession) return
    postMonitoringEvent(activeSession.session_id, eventType, detail != null ? String(detail) : undefined)
      .then((event) => setEvents((prev) => [event, ...prev]))
      .catch(() => { /* best-effort -- a missed behavioral event shouldn't surface as a user-facing error */ })
  }, [])

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

  const runCheck = useCallback(async () => {
    const video = videoRef.current
    const activeSession = sessionRef.current
    const stream = streamRef.current
    const track = stream?.getVideoTracks()[0]
    if (!video || !video.videoWidth || !activeSession || !track || track.readyState !== 'live') return

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)

    const [videoBlob, audioBlob] = await Promise.all([canvasToBlob(canvas), recordAudioClip(stream)])
    if (!videoBlob) return

    try {
      const file = new File([videoBlob], 'monitor-frame.jpg', { type: 'image/jpeg' })
      const uploadRes = await uploadMedia(file)

      let audioUploadId
      if (audioBlob) {
        const audioFile = new File([audioBlob], 'monitor-audio.webm', { type: 'audio/webm' })
        const audioUploadRes = await uploadMedia(audioFile)
        audioUploadId = audioUploadRes.data.upload_id
      }

      let job = await startMonitoringCheckJob(activeSession.session_id, uploadRes.data.upload_id, audioUploadId)
      while (job.state === 'queued' || job.state === 'running') {
        await sleep(CHECK_POLL_INTERVAL_MS)
        if (sessionRef.current?.session_id !== activeSession.session_id) return
        job = await getMonitoringCheckJob(activeSession.session_id, job.job_id)
      }
      if (job.state === 'failed') {
        setError(job.error || 'The check failed.')
        return
      }
      const check = job.result
      setChecks((prev) => [check, ...prev])
      if (check.flagged) lastFlagAtRef.current = Date.now()
      onCheckRef.current?.(check)
    } catch (err) {
      if (err.status === 400) {
        stopScheduledCheck()
        stopStream()
        return
      }
      setError(err.message)
    }
  }, [])

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

  const handleSustainedIssue = useCallback((type) => {
    const eventType = { no_face: 'live_no_face', looking_away: 'live_looking_away', multiple_faces: 'live_multiple_faces' }[type]
    if (!eventType || !sessionRef.current) return
    lastFlagAtRef.current = Date.now()
    postEvent(eventType)
    stopScheduledCheck()
    runCheck().then(scheduleNextCheck)
  }, [postEvent, runCheck, scheduleNextCheck])

  const startSession = async () => {
    setStarting(true)
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: true })
      streamRef.current = stream
      deliberateStopRef.current = false
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
      const started = await startSessionFn()
      setSession(started)
      sessionRef.current = started
      setLastSessionId(started.session_id)
      setChecks([])
      setEvents([])
      lastFlagAtRef.current = 0
      scheduleNextCheck()
      return started
    } catch (err) {
      stopStream()
      setError(err?.message || 'Could not access the camera.')
      return null
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

  const lastCheck = checks[0]
  const flaggedCount = checks.filter((c) => c.flagged).length + events.length
  const guidanceReason = lastCheck?.flag_reasons?.find((r) => REASON_GUIDANCE_PRIORITY.includes(r))
  const guidanceText = guidanceReason ? REASON_GUIDANCE[guidanceReason] : null

  return {
    videoRef, session, checks, events, error, setError, starting, lastSessionId,
    startSession, stopSession, handleSustainedIssue,
    lastCheck, flaggedCount, guidanceText,
  }
}
