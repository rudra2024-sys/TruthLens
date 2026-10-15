import React, { useEffect, useRef, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ShieldAlert, ShieldCheck, Lock } from 'lucide-react'
import GovernmentIdCapture from './components/GovernmentIdCapture'
import LiveFaceMesh from './components/LiveFaceMesh'
import WebcamCapture from './components/WebcamCapture'
import useMonitoringSession from './hooks/useMonitoringSession'
import { REASON_LABELS, EVENT_LABELS } from './monitoringLabels'
import {
  finishExamAttempt,
  getExam,
  getExamAttempt,
  getExamQuestions,
  startExamAttempt,
  submitExamAnswer,
  uploadMedia,
  verifyIdentityForAttempt,
} from './api/client'

const ArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 8H13" /><path d="M9 4L13 8L9 12" />
  </svg>
)

function QuestionCard({ question, selected, onSelect, disabled }) {
  return (
    <div className="border-b border-line py-6 last:border-b-0">
      <p className="text-[17px] text-bone mb-4">
        <span className="text-bone-faint mr-2">{question.order_index + 1}.</span>
        {question.text}
      </p>
      <div className="flex flex-col gap-2">
        {question.options.map((opt, i) => (
          <button
            key={i}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(question.question_id, i)}
            className={`text-left px-4 py-3.5 rounded-[4px] border text-[15px] transition-colors duration-300 disabled:opacity-60 disabled:pointer-events-none ${
              selected === i
                ? 'border-brass bg-[rgba(200,147,97,0.06)] text-brass'
                : 'border-line hover:border-line-strong text-bone-dim'
            }`}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function ExamAttempt() {
  const { examId } = useParams()
  const [exam, setExam] = useState(undefined) // undefined=loading, null=not found, object=loaded
  const [attempt, setAttempt] = useState(null)
  const [questions, setQuestions] = useState(null)
  const [answers, setAnswers] = useState({}) // question_id -> selected_option_index
  const [rollNo, setRollNo] = useState('')
  const [startError, setStartError] = useState(null)
  const [finishing, setFinishing] = useState(false)

  // Per-attempt identity check, run fresh every time (not a one-time enrollment): upload a
  // government ID, confirm it's you with a live camera look, then agree to terms -- only then
  // does the test itself (questions + continuous monitoring) begin. See CLAUDE.md's ProctorLens
  // section and app/services/identity/detector.py::verify_identity_with_selfie.
  const [step, setStep] = useState('verify-id') // 'verify-id' | 'verify-face' | 'terms'
  const [idConsent, setIdConsent] = useState(false)
  const [idBusy, setIdBusy] = useState(false)
  const [idError, setIdError] = useState(null)
  const [idUploadId, setIdUploadId] = useState(null)
  const [faceBusy, setFaceBusy] = useState(false)
  const [faceError, setFaceError] = useState(null)
  const [verifyResult, setVerifyResult] = useState(null) // { similarity, verdict, passed } once checked

  const attemptRef = useRef(null)
  attemptRef.current = attempt

  const handleCheck = () => {
    const current = attemptRef.current
    if (!current || current.status !== 'in_progress') return
    getExamAttempt(current.attempt_id)
      .then((updated) => setAttempt(updated))
      .catch(() => {})
  }

  const {
    videoRef, session, checks, events, starting, error, stopSession, startSession, resumeSession, handleSustainedIssue, flaggedCount, guidanceText,
  } = useMonitoringSession({
    onCheck: handleCheck,
    startSessionFn: async () => {
      const started = await startExamAttempt(examId, { roll_no: rollNo })
      setAttempt(started)
      return started
    },
  })

  useEffect(() => {
    let cancelled = false
    getExam(examId)
      .then((e) => { if (!cancelled) setExam(e) })
      .catch(() => { if (!cancelled) setExam(null) })
    return () => { cancelled = true }
  }, [examId])

  // Resuming an already-in-progress attempt (e.g. a page reload, or arriving via ProctorLens's
  // "Continue" link) -- jump straight past the identity-check wizard and re-attach the camera to
  // the existing server-side session, rather than attempting to start a second attempt (which the
  // backend rejects outright, see services/exams/attempts.py::start_attempt).
  useEffect(() => {
    if (!exam || attempt) return
    if (exam.my_attempt_status !== 'in_progress' || !exam.my_attempt_id) return
    let cancelled = false
    getExamAttempt(exam.my_attempt_id)
      .then((fetched) => {
        if (cancelled || fetched.status !== 'in_progress') return
        setAttempt(fetched)
        return resumeSession(fetched)
      })
      .catch(() => {})
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exam, attempt])

  useEffect(() => {
    if (!attempt || attempt.status !== 'in_progress' || questions) return
    getExamQuestions(examId)
      .then((qs) => setQuestions(qs))
      .catch(() => {})
  }, [attempt, examId, questions])

  // Once the server marks the attempt terminal (e.g. terminated_for_flags, detected via the
  // polling triggered by handleCheck above), stop the camera -- the server has already ended
  // things, this is purely client-side cleanup.
  useEffect(() => {
    if (attempt && attempt.status !== 'in_progress' && session) {
      stopSession()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt?.status])

  const handleIdFile = async (file) => {
    if (!idConsent) return
    setIdBusy(true)
    setIdError(null)
    try {
      const uploadRes = await uploadMedia(file)
      setIdUploadId(uploadRes.data.upload_id)
      setStep('verify-face')
    } catch (err) {
      setIdError(err.message)
    } finally {
      setIdBusy(false)
    }
  }

  const handleFaceCapture = async (file) => {
    setFaceBusy(true)
    setFaceError(null)
    try {
      const uploadRes = await uploadMedia(file)
      const result = await verifyIdentityForAttempt(examId, idUploadId, uploadRes.data.upload_id, true)
      setVerifyResult(result)
      if (result.passed) setStep('terms')
    } catch (err) {
      setFaceError(err.message)
    } finally {
      setFaceBusy(false)
    }
  }

  const handleRetryVerification = () => {
    setIdUploadId(null)
    setVerifyResult(null)
    setFaceError(null)
    setIdError(null)
    setIdConsent(false)
    setStep('verify-id')
  }

  const handleStart = async () => {
    setStartError(null)
    const started = await startSession()
    if (!started) setStartError(error || 'Could not start the exam.')
  }

  const handleSelect = (questionId, index) => {
    setAnswers((prev) => ({ ...prev, [questionId]: index }))
    submitExamAnswer(attempt.attempt_id, questionId, index).catch(() => {})
  }

  const handleFinish = async () => {
    setFinishing(true)
    try {
      const finished = await finishExamAttempt(attempt.attempt_id)
      setAttempt(finished)
    } finally {
      setFinishing(false)
    }
  }

  // Persistent, newest-first record of every flag so far -- the guidance banner only reflects
  // the single latest check and disappears the moment a later check passes, which isn't enough
  // for a student to see "I've been flagged twice for X" at a glance. Flagged checks + all
  // behavioral events (tab-hidden, devtools, clipboard paste, live mesh issues -- these are
  // always flag-worthy by definition) both count toward flaggedCount already, so both belong here.
  const flagHistory = [
    ...checks.filter((c) => c.flagged).map((c) => ({
      key: `check-${c.check_id}`, ts: c.checked_at,
      label: (c.flag_reasons || []).map((r) => REASON_LABELS[r] || r).join(', ') || 'Flagged',
    })),
    ...events.map((e) => ({ key: `event-${e.event_id}`, ts: e.occurred_at, label: EVENT_LABELS[e.event_type] || e.event_type })),
  ].sort((a, b) => (a.ts < b.ts ? 1 : -1))

  if (exam === undefined) {
    return <div className="min-h-screen bg-ground pt-[72px] flex items-center justify-center text-bone-dim text-[16px]">Loading…</div>
  }
  if (exam === null) {
    return (
      <div className="min-h-screen bg-ground pt-[72px] flex items-center justify-center text-bone-dim text-[16px]">
        Exam not found.
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />
      <section className="relative min-h-[calc(100vh-72px)] flex flex-col items-center py-16 md:py-24 tl-inspection-grid">
        <div className="absolute inset-0 bg-gradient-to-b from-ground via-ground to-panel/40" />

        <div
          className={`relative z-10 mx-auto px-6 w-full animate-fade-in-up animate-delay-1 ${
            attempt && attempt.status === 'in_progress' ? 'max-w-[1120px]' : 'max-w-[840px]'
          }`}
        >
          <div className="text-center mb-12">
            <p className="tl-hud-label !text-brass mb-4">ProctorLens</p>
            <h1 className="font-serif text-display-l text-bone mb-6">{exam.title}</h1>
            <p className="text-[16px] text-bone-dim">
              Pass threshold: {exam.pass_threshold} · Max flags: {exam.max_flags}
            </p>
          </div>

          <div className="bg-panel border border-line rounded-[6px] p-10 md:p-12">
            {!attempt && step === 'verify-id' && (
              <div className="max-w-[520px] mx-auto flex flex-col items-center gap-6">
                <p className="tl-hud-label !text-[13px]">Step 1 of 3 — Verify Your ID</p>
                <p className="text-[16px] text-bone-dim text-center leading-relaxed">
                  Every attempt starts with a fresh identity check — photograph or upload your
                  government ID, then confirm it's you with a quick live camera look.
                </p>
                <GovernmentIdCapture
                  busy={idBusy}
                  error={idError}
                  consented={idConsent}
                  onConsentChange={setIdConsent}
                  onFile={handleIdFile}
                />
              </div>
            )}

            {!attempt && step === 'verify-face' && (
              <div className="max-w-[520px] mx-auto flex flex-col items-center gap-6">
                <p className="tl-hud-label !text-[13px]">Step 2 of 3 — Confirm It's You</p>
                <p className="text-[16px] text-bone-dim text-center leading-relaxed">
                  Look at the camera so we can compare your live face against the ID you just showed.
                </p>
                {!verifyResult && !faceBusy && (
                  <WebcamCapture onCapture={handleFaceCapture} onCancel={handleRetryVerification} className="w-full" />
                )}
                {faceBusy && <p className="text-[16px] text-bone-dim">Checking confidence…</p>}
                {faceError && !faceBusy && (
                  <div className="flex flex-col items-center gap-4 text-center">
                    <p className="text-[16px] text-verdictDanger max-w-[400px]">{faceError}</p>
                    <button
                      type="button"
                      onClick={handleRetryVerification}
                      className="text-[14px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
                    >
                      Start over
                    </button>
                  </div>
                )}
                {verifyResult && !verifyResult.passed && !faceBusy && !faceError && (
                  <div className="flex flex-col items-center gap-4 text-center">
                    <p className="text-[16px] text-verdictDanger max-w-[400px]">
                      That's not a confident enough match ({Math.round(verifyResult.similarity * 100)}%
                      confidence). Make sure you're well-lit and facing the camera, then try again.
                    </p>
                    <button
                      type="button"
                      onClick={handleRetryVerification}
                      className="text-[14px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
                    >
                      Try again
                    </button>
                  </div>
                )}
              </div>
            )}

            {!attempt && step === 'terms' && (
              <div className="max-w-[580px] mx-auto">
                <p className="tl-hud-label !text-[13px] mb-2">Step 3 of 3 — Terms & Conditions</p>
                {verifyResult && (
                  <p className="text-[15px] text-verdictReal mb-6">
                    Identity confirmed — {Math.round(verifyResult.similarity * 100)}% confidence.
                  </p>
                )}
                <ul className="space-y-4 text-[16px] text-bone-dim leading-relaxed mb-8">
                  <li className="flex gap-3">
                    <ShieldAlert size={18} strokeWidth={1.5} className="text-brass shrink-0 mt-0.5" />
                    <span>Your camera and microphone stay on and are continuously monitored for the entire exam.</span>
                  </li>
                  <li className="flex gap-3">
                    <ShieldAlert size={18} strokeWidth={1.5} className="text-brass shrink-0 mt-0.5" />
                    <span>
                      Each check compares your face against the identity you just confirmed,
                      screens for signs of manipulation, and watches gaze, talking, and
                      phones/books in frame — every 20s normally, every 5s for a minute after
                      anything looks off.
                    </span>
                  </li>
                  <li className="flex gap-3">
                    <ShieldAlert size={18} strokeWidth={1.5} className="text-brass shrink-0 mt-0.5" />
                    <span>
                      Tab switches, window focus loss, clipboard pastes, and devtools use in this
                      browser tab are also logged. Activity outside this tab is not visible to it.
                    </span>
                  </li>
                  <li className="flex gap-3">
                    <ShieldCheck size={18} strokeWidth={1.5} className="text-verdictDanger shrink-0 mt-0.5" />
                    <span>
                      Being flagged {exam.max_flags} times automatically ends the exam and marks
                      the attempt failed — there is no warning before the final flag.
                    </span>
                  </li>
                  <li className="flex gap-3">
                    <ShieldCheck size={18} strokeWidth={1.5} className="text-brass shrink-0 mt-0.5" />
                    <span>
                      A session report is generated for your organisation to review — nothing here
                      is proof of misconduct on its own, it's a record for a human to judge.
                    </span>
                  </li>
                </ul>
                <div className="flex flex-col items-center gap-6">
                  <input
                    type="text"
                    value={rollNo}
                    onChange={(e) => setRollNo(e.target.value)}
                    placeholder="Roll number (optional)"
                    className="w-full max-w-[420px] bg-ground border border-line rounded-[3px] px-4 py-3.5 text-[16px] text-bone placeholder:text-bone-faint focus:outline-none focus:border-brass transition-colors duration-300"
                  />
                  <button
                    type="button"
                    disabled={starting}
                    onClick={handleStart}
                    className="group flex items-center gap-3 bg-brass text-ground px-8 py-4 rounded-[4px] text-[15px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-50"
                  >
                    {starting ? 'Starting…' : 'I Agree — Start Test'}
                    <span className="transition-transform duration-300 group-hover:translate-x-1"><ArrowRight /></span>
                  </button>
                  {startError && <p className="text-[15px] text-verdictDanger text-center">{startError}</p>}
                </div>
              </div>
            )}

            {attempt && attempt.status === 'in_progress' && (
              <div className="grid grid-cols-1 md:grid-cols-[1fr_300px] gap-8 items-start">
                <div className="order-2 md:order-1 min-w-0">
                  {questions === null && <p className="text-[15px] text-bone-dim text-center">Loading questions…</p>}

                  {questions && questions.map((q) => (
                    <QuestionCard
                      key={q.question_id}
                      question={q}
                      selected={answers[q.question_id]}
                      onSelect={handleSelect}
                      disabled={finishing}
                    />
                  ))}

                  {questions && questions.length > 0 && (
                    <div className="flex justify-center mt-8">
                      <button
                        type="button"
                        disabled={finishing}
                        onClick={handleFinish}
                        className="bg-brass text-ground px-8 py-4 rounded-[4px] text-[15px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-50"
                      >
                        {finishing ? 'Submitting…' : 'Submit Exam'}
                      </button>
                    </div>
                  )}

                  {error && <p className="text-[15px] text-verdictDanger text-center mt-4">{error}</p>}
                </div>

                {/* Side-docked camera widget -- sticky so it stays in view while scrolling through
                    questions. Monitoring keeps running (scheduleNextCheck inside
                    useMonitoringSession is unaffected by this box's size/position) regardless of
                    how large the preview is; sizing it is purely a layout choice. */}
                <div className="order-1 md:order-2 md:sticky md:top-24 flex flex-col items-center gap-3 mx-auto md:mx-0">
                  <div className="relative w-[260px] md:w-[300px] aspect-[4/3] bg-ground border border-line rounded-[4px] overflow-hidden">
                    <div className="relative w-full h-full scale-x-[-1]">
                      <video ref={videoRef} muted playsInline className="w-full h-full object-cover" />
                      <LiveFaceMesh videoRef={videoRef} active={!!session} onSustainedIssue={handleSustainedIssue} />
                    </div>
                    {!session && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-center px-4">
                        <p className="text-[13px] text-bone-faint">
                          {starting ? 'Connecting…' : 'Camera not connected'}
                        </p>
                        {!starting && (
                          <button
                            type="button"
                            onClick={() => resumeSession(attempt)}
                            className="text-[13px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
                          >
                            Resume Monitoring
                          </button>
                        )}
                      </div>
                    )}
                  </div>

                  {session && guidanceText && (
                    <div className="w-[260px] md:w-[300px] bg-[rgba(209,101,101,0.1)] border border-[rgba(209,101,101,0.3)] rounded-[4px] px-3 py-2.5">
                      <p className="text-[14px] text-verdictDanger leading-snug text-center">{guidanceText}</p>
                    </div>
                  )}

                  <div className="flex items-center justify-center gap-2 text-[14px] text-bone-dim text-center">
                    {flaggedCount > 0 ? (
                      <>
                        <ShieldAlert size={16} strokeWidth={1.5} className="text-verdictDanger shrink-0" />
                        <span>{attempt.flagged_checks} / {attempt.max_flags} flags</span>
                      </>
                    ) : (
                      <>
                        <ShieldCheck size={16} strokeWidth={1.5} className="text-verdictReal shrink-0" />
                        <span>0 / {attempt.max_flags} flags</span>
                      </>
                    )}
                  </div>
                  <p className="text-[12px] text-bone-faint text-center leading-relaxed">
                    Monitored continuously
                  </p>

                  {flagHistory.length > 0 && (
                    <div className="w-[260px] md:w-[300px] border-t border-line pt-3 mt-1">
                      <p className="text-[11px] font-medium tracking-[0.06em] uppercase text-bone-faint mb-2">
                        Flag history
                      </p>
                      <div className="flex flex-col gap-1.5 max-h-[220px] overflow-y-auto">
                        {flagHistory.map((item) => (
                          <div key={item.key} className="flex items-start gap-2 text-[12px] text-bone-dim">
                            <span className="w-1.5 h-1.5 rounded-full bg-verdictDanger shrink-0 mt-1.5" aria-hidden="true" />
                            <span className="leading-snug">{item.label}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {attempt && attempt.status !== 'in_progress' && (
              <div className="flex flex-col items-center gap-6 text-center">
                {attempt.status === 'terminated_for_flags' ? (
                  <>
                    <Lock size={28} strokeWidth={1.5} className="text-verdictDanger" />
                    <p className="font-serif text-[22px] text-bone">Attempt ended</p>
                    <p className="text-[15px] text-bone-dim max-w-[400px]">
                      You were flagged {attempt.max_flags} times and this attempt was
                      automatically ended and marked failed.
                    </p>
                  </>
                ) : (
                  <>
                    <ShieldCheck size={28} strokeWidth={1.5} className="text-brass" />
                    <p className="font-serif text-[22px] text-bone">Exam submitted</p>
                  </>
                )}
                <Link
                  to={`/exams/attempts/${attempt.attempt_id}/result`}
                  className="text-[14px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
                >
                  View Result
                </Link>
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
