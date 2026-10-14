import React, { useEffect, useRef, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ShieldAlert, ShieldCheck, Lock } from 'lucide-react'
import LiveFaceMesh from './components/LiveFaceMesh'
import useMonitoringSession from './hooks/useMonitoringSession'
import {
  finishExamAttempt,
  getExam,
  getExamAttempt,
  getExamQuestions,
  startExamAttempt,
  submitExamAnswer,
} from './api/client'

const ArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 8H13" /><path d="M9 4L13 8L9 12" />
  </svg>
)

function QuestionCard({ question, selected, onSelect, disabled }) {
  return (
    <div className="border-b border-line py-6 last:border-b-0">
      <p className="text-[14px] text-bone mb-4">
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
            className={`text-left px-4 py-3 rounded-[4px] border text-[13px] transition-colors duration-300 disabled:opacity-60 disabled:pointer-events-none ${
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
    videoRef, session, starting, error, stopSession, startSession, handleSustainedIssue, flaggedCount,
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

  if (exam === undefined) {
    return <div className="min-h-screen bg-ground pt-[72px] flex items-center justify-center text-bone-dim text-[13px]">Loading…</div>
  }
  if (exam === null) {
    return (
      <div className="min-h-screen bg-ground pt-[72px] flex items-center justify-center text-bone-dim text-[13px]">
        Exam not found.
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />
      <section className="relative min-h-[calc(100vh-72px)] flex flex-col items-center py-16 md:py-24 tl-inspection-grid">
        <div className="absolute inset-0 bg-gradient-to-b from-ground via-ground to-panel/40" />

        <div className="relative z-10 max-w-[840px] mx-auto px-6 w-full animate-fade-in-up animate-delay-1">
          <div className="text-center mb-12">
            <p className="tl-hud-label !text-brass mb-4">Proctored Exam</p>
            <h1 className="font-serif text-display-l text-bone mb-6">{exam.title}</h1>
            <p className="text-[14px] text-bone-dim">
              Pass threshold: {exam.pass_threshold} · Max flags: {exam.max_flags}
            </p>
          </div>

          <div className="bg-panel border border-line rounded-[6px] p-10 md:p-12">
            {!attempt && (
              <div className="max-w-[420px] mx-auto flex flex-col items-center gap-6">
                <p className="text-[13px] text-bone-dim text-center leading-relaxed">
                  Your camera stays on and continuously monitored for the whole exam. Being
                  flagged {exam.max_flags} times ends it automatically and marks it failed.
                </p>
                <input
                  type="text"
                  value={rollNo}
                  onChange={(e) => setRollNo(e.target.value)}
                  placeholder="Roll number (optional)"
                  className="w-full bg-ground border border-line rounded-[3px] px-4 py-3 text-[13px] text-bone placeholder:text-bone-faint focus:outline-none focus:border-brass transition-colors duration-300"
                />
                <button
                  type="button"
                  disabled={starting}
                  onClick={handleStart}
                  className="group flex items-center gap-3 bg-brass text-ground px-8 py-4 rounded-[4px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-50"
                >
                  {starting ? 'Starting…' : 'Start Exam'}
                  <span className="transition-transform duration-300 group-hover:translate-x-1"><ArrowRight /></span>
                </button>
                {startError && <p className="text-[13px] text-verdictDanger text-center">{startError}</p>}
              </div>
            )}

            {attempt && attempt.status === 'in_progress' && (
              <>
                <div className="relative w-full max-w-[480px] mx-auto aspect-[4/3] bg-ground border border-line rounded-[4px] overflow-hidden mb-6">
                  <div className="relative w-full h-full scale-x-[-1]">
                    <video ref={videoRef} muted playsInline className="w-full h-full object-cover" />
                    <LiveFaceMesh videoRef={videoRef} active={!!session} onSustainedIssue={handleSustainedIssue} />
                  </div>
                </div>

                <div className="flex items-center justify-center gap-2 mb-8 text-[12px] text-bone-dim">
                  {flaggedCount > 0 ? (
                    <>
                      <ShieldAlert size={14} strokeWidth={1.5} className="text-verdictDanger" />
                      <span>{attempt.flagged_checks} / {attempt.max_flags} flags</span>
                    </>
                  ) : (
                    <>
                      <ShieldCheck size={14} strokeWidth={1.5} className="text-verdictReal" />
                      <span>0 / {attempt.max_flags} flags</span>
                    </>
                  )}
                </div>

                {questions === null && <p className="text-[13px] text-bone-dim text-center">Loading questions…</p>}

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
                      className="bg-brass text-ground px-8 py-4 rounded-[4px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-50"
                    >
                      {finishing ? 'Submitting…' : 'Submit Exam'}
                    </button>
                  </div>
                )}

                {error && <p className="text-[13px] text-verdictDanger text-center mt-4">{error}</p>}
              </>
            )}

            {attempt && attempt.status !== 'in_progress' && (
              <div className="flex flex-col items-center gap-6 text-center">
                {attempt.status === 'terminated_for_flags' ? (
                  <>
                    <Lock size={28} strokeWidth={1.5} className="text-verdictDanger" />
                    <p className="font-serif text-[20px] text-bone">Attempt ended</p>
                    <p className="text-[13px] text-bone-dim max-w-[360px]">
                      You were flagged {attempt.max_flags} times and this attempt was
                      automatically ended and marked failed.
                    </p>
                  </>
                ) : (
                  <>
                    <ShieldCheck size={28} strokeWidth={1.5} className="text-brass" />
                    <p className="font-serif text-[20px] text-bone">Exam submitted</p>
                  </>
                )}
                <Link
                  to={`/exams/attempts/${attempt.attempt_id}/result`}
                  className="text-[12px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
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
