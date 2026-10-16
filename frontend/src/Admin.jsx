import React, { useEffect, useState } from 'react'
import { AlertCircle, Check, ChevronDown, ChevronUp, Copy, Download, Plus, RefreshCw, SearchX, ShieldAlert, Trash2, X } from 'lucide-react'
import {
  addExamQuestion,
  createExam,
  deleteExamQuestion,
  downloadAdminSessionReport,
  getAdminSessionDetail,
  getAdminSessions,
  getExamAttemptsAdmin,
  getExamQuestionsAdmin,
  getMyExams,
  getOrganizationJoinCode,
  getRoster,
  regenerateOrganizationJoinCode,
  removeFromRoster,
} from './api/client'
import LoadingState from './components/LoadingState'

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

function formatJoinCode(code) {
  if (!code || code.length !== 8) return code
  return `${code.slice(0, 4)}-${code.slice(4)}`
}

function JoinCodePanel() {
  const [code, setCode] = useState(undefined)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)
  const [confirmingRegenerate, setConfirmingRegenerate] = useState(false)
  const [regenerating, setRegenerating] = useState(false)

  useEffect(() => {
    let cancelled = false
    getOrganizationJoinCode()
      .then((r) => { if (!cancelled) setCode(r.join_code) })
      .catch((err) => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(formatJoinCode(code))
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      // Clipboard access can be denied/unavailable -- the code is still visible on screen either way.
    }
  }

  const regenerate = async () => {
    setRegenerating(true)
    setError(null)
    try {
      const r = await regenerateOrganizationJoinCode()
      setCode(r.join_code)
      setConfirmingRegenerate(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setRegenerating(false)
    }
  }

  return (
    <div className="mt-8 flex flex-col gap-2 w-fit">
      <p className="tl-hud-label !text-[9px]">Your Join Code</p>
      {error && <p className="text-[12px] text-verdictDanger">{error}</p>}
      {code === undefined && !error && <p className="text-[13px] text-bone-dim">Loading…</p>}
      {code && (
        <>
          <div className="flex items-center gap-3">
            <span className="tl-figure text-[18px] text-bone tracking-[0.12em]">{formatJoinCode(code)}</span>
            <button
              type="button"
              onClick={copy}
              aria-label="Copy join code"
              className="text-bone-faint hover:text-brass transition-colors duration-300"
            >
              {copied ? <Check size={15} strokeWidth={1.75} className="text-verdictReal" /> : <Copy size={15} strokeWidth={1.75} />}
            </button>
          </div>
          <p className="text-[12px] text-bone-dim max-w-[360px]">
            Share this with your students — they enter it once on the ProctorLens page to see
            the exams you've created.
          </p>
          {!confirmingRegenerate ? (
            <button
              type="button"
              onClick={() => setConfirmingRegenerate(true)}
              className="flex items-center gap-1.5 text-[11px] font-medium tracking-[0.06em] text-bone-faint hover:text-bone w-fit mt-1"
            >
              <RefreshCw size={12} strokeWidth={1.75} /> Regenerate
            </button>
          ) : (
            <div className="flex items-center gap-3 text-[12px] mt-1">
              <span className="text-bone-dim">This won't affect students already linked. Confirm?</span>
              <button type="button" onClick={regenerate} disabled={regenerating} className="text-brass hover:text-bone link-underline">
                {regenerating ? 'Regenerating…' : 'Yes'}
              </button>
              <button type="button" onClick={() => setConfirmingRegenerate(false)} className="text-bone-faint hover:text-bone link-underline">
                Cancel
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function Row({ time, flagged, labels }) {
  return (
    <div className="flex items-start gap-3 py-2.5 border-b border-line last:border-b-0">
      <div className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${flagged ? 'bg-verdictDanger' : 'bg-verdictReal'}`} aria-hidden="true" />
      <div className="flex-1">
        <span className="tl-figure text-[12px] text-bone-dim">{time}</span>
        {labels.length > 0 ? (
          <div className="flex flex-wrap gap-2 mt-1">
            {labels.map((l) => (
              <span key={l} className="text-[10px] font-medium tracking-[0.06em] uppercase text-verdictDanger bg-[rgba(209,101,101,0.1)] border border-[rgba(209,101,101,0.3)] rounded-[3px] px-2 py-0.5">
                {l}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-[12px] text-bone-dim">Normal</p>
        )}
      </div>
    </div>
  )
}

function SessionDetail({ sessionId, onClose }) {
  const [detail, setDetail] = useState(undefined)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    getAdminSessionDetail(sessionId)
      .then((d) => { if (!cancelled) setDetail(d) })
      .catch((err) => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [sessionId])

  const timeline = detail
    ? [
        ...detail.checks.map((c) => ({
          key: c.check_id, at: c.checked_at, flagged: c.flagged,
          labels: c.flag_reasons.map((r) => REASON_LABELS[r] || r),
        })),
        ...detail.events.map((e) => ({
          key: e.event_id, at: e.occurred_at, flagged: true,
          labels: [EVENT_LABELS[e.event_type] || e.event_type],
        })),
      ].sort((a, b) => (a.at > b.at ? 1 : -1))
    : []

  return (
    <div className="bg-panel border border-line rounded-[6px] p-6 mt-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="tl-hud-label !text-[9px]">{detail ? `${detail.user_name} · ${detail.user_email}` : 'Loading…'}</p>
          <p className="text-[13px] text-bone-dim mt-1">Session {sessionId}</p>
        </div>
        <div className="flex items-center gap-4">
          <button
            type="button"
            onClick={() => downloadAdminSessionReport(sessionId).catch((err) => setError(err.message))}
            className="flex items-center gap-2 text-[12px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
          >
            <Download size={14} strokeWidth={1.75} /> Download Report
          </button>
          <button type="button" onClick={onClose} className="text-[12px] text-bone-faint hover:text-bone">
            Close
          </button>
        </div>
      </div>
      {error && <p className="text-[13px] text-verdictDanger mb-3">{error}</p>}
      {detail === undefined && !error && <p className="text-[13px] text-bone-dim">Loading…</p>}
      {detail && timeline.length === 0 && <p className="text-[13px] text-bone-dim">No activity recorded yet.</p>}
      {timeline.map((row) => (
        <Row
          key={row.key}
          time={new Date(row.at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
          flagged={row.flagged}
          labels={row.labels}
        />
      ))}
    </div>
  )
}

function RosterPanel() {
  const [students, setStudents] = useState(undefined)
  const [error, setError] = useState(null)
  const [confirmingRemove, setConfirmingRemove] = useState(null) // user_id currently confirming, or null
  const [removing, setRemoving] = useState(false)

  const load = () => {
    getRoster()
      .then((rows) => setStudents(rows))
      .catch((err) => setError(err.message))
  }

  useEffect(() => { load() }, [])

  const confirmRemove = async (userId) => {
    setRemoving(true)
    try {
      await removeFromRoster(userId)
      setStudents((prev) => prev.filter((s) => s.user_id !== userId))
      setConfirmingRemove(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setRemoving(false)
    }
  }

  return (
    <div className="mt-8">
      <p className="tl-hud-label !text-[9px] mb-3">Your Roster</p>
      {error && <p className="text-[12px] text-verdictDanger mb-2">{error}</p>}
      {students === undefined && !error && <p className="text-[13px] text-bone-dim">Loading…</p>}
      {students && students.length === 0 && (
        <p className="text-[12px] text-bone-dim">No students linked yet.</p>
      )}
      {students && students.length > 0 && (
        <div className="flex flex-col gap-2 max-w-[420px]">
          {students.map((s) => (
            <div key={s.user_id} className="flex items-center justify-between gap-3 py-1.5 border-b border-line last:border-b-0">
              <div className="min-w-0">
                <p className="text-[13px] text-bone truncate">{s.name}</p>
                <p className="text-[11px] text-bone-faint truncate">{s.email}</p>
              </div>
              {confirmingRemove === s.user_id ? (
                <div className="flex items-center gap-2 shrink-0 text-[11px]">
                  <button type="button" onClick={() => confirmRemove(s.user_id)} disabled={removing} className="text-verdictDanger hover:text-bone link-underline">
                    {removing ? 'Removing…' : 'Confirm'}
                  </button>
                  <button type="button" onClick={() => setConfirmingRemove(null)} className="text-bone-faint hover:text-bone">
                    <X size={13} strokeWidth={1.75} />
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => setConfirmingRemove(s.user_id)}
                  className="text-[11px] text-bone-faint hover:text-verdictDanger shrink-0"
                >
                  Remove
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const STATUS_LABELS = {
  in_progress: 'In progress',
  passed: 'Passed',
  failed: 'Failed',
  terminated_for_flags: 'Ended — flag limit',
}

function QuestionEditor({ examId }) {
  const [questions, setQuestions] = useState(undefined)
  const [error, setError] = useState(null)
  const [adding, setAdding] = useState(false)
  const [text, setText] = useState('')
  const [options, setOptions] = useState(['', ''])
  const [correctIndex, setCorrectIndex] = useState(0)
  const [saving, setSaving] = useState(false)
  const [confirmingDelete, setConfirmingDelete] = useState(null) // question_id

  const load = () => {
    getExamQuestionsAdmin(examId)
      .then((rows) => setQuestions(rows))
      .catch((err) => setError(err.message))
  }

  useEffect(() => { load() }, [examId])

  const resetForm = () => {
    setText('')
    setOptions(['', ''])
    setCorrectIndex(0)
    setAdding(false)
  }

  const handleAddOption = () => setOptions((prev) => [...prev, ''])
  const handleRemoveOption = (i) => {
    setOptions((prev) => prev.filter((_, idx) => idx !== i))
    if (correctIndex >= options.length - 1) setCorrectIndex(0)
  }
  const handleOptionChange = (i, value) => {
    setOptions((prev) => prev.map((o, idx) => (idx === i ? value : o)))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSaving(true)
    setError(null)
    try {
      const cleanOptions = options.map((o) => o.trim())
      await addExamQuestion(examId, text.trim(), cleanOptions, correctIndex)
      resetForm()
      load()
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  const handleDelete = async (questionId) => {
    setSaving(true)
    try {
      await deleteExamQuestion(examId, questionId)
      setQuestions((prev) => prev.filter((q) => q.question_id !== questionId))
      setConfirmingDelete(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mt-4">
      <div className="flex items-center justify-between mb-3">
        <p className="tl-hud-label !text-[9px]">Questions</p>
        {!adding && (
          <button
            type="button"
            onClick={() => setAdding(true)}
            className="flex items-center gap-1.5 text-[11px] font-medium tracking-[0.06em] text-brass hover:text-bone"
          >
            <Plus size={12} strokeWidth={1.75} /> Add Question
          </button>
        )}
      </div>

      {error && <p className="text-[12px] text-verdictDanger mb-2">{error}</p>}
      {questions === undefined && !error && <p className="text-[13px] text-bone-dim">Loading…</p>}
      {questions && questions.length === 0 && !adding && (
        <p className="text-[12px] text-bone-dim">No questions yet.</p>
      )}

      {questions && questions.length > 0 && (
        <div className="flex flex-col gap-2 mb-4">
          {questions.map((q, i) => (
            <div key={q.question_id} className="border border-line rounded-[4px] p-3">
              <div className="flex items-start justify-between gap-3">
                <p className="text-[13px] text-bone">
                  <span className="text-bone-faint mr-1.5">{i + 1}.</span>{q.text}
                </p>
                {confirmingDelete === q.question_id ? (
                  <div className="flex items-center gap-2 shrink-0 text-[11px]">
                    <button type="button" onClick={() => handleDelete(q.question_id)} disabled={saving} className="text-verdictDanger hover:text-bone link-underline">
                      Confirm
                    </button>
                    <button type="button" onClick={() => setConfirmingDelete(null)} className="text-bone-faint hover:text-bone">
                      <X size={12} strokeWidth={1.75} />
                    </button>
                  </div>
                ) : (
                  <button type="button" onClick={() => setConfirmingDelete(q.question_id)} className="text-bone-faint hover:text-verdictDanger shrink-0">
                    <Trash2 size={13} strokeWidth={1.75} />
                  </button>
                )}
              </div>
              <div className="flex flex-col gap-1 mt-2">
                {q.options.map((opt, idx) => (
                  <p key={idx} className={`text-[12px] ${idx === q.correct_option_index ? 'text-verdictReal' : 'text-bone-dim'}`}>
                    {idx === q.correct_option_index && <Check size={11} strokeWidth={2} className="inline mr-1 -mt-0.5" />}
                    {opt}
                  </p>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {adding && (
        <form onSubmit={handleSubmit} className="border border-line rounded-[4px] p-4 flex flex-col gap-3">
          <input
            type="text"
            required
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Question text"
            className="bg-ground border border-line rounded-[3px] px-3 py-2 text-[13px] text-bone placeholder:text-bone-faint focus:outline-none focus:border-brass transition-colors duration-300"
          />
          <div className="flex flex-col gap-2">
            {options.map((opt, i) => (
              <div key={i} className="flex items-center gap-2">
                <input
                  type="radio"
                  name="correct-option"
                  checked={correctIndex === i}
                  onChange={() => setCorrectIndex(i)}
                  className="accent-brass"
                  aria-label={`Option ${i + 1} is correct`}
                />
                <input
                  type="text"
                  required
                  value={opt}
                  onChange={(e) => handleOptionChange(i, e.target.value)}
                  placeholder={`Option ${i + 1}`}
                  className="flex-1 bg-ground border border-line rounded-[3px] px-3 py-1.5 text-[13px] text-bone placeholder:text-bone-faint focus:outline-none focus:border-brass transition-colors duration-300"
                />
                {options.length > 2 && (
                  <button type="button" onClick={() => handleRemoveOption(i)} className="text-bone-faint hover:text-verdictDanger">
                    <X size={13} strokeWidth={1.75} />
                  </button>
                )}
              </div>
            ))}
          </div>
          <button type="button" onClick={handleAddOption} className="flex items-center gap-1.5 text-[11px] text-bone-faint hover:text-bone w-fit">
            <Plus size={11} strokeWidth={1.75} /> Add Option
          </button>
          <div className="flex items-center gap-4 mt-1">
            <button
              type="submit"
              disabled={saving}
              className="bg-brass text-ground px-4 py-2 rounded-[3px] text-[12px] font-medium tracking-[0.06em] uppercase btn-lift disabled:opacity-50"
            >
              {saving ? 'Saving…' : 'Save Question'}
            </button>
            <button type="button" onClick={resetForm} className="text-[12px] text-bone-faint hover:text-bone">
              Cancel
            </button>
          </div>
        </form>
      )}
    </div>
  )
}

function AttemptsTable({ examId }) {
  const [attempts, setAttempts] = useState(undefined)
  const [error, setError] = useState(null)
  const [openAttemptId, setOpenAttemptId] = useState(null)

  useEffect(() => {
    getExamAttemptsAdmin(examId)
      .then((rows) => setAttempts(rows))
      .catch((err) => setError(err.message))
  }, [examId])

  return (
    <div className="mt-6">
      <p className="tl-hud-label !text-[9px] mb-3">Attempts</p>
      {error && <p className="text-[12px] text-verdictDanger">{error}</p>}
      {attempts === undefined && !error && <p className="text-[13px] text-bone-dim">Loading…</p>}
      {attempts && attempts.length === 0 && <p className="text-[12px] text-bone-dim">No attempts yet.</p>}
      {attempts && attempts.length > 0 && (
        <div className="flex flex-col">
          {attempts.map((a) => (
            <div key={a.attempt_id}>
              <div
                role="button"
                tabIndex={0}
                onClick={() => setOpenAttemptId(openAttemptId === a.attempt_id ? null : a.attempt_id)}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpenAttemptId(openAttemptId === a.attempt_id ? null : a.attempt_id) } }}
                className="flex items-center gap-4 py-3 border-b border-line cursor-pointer hover:bg-panel/50 transition-colors duration-300 -mx-3 px-3"
              >
                {openAttemptId === a.attempt_id ? <ChevronUp size={14} strokeWidth={1.75} className="text-bone-faint shrink-0" /> : <ChevronDown size={14} strokeWidth={1.75} className="text-bone-faint shrink-0" />}
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] text-bone truncate">{a.user_name}</p>
                  <p className="text-[11px] text-bone-faint truncate">{a.user_email}</p>
                </div>
                {a.score != null && <span className="tl-figure text-[12px] text-bone-dim">{a.score}</span>}
                <span className={`text-[11px] font-medium tracking-[0.06em] uppercase ${
                  a.status === 'passed' ? 'text-verdictReal' : a.status === 'in_progress' ? 'text-bone-faint' : 'text-verdictDanger'
                }`}>
                  {STATUS_LABELS[a.status] || a.status}
                </span>
              </div>
              {openAttemptId === a.attempt_id && (
                <SessionDetail sessionId={a.session_id} onClose={() => setOpenAttemptId(null)} />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ExamRow({ exam }) {
  const [open, setOpen] = useState(false)

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen((v) => !v) } }}
        className="flex items-center gap-4 py-4 border-b border-line cursor-pointer hover:bg-panel/50 transition-colors duration-300 -mx-4 px-4"
      >
        {open ? <ChevronUp size={14} strokeWidth={1.75} className="text-bone-faint shrink-0" /> : <ChevronDown size={14} strokeWidth={1.75} className="text-bone-faint shrink-0" />}
        <div className="min-w-0 flex-1">
          <p className="font-serif text-[16px] text-bone truncate">{exam.title}</p>
          <p className="tl-figure text-[10px] text-bone-faint mt-0.5">
            Min. correct to pass: {exam.pass_threshold} · Max flags {exam.max_flags} · {exam.question_count} question{exam.question_count === 1 ? '' : 's'}
          </p>
        </div>
      </div>
      {open && (
        <div className="bg-panel border border-line rounded-[6px] p-6 mt-4 mb-2">
          <QuestionEditor examId={exam.exam_id} />
          <AttemptsTable examId={exam.exam_id} />
        </div>
      )}
    </div>
  )
}

function ExamsPanel() {
  const [exams, setExams] = useState(undefined)
  const [error, setError] = useState(null)
  const [creating, setCreating] = useState(false)
  const [title, setTitle] = useState('')
  const [passThreshold, setPassThreshold] = useState(1)
  const [maxFlags, setMaxFlags] = useState(4)
  const [saving, setSaving] = useState(false)

  const load = () => {
    getMyExams()
      .then((rows) => setExams(rows))
      .catch((err) => setError(err.message))
  }

  useEffect(() => { load() }, [])

  const handleCreate = async (e) => {
    e.preventDefault()
    setSaving(true)
    setError(null)
    try {
      await createExam(title.trim(), Number(passThreshold), Number(maxFlags))
      setTitle('')
      setPassThreshold(60)
      setMaxFlags(4)
      setCreating(false)
      load()
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mt-8">
      <div className="flex items-center justify-between mb-3">
        <p className="tl-hud-label !text-[9px]">ProctorLens Exams</p>
        {!creating && (
          <button
            type="button"
            onClick={() => setCreating(true)}
            className="flex items-center gap-1.5 text-[11px] font-medium tracking-[0.06em] text-brass hover:text-bone"
          >
            <Plus size={12} strokeWidth={1.75} /> New Exam
          </button>
        )}
      </div>

      {error && <p className="text-[12px] text-verdictDanger mb-2">{error}</p>}

      {creating && (
        <form onSubmit={handleCreate} className="border border-line rounded-[4px] p-4 flex flex-col gap-3 max-w-[480px] mb-4">
          <input
            type="text"
            required
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Exam title"
            className="bg-ground border border-line rounded-[3px] px-3 py-2 text-[13px] text-bone placeholder:text-bone-faint focus:outline-none focus:border-brass transition-colors duration-300"
          />
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-[12px] text-bone-dim">
              Min. correct to pass
              <input
                type="number"
                min={0}
                required
                value={passThreshold}
                onChange={(e) => setPassThreshold(e.target.value)}
                className="w-20 bg-ground border border-line rounded-[3px] px-2 py-1.5 text-[13px] text-bone focus:outline-none focus:border-brass transition-colors duration-300"
              />
            </label>
            <label className="flex items-center gap-2 text-[12px] text-bone-dim">
              Max flags
              <input
                type="number"
                min={1}
                required
                value={maxFlags}
                onChange={(e) => setMaxFlags(e.target.value)}
                className="w-20 bg-ground border border-line rounded-[3px] px-2 py-1.5 text-[13px] text-bone focus:outline-none focus:border-brass transition-colors duration-300"
              />
            </label>
          </div>
          <div className="flex items-center gap-4 mt-1">
            <button
              type="submit"
              disabled={saving}
              className="bg-brass text-ground px-4 py-2 rounded-[3px] text-[12px] font-medium tracking-[0.06em] uppercase btn-lift disabled:opacity-50"
            >
              {saving ? 'Creating…' : 'Create Exam'}
            </button>
            <button type="button" onClick={() => setCreating(false)} className="text-[12px] text-bone-faint hover:text-bone">
              Cancel
            </button>
          </div>
        </form>
      )}

      {exams === undefined && !error && <p className="text-[13px] text-bone-dim">Loading…</p>}
      {exams && exams.length === 0 && !creating && (
        <p className="text-[12px] text-bone-dim">No exams yet.</p>
      )}
      {exams && exams.length > 0 && (
        <div>
          {exams.map((exam) => <ExamRow key={exam.exam_id} exam={exam} />)}
        </div>
      )}
    </div>
  )
}

function sessionsToCsv(sessions) {
  const header = ['session_id', 'student_name', 'student_email', 'started_at', 'ended_at', 'total_checks', 'flagged_checks']
  const escape = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`
  const rows = sessions.map((s) => [
    s.session_id, s.user_name, s.user_email, s.started_at, s.ended_at || '', s.total_checks, s.flagged_checks,
  ].map(escape).join(','))
  return [header.join(','), ...rows].join('\n')
}

function downloadSessionsCsv(sessions) {
  const csv = sessionsToCsv(sessions)
  const blob = new Blob([csv], { type: 'text/csv' })
  const blobUrl = window.URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = blobUrl
  link.download = `truthlens-sessions-${new Date().toISOString().slice(0, 10)}.csv`
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.URL.revokeObjectURL(blobUrl)
}

const TABS = [
  { id: 'setup', label: 'Setup' },
  { id: 'exams', label: 'Exams' },
  { id: 'activity', label: 'Activity' },
]

function TabBar({ active, onChange }) {
  return (
    <div className="flex items-center gap-1 mt-8 border-b border-line w-fit">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onChange(tab.id)}
          className={`px-5 py-3 text-[13px] font-medium tracking-[0.06em] uppercase transition-colors duration-300 border-b-2 -mb-px ${
            active === tab.id
              ? 'text-brass border-brass'
              : 'text-bone-faint border-transparent hover:text-bone'
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

export default function Admin() {
  const [activeTab, setActiveTab] = useState('setup')
  const [sessions, setSessions] = useState(undefined)
  const [error, setError] = useState(null)
  const [flaggedOnly, setFlaggedOnly] = useState(false)
  const [openSessionId, setOpenSessionId] = useState(null)

  useEffect(() => {
    if (activeTab !== 'activity') return
    let cancelled = false
    setSessions(undefined)
    getAdminSessions({ flaggedOnly })
      .then((rows) => { if (!cancelled) setSessions(rows) })
      .catch((err) => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [activeTab, flaggedOnly])

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />
      <section className="pt-16 md:pt-24 pb-12 border-b border-line">
        <div className="max-w-[1100px] mx-auto px-6 md:px-12 lg:px-16">
          <p className="tl-hud-label mb-4">Organisation Dashboard</p>
          <h1 className="font-serif text-display-l text-bone">
            Your students,<br /><span className="italic text-brass">under watch.</span>
          </h1>
          <TabBar active={activeTab} onChange={setActiveTab} />
        </div>
      </section>

      {activeTab === 'setup' && (
        <section className="pt-12 pb-24 md:pb-32">
          <div className="max-w-[1100px] mx-auto px-6 md:px-12 lg:px-16">
            <JoinCodePanel />
            <RosterPanel />
          </div>
        </section>
      )}

      {activeTab === 'exams' && (
        <section className="pt-12 pb-24 md:pb-32">
          <div className="max-w-[1100px] mx-auto px-6 md:px-12 lg:px-16">
            <ExamsPanel />
          </div>
        </section>
      )}

      {activeTab === 'activity' && (
      <section className="pt-12 pb-24 md:pb-32">
        <div className="max-w-[1100px] mx-auto px-6 md:px-12 lg:px-16">
          <label className="flex items-center gap-2 mb-6 cursor-pointer select-none w-fit">
            <input
              type="checkbox"
              checked={flaggedOnly}
              onChange={(e) => setFlaggedOnly(e.target.checked)}
              className="accent-brass"
            />
            <span className="text-[13px] text-bone-dim">Flagged sessions only</span>
          </label>
          {sessions === undefined && !error && <LoadingState label="Loading sessions" />}

          {error && (
            <div className="flex flex-col items-center text-center py-24">
              <AlertCircle size={32} strokeWidth={1.25} className="text-verdictDanger mb-4" />
              <p className="font-serif text-[20px] text-bone mb-2">Unable to load sessions</p>
              <p className="text-[14px] text-verdictDanger">{error}</p>
            </div>
          )}

          {sessions && sessions.length === 0 && (
            <div className="flex flex-col items-center text-center py-24">
              <SearchX size={32} strokeWidth={1.25} className="text-bone-faint mb-5" />
              <p className="font-serif text-[24px] text-bone mb-3">No sessions found</p>
            </div>
          )}

          {sessions && sessions.length > 0 && (
            <div>
              <div className="flex justify-end mb-4">
                <button
                  type="button"
                  onClick={() => downloadSessionsCsv(sessions)}
                  className="flex items-center gap-1.5 text-[12px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
                >
                  <Download size={13} strokeWidth={1.75} /> Export CSV
                </button>
              </div>
              {sessions.map((s) => (
                <div key={s.session_id}>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => setOpenSessionId(openSessionId === s.session_id ? null : s.session_id)}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpenSessionId(openSessionId === s.session_id ? null : s.session_id) } }}
                    className="flex items-center gap-4 md:gap-6 py-4 border-b border-line cursor-pointer hover:bg-panel/50 transition-colors duration-300 -mx-4 px-4"
                  >
                    <div className="min-w-0 flex-1">
                      <p className="font-serif text-[16px] text-bone truncate">{s.user_name}</p>
                      <p className="tl-figure text-[10px] text-bone-faint mt-0.5">
                        {s.user_email} · {new Date(s.started_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                        {!s.ended_at && ' · active'}
                      </p>
                    </div>
                    <span className="tl-figure text-[12px] text-bone-dim hidden sm:block">
                      {s.total_checks} check{s.total_checks === 1 ? '' : 's'}
                    </span>
                    {s.flagged_checks > 0 ? (
                      <span className="flex items-center gap-1.5 text-[11px] font-medium tracking-[0.06em] uppercase text-verdictDanger">
                        <ShieldAlert size={14} strokeWidth={1.75} /> {s.flagged_checks} flagged
                      </span>
                    ) : (
                      <span className="text-[11px] text-bone-faint">Clean</span>
                    )}
                  </div>
                  {openSessionId === s.session_id && (
                    <SessionDetail sessionId={s.session_id} onClose={() => setOpenSessionId(null)} />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
      )}
    </div>
  )
}
