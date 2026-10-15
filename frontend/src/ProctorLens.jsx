import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Lock, FileText } from 'lucide-react'
import { listMyExams } from './api/client'

const ArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 8H13" /><path d="M9 4L13 8L9 12" />
  </svg>
)

const STATUS_LABEL = {
  passed: 'Passed',
  failed: 'Failed',
  terminated_for_flags: 'Ended — too many flags',
  in_progress: 'In progress',
}

function ExamCard({ exam }) {
  const status = exam.my_attempt_status

  return (
    <div className="bg-panel border border-line rounded-[6px] p-6 flex items-center justify-between gap-6">
      <div className="min-w-0">
        <p className="font-serif text-[19px] text-bone truncate">{exam.title}</p>
        <p className="text-[14px] text-bone-dim mt-1">
          Min. correct to pass: {exam.pass_threshold} · Max flags {exam.max_flags} ·{' '}
          {exam.question_count} question{exam.question_count === 1 ? '' : 's'}
        </p>
        {status && (
          <p className="text-[13px] font-medium tracking-[0.06em] uppercase text-bone-faint mt-2">
            {STATUS_LABEL[status] || status}
          </p>
        )}
      </div>

      {!status && (
        <Link
          to={`/exams/${exam.exam_id}/take`}
          className="shrink-0 group flex items-center gap-2 bg-brass text-ground px-5 py-3.5 rounded-[4px] text-[14px] font-medium tracking-[0.08em] uppercase btn-lift"
        >
          Start
          <span className="transition-transform duration-300 group-hover:translate-x-1"><ArrowRight /></span>
        </Link>
      )}

      {status === 'in_progress' && (
        <Link
          to={`/exams/${exam.exam_id}/take`}
          className="shrink-0 text-[14px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
        >
          Continue
        </Link>
      )}

      {(status === 'passed' || status === 'failed' || status === 'terminated_for_flags') && (
        <Link
          to={`/exams/attempts/${exam.my_attempt_id}/result`}
          className="shrink-0 text-[14px] font-medium tracking-[0.06em] text-brass hover:text-bone link-underline"
        >
          View Result
        </Link>
      )}
    </div>
  )
}

export default function ProctorLens() {
  const [exams, setExams] = useState(undefined) // undefined=loading, array=loaded
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    listMyExams()
      .then((e) => { if (!cancelled) setExams(e) })
      .catch((err) => { if (!cancelled) { setExams([]); setError(err.message) } })
    return () => { cancelled = true }
  }, [])

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />
      <section className="relative min-h-[calc(100vh-72px)] flex flex-col items-center py-16 md:py-24 tl-inspection-grid">
        <div className="absolute inset-0 bg-gradient-to-b from-ground via-ground to-panel/40" />

        <div className="relative z-10 max-w-[720px] mx-auto px-6 w-full animate-fade-in-up animate-delay-1">
          <div className="text-center mb-12">
            <p className="tl-hud-label !text-brass mb-4">ProctorLens</p>
            <h1 className="font-serif text-display-l text-bone mb-6">
              Verified identity. <span className="italic text-brass">Watched exams.</span>
            </h1>
            <p className="text-[16px] text-bone-dim max-w-[520px] mx-auto">
              Pick an exam below. Starting one walks you through a fresh ID check, then the test
              itself — questions and continuous camera monitoring running together, side by side,
              start to finish.
            </p>
          </div>

          {exams === undefined && <p className="text-[15px] text-bone-dim text-center">Loading…</p>}

          {exams && exams.length === 0 && (
            <div className="bg-panel border border-line rounded-[6px] p-10 text-center">
              <FileText size={24} strokeWidth={1.5} className="text-bone-faint mx-auto mb-4" />
              <p className="text-[15px] text-bone-dim">
                {error
                  ? error
                  : "No exams available yet. If your organisation hasn't shared a join code with you, "}
                {!error && (
                  <Link to="/identity" className="text-brass hover:text-bone link-underline">
                    join one from the Identity page
                  </Link>
                )}
                {!error && '.'}
              </p>
            </div>
          )}

          {exams && exams.length > 0 && (
            <div className="flex flex-col gap-4">
              {exams.map((exam) => <ExamCard key={exam.exam_id} exam={exam} />)}
            </div>
          )}

          {exams && exams.length > 0 && (
            <p className="text-[13px] text-bone-faint text-center mt-8 flex items-center justify-center gap-2">
              <Lock size={14} strokeWidth={1.5} />
              Every attempt is camera-monitored start to finish — identity drift, a second face, a
              phone in frame, or leaving the tab all land in the report your organisation reviews.
            </p>
          )}
        </div>
      </section>
    </div>
  )
}
