import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Download, ShieldCheck, ShieldAlert, Lock } from 'lucide-react'
import { downloadSessionReport, getExamAttempt } from './api/client'

const STATUS_INFO = {
  passed: { label: 'Passed', Icon: ShieldCheck, className: 'text-verdictReal' },
  failed: { label: 'Failed', Icon: ShieldAlert, className: 'text-verdictDanger' },
  terminated_for_flags: { label: 'Ended — Flag limit exceeded', Icon: Lock, className: 'text-verdictDanger' },
  in_progress: { label: 'In progress', Icon: ShieldAlert, className: 'text-bone-dim' },
}

export default function ExamResult() {
  const { attemptId } = useParams()
  const [attempt, setAttempt] = useState(undefined)
  const [error, setError] = useState(null)
  const [downloading, setDownloading] = useState(false)

  useEffect(() => {
    let cancelled = false
    getExamAttempt(attemptId)
      .then((a) => { if (!cancelled) setAttempt(a) })
      .catch((err) => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [attemptId])

  const handleDownload = async () => {
    if (!attempt) return
    setDownloading(true)
    try {
      await downloadSessionReport(attempt.session_id)
    } catch (err) {
      setError(err.message)
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />
      <section className="relative min-h-[calc(100vh-72px)] flex flex-col items-center justify-center py-16 md:py-24 tl-inspection-grid">
        <div className="absolute inset-0 bg-gradient-to-b from-ground via-ground to-panel/40" />

        <div className="relative z-10 max-w-[480px] mx-auto px-6 w-full animate-fade-in-up animate-delay-1">
          <div className="bg-panel border border-line rounded-[6px] p-10 md:p-12 flex flex-col items-center text-center gap-6">
            {attempt === undefined && !error && <p className="text-[13px] text-bone-dim">Loading…</p>}
            {error && <p className="text-[14px] text-verdictDanger">{error}</p>}

            {attempt && (
              <>
                <p className="tl-hud-label !text-brass">ProctorLens</p>
                {(() => {
                  const info = STATUS_INFO[attempt.status] || STATUS_INFO.in_progress
                  const Icon = info.Icon
                  return (
                    <>
                      <Icon size={32} strokeWidth={1.5} className={info.className} />
                      <p className={`font-serif text-[24px] ${info.className}`}>{info.label}</p>
                    </>
                  )
                })()}
                {attempt.score != null && (
                  <p className="text-[14px] text-bone-dim">Score: {attempt.score}</p>
                )}
                <p className="text-[12px] text-bone-faint max-w-[360px]">
                  This result reflects your answers and the proctoring session's flag count —
                  nothing here is proof of misconduct on its own, it's a record for your
                  organisation to review.
                </p>
                <button
                  type="button"
                  disabled={downloading}
                  onClick={handleDownload}
                  className="flex items-center gap-2 bg-brass text-ground px-6 py-3 rounded-[4px] text-[12px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-50"
                >
                  <Download size={14} strokeWidth={1.75} /> {downloading ? 'Downloading…' : 'Download Session Report'}
                </button>
                <Link to="/proctorlens" className="text-[12px] text-bone-faint hover:text-bone link-underline">
                  Back to ProctorLens
                </Link>
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
