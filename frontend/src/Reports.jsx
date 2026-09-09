import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Eye, Download, FileX, AlertCircle } from 'lucide-react'
import { getHistory, reportUrl } from './api/client'
import VerdictBadge from './components/VerdictBadge'
import LoadingState from './components/LoadingState'
import CaseTag from './components/CaseTag'
import Reveal from './components/Reveal'
import { isFresh } from './lib/freshness'
import { stationTag } from './lib/stations'

export default function Reports() {
  const [reports, setReports] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await getHistory(50)
        setReports(res.data || [])
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetch()
  }, [])

  const verdictOf = (r) => (r.verdict || r.label || '').toString().toUpperCase()
  const realCount = reports.filter(r => verdictOf(r) === 'REAL').length
  const fakeCount = reports.filter(r => verdictOf(r) === 'FAKE').length
  const uncertainCount = reports.filter(r => verdictOf(r) === 'UNCERTAIN').length
  const total = reports.length || 1

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />

      <section className="pt-16 md:pt-24 pb-12 border-b border-line">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <p className="tl-hud-label mb-4">{stationTag(4)} — Evidence Registry</p>
          <div className="flex items-end justify-between flex-wrap gap-6">
            <h1 className="font-serif text-display-l text-bone">
              Every verdict,<br /><span className="italic text-brass">indexed.</span>
            </h1>
            <p className="text-[15px] text-bone-dim max-w-[320px]">
              Forensic documents for every investigation, ready to review or download.
            </p>
          </div>
          <p className="text-[13px] text-bone-dim mt-4">
            Looking for a chronological view instead? See{' '}
            <button onClick={() => navigate('/history')} className="text-brass hover:text-bone transition-colors duration-300 link-underline">History</button>.
          </p>
        </div>
      </section>

      {!loading && !error && reports.length > 0 && (
        <section className="py-10 border-b border-line">
          <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
            <Reveal>
              <div className="flex items-center gap-10 md:gap-14 flex-wrap mb-6">
                <div>
                  <p className="tl-figure text-[30px] text-bone">{reports.length}</p>
                  <p className="tl-hud-label !text-[9px] mt-1">Total reports</p>
                </div>
                <div className="w-px h-9 bg-line-strong" />
                <div>
                  <p className="tl-figure text-[30px] text-verdictReal">{realCount}</p>
                  <p className="tl-hud-label !text-[9px] mt-1">Authentic</p>
                </div>
                <div className="w-px h-9 bg-line-strong" />
                <div>
                  <p className="tl-figure text-[30px] text-verdictDanger">{fakeCount}</p>
                  <p className="tl-hud-label !text-[9px] mt-1">Manipulated</p>
                </div>
                <div className="w-px h-9 bg-line-strong" />
                <div>
                  <p className="tl-figure text-[30px] text-verdictCaution">{uncertainCount}</p>
                  <p className="tl-hud-label !text-[9px] mt-1">Uncertain</p>
                </div>
              </div>
              <div className="h-1 rounded-full overflow-hidden flex bg-line">
                <div className="h-full bg-verdictReal" style={{ width: `${(realCount / total) * 100}%` }} />
                <div className="h-full bg-verdictDanger" style={{ width: `${(fakeCount / total) * 100}%` }} />
                <div className="h-full bg-verdictCaution" style={{ width: `${(uncertainCount / total) * 100}%` }} />
              </div>
            </Reveal>
          </div>
        </section>
      )}

      <section className="pb-24 md:pb-32">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          {loading && <LoadingState label="Retrieving documents" />}

          {error && (
            <div className="flex flex-col items-center text-center py-24">
              <AlertCircle size={32} strokeWidth={1.25} className="text-verdictDanger mb-4" />
              <p className="font-serif text-[20px] text-bone mb-2">Unable to load reports</p>
              <p className="text-[14px] text-verdictDanger">{error}</p>
            </div>
          )}

          {!loading && !error && reports.length === 0 && (
            <div className="flex flex-col items-center text-center py-24">
              <FileX size={32} strokeWidth={1.25} className="text-bone-faint mb-5" />
              <p className="font-serif text-[24px] text-bone mb-3">No reports yet</p>
              <p className="text-[15px] text-bone-dim mb-8">Generate your first report by verifying a media file.</p>
              <button onClick={() => navigate('/verify')} className="bg-brass text-ground px-8 py-4 rounded-[3px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift">
                Verify Media
              </button>
            </div>
          )}

          {!loading && !error && reports.length > 0 && (
            <div className="space-y-0 stagger-children mt-8">
              {reports.map((report, i) => {
                const date = new Date(report.uploaded_at || Date.now())
                const confidence = (report.confidence_score || 0) * 100
                const fresh = isFresh(report)
                return (
                  <div
                    key={report.upload_id || i}
                    role="button"
                    tabIndex={0}
                    aria-label={`View report for ${report.file_name || 'untitled report'}${fresh ? ' (just completed)' : ''}`}
                    className={`group flex items-center gap-6 py-6 border-b border-line cursor-pointer hover:bg-panel/50 transition-colors duration-300 -mx-4 px-4 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-brass ${fresh ? 'tl-fresh' : ''}`}
                    onClick={() => navigate(`/report/${report.upload_id}`)}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); navigate(`/report/${report.upload_id}`) } }}
                  >
                    <CaseTag id={report.upload_id} className="hidden md:block w-24 shrink-0" />

                    <VerdictBadge verdict={report.verdict || report.label} />

                    <div className="flex-1 min-w-0">
                      <p className="font-serif text-[16px] md:text-[18px] text-bone truncate group-hover:text-brass transition-colors duration-300">
                        {report.file_name || 'Untitled Report'}
                      </p>
                    </div>

                    {fresh && <span className="tl-hud-label !text-[9px] text-brass hidden sm:block">New</span>}

                    <div className="hidden md:flex items-center gap-2 w-28">
                      <div className="flex-1 h-[2px] bg-line-strong rounded-full overflow-hidden">
                        <div className="h-full bg-brass rounded-full" style={{ width: `${confidence.toFixed(1)}%` }} />
                      </div>
                      <span className="tl-figure text-[11px] text-bone-dim">{confidence.toFixed(1)}%</span>
                    </div>

                    <span className="tl-figure text-[11px] text-bone-dim hidden lg:block w-24">
                      {date.toLocaleDateString()}
                    </span>

                    <div className="hidden md:flex items-center gap-3 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity duration-300">
                      <button
                        onClick={(e) => { e.stopPropagation(); navigate(`/report/${report.upload_id}`) }}
                        className="w-11 h-11 rounded-full bg-panel-raised border border-line flex items-center justify-center text-brass hover:bg-brass hover:text-ground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass transition-colors duration-300"
                        title="View report"
                        aria-label="View report"
                      >
                        <Eye size={15} strokeWidth={1.75} />
                      </button>
                      <a
                        href={reportUrl(report.upload_id)}
                        download
                        onClick={(e) => e.stopPropagation()}
                        className="w-11 h-11 rounded-full bg-panel-raised border border-line flex items-center justify-center text-bone-dim hover:bg-bone hover:text-ground focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass transition-colors duration-300"
                        title="Download PDF"
                        aria-label="Download PDF report"
                      >
                        <Download size={15} strokeWidth={1.75} />
                      </a>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
