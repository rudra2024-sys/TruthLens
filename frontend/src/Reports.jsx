import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getHistory, reportUrl } from './api/client'

const DownloadIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M8 2V10" /><path d="M4 8L8 12L12 8" /><path d="M2 14H14" />
  </svg>
)

const EyeIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M1.5 8C1.5 8 4 3 8 3C12 3 14.5 8 14.5 8C14.5 8 12 13 8 13C4 13 1.5 8 1.5 8Z" />
    <circle cx="8" cy="8" r="2" />
  </svg>
)

const getVerdictStyle = (v) => {
  const val = (v || '').toString().toUpperCase()
  if (val === 'REAL' || val === 'AUTHENTIC') return { text: 'text-[#5A7A5A]', bg: 'bg-[rgba(90,122,90,0.08)]', border: 'border-[rgba(90,122,90,0.2)]', label: 'Authentic' }
  if (val === 'UNCERTAIN' || val === 'SUSPICIOUS') return { text: 'text-[#B89A6A]', bg: 'bg-[rgba(184,154,106,0.08)]', border: 'border-[rgba(184,154,106,0.2)]', label: 'Suspicious' }
  return { text: 'text-[#9A5A5A]', bg: 'bg-[rgba(154,90,90,0.08)]', border: 'border-[rgba(154,90,90,0.2)]', label: 'Manipulated' }
}

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

  return (
    <div className="min-h-screen bg-[#F7F5F0] pt-[72px]">
      <div className="tl-grain" />

      {/* Header */}
      <section className="pt-16 md:pt-24 pb-12">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <p className="text-[11px] tracking-[0.2em] uppercase text-[#A67B5B] mb-4 animate-fade-in-up">Documents</p>
          <div className="flex items-end justify-between flex-wrap gap-6">
            <h1 className="font-serif text-[clamp(2.5rem,5vw,4rem)] leading-[1.1] text-[#1A1A1A] animate-fade-in-up animate-delay-1">
              Your<br /><span className="italic">reports.</span>
            </h1>
            <p className="text-[15px] text-[#8A8580] max-w-[320px] animate-fade-in-up animate-delay-2">
              Detailed analysis documents, ready to review or download at any time.
            </p>
          </div>
        </div>
      </section>

      {/* Stats bar */}
      {!loading && !error && reports.length > 0 && (
        <section className="pb-12">
          <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
            <div className="flex items-center gap-12 py-8 border-y border-[rgba(138,133,128,0.1)]">
              <div>
                <p className="font-serif text-[32px] text-[#1A1A1A]">{reports.length}</p>
                <p className="text-[10px] tracking-[0.15em] uppercase text-[#8A8580] mt-1">Total reports</p>
              </div>
              <div className="w-px h-10 bg-[rgba(138,133,128,0.15)]" />
              <div>
                <p className="font-serif text-[32px] text-[#5A7A5A]">
                  {reports.filter(r => (r.verdict || r.label || '').toString().toUpperCase() === 'REAL').length}
                </p>
                <p className="text-[10px] tracking-[0.15em] uppercase text-[#8A8580] mt-1">Authentic</p>
              </div>
              <div className="w-px h-10 bg-[rgba(138,133,128,0.15)]" />
              <div>
                <p className="font-serif text-[32px] text-[#9A5A5A]">
                  {reports.filter(r => (r.verdict || r.label || '').toString().toUpperCase() === 'FAKE').length}
                </p>
                <p className="text-[10px] tracking-[0.15em] uppercase text-[#8A8580] mt-1">Manipulated</p>
              </div>
              <div className="w-px h-10 bg-[rgba(138,133,128,0.15)]" />
              <div>
                <p className="font-serif text-[32px] text-[#B89A6A]">
                  {reports.filter(r => (r.verdict || r.label || '').toString().toUpperCase() === 'UNCERTAIN').length}
                </p>
                <p className="text-[10px] tracking-[0.15em] uppercase text-[#8A8580] mt-1">Suspicious</p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Reports list */}
      <section className="pb-24 md:pb-32">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          {loading && (
            <div className="flex items-center justify-center py-24">
              <div className="w-12 h-12 rounded-full border-2 border-[rgba(166,123,91,0.15)] border-t-[#A67B5B] animate-spin" />
            </div>
          )}

          {error && (
            <div className="text-center py-24">
              <p className="font-serif text-[20px] text-[#1A1A1A] mb-2">Unable to load reports</p>
              <p className="text-[14px] text-[#9A5A5A]">{error}</p>
            </div>
          )}

          {!loading && !error && reports.length === 0 && (
            <div className="text-center py-24">
              <p className="font-serif text-[24px] text-[#1A1A1A] mb-3">No reports yet</p>
              <p className="text-[15px] text-[#8A8580] mb-8">Generate your first report by verifying a media file.</p>
              <button onClick={() => navigate('/verify')} className="bg-[#1A1A1A] text-[#F7F5F0] px-8 py-4 rounded-[4px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift">
                Verify Media
              </button>
            </div>
          )}

          {!loading && !error && reports.length > 0 && (
            <div className="space-y-0">
              {reports.map((report, i) => {
                const verdict = report.verdict || report.label || 'analyzed'
                const s = getVerdictStyle(verdict)
                const date = new Date(report.uploaded_at || Date.now())
                const confidence = (report.confidence_score || 0) * 100
                return (
                  <div
                    key={report.upload_id || i}
                    className="group flex items-center gap-6 py-6 border-b border-[rgba(138,133,128,0.1)] cursor-pointer hover:bg-[rgba(166,123,91,0.02)] transition-colors duration-300 -mx-4 px-4"
                    onClick={() => navigate(`/report/${report.upload_id}`)}
                  >
                    {/* Number */}
                    <span className="font-mono text-[11px] text-[#B8B0A8] w-8 hidden md:block">{String(i + 1).padStart(2, '0')}</span>

                    {/* Verdict badge */}
                    <span className={`text-[10px] font-medium tracking-[0.08em] uppercase px-3 py-1 rounded-full border whitespace-nowrap ${s.bg} ${s.border} ${s.text}`}>
                      {s.label}
                    </span>

                    {/* Filename */}
                    <div className="flex-1 min-w-0">
                      <p className="font-serif text-[16px] md:text-[18px] text-[#1A1A1A] truncate group-hover:text-[#A67B5B] transition-colors duration-300">
                        {report.file_name || 'Untitled Report'}
                      </p>
                    </div>

                    {/* Confidence */}
                    <div className="hidden md:flex items-center gap-2 w-28">
                      <div className="flex-1 h-[2px] bg-[rgba(138,133,128,0.15)] rounded-full overflow-hidden">
                        <div className="h-full bg-[#A67B5B] rounded-full" style={{ width: `${confidence.toFixed(1)}%` }} />
                      </div>
                      <span className="font-mono text-[11px] text-[#8A8580]">{confidence.toFixed(1)}%</span>
                    </div>

                    {/* Date */}
                    <span className="font-mono text-[11px] text-[#8A8580] hidden lg:block w-24">
                      {date.toLocaleDateString()}
                    </span>

                    {/* Actions */}
                    <div className="flex items-center gap-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                      <button
                        onClick={(e) => { e.stopPropagation(); navigate(`/report/${report.upload_id}`) }}
                        className="w-8 h-8 rounded-full bg-[rgba(166,123,91,0.08)] flex items-center justify-center text-[#A67B5B] hover:bg-[#A67B5B] hover:text-white transition-all duration-300"
                        title="View report"
                      >
                        <EyeIcon />
                      </button>
                      <a
                        href={reportUrl(report.upload_id)}
                        download
                        onClick={(e) => e.stopPropagation()}
                        className="w-8 h-8 rounded-full bg-[rgba(138,133,128,0.08)] flex items-center justify-center text-[#8A8580] hover:bg-[#1A1A1A] hover:text-white transition-all duration-300"
                        title="Download PDF"
                      >
                        <DownloadIcon />
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
