import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Image as ImageIcon, Video as VideoIcon, AudioLines, ArrowRight, SearchX, AlertCircle } from 'lucide-react'
import { getHistory } from './api/client'
import VerdictBadge from './components/VerdictBadge'
import LoadingState from './components/LoadingState'
import CaseTag from './components/CaseTag'
import Reveal from './components/Reveal'
import { isFresh } from './lib/freshness'

const getFileIcon = (type) => {
  if (!type) return <ImageIcon size={14} strokeWidth={1.5} />
  if (type.startsWith('image')) return <ImageIcon size={14} strokeWidth={1.5} />
  if (type.startsWith('video')) return <VideoIcon size={14} strokeWidth={1.5} />
  if (type.startsWith('audio')) return <AudioLines size={14} strokeWidth={1.5} />
  return <ImageIcon size={14} strokeWidth={1.5} />
}

const groupByDate = (items) => {
  const groups = {}
  items.forEach(item => {
    const date = new Date(item.uploaded_at || Date.now())
    const key = date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
    if (!groups[key]) groups[key] = []
    groups[key].push(item)
  })
  return groups
}

export default function History() {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await getHistory(50)
        setHistory(res.data || [])
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    fetch()
  }, [])

  const groups = groupByDate(history)

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />

      <section className="pt-16 md:pt-24 pb-12 border-b border-line">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <p className="tl-hud-label mb-4">Case Archive</p>
          <div className="flex items-end justify-between flex-wrap gap-6">
            <h1 className="font-serif text-display-l text-bone">
              Every investigation,<br /><span className="italic text-brass">on record.</span>
            </h1>
            <p className="text-[15px] text-bone-dim max-w-[300px]">
              Browsable by month, in the order you ran it.
            </p>
          </div>
          <p className="text-[13px] text-bone-dim mt-4">
            Looking for a document to open or download? See{' '}
            <button onClick={() => navigate('/reports')} className="text-brass hover:text-bone transition-colors duration-300 link-underline">Reports</button>.
          </p>
        </div>
      </section>

      <section className="pb-24 md:pb-32">
        <div className="max-w-[1100px] mx-auto px-6 md:px-12 lg:px-16">
          {loading && <LoadingState label="Retrieving records" />}

          {error && (
            <div className="flex flex-col items-center text-center py-24">
              <AlertCircle size={32} strokeWidth={1.25} className="text-verdictDanger mb-4" />
              <p className="font-serif text-[20px] text-bone mb-2">Unable to load history</p>
              <p className="text-[14px] text-verdictDanger">{error}</p>
            </div>
          )}

          {!loading && !error && history.length === 0 && (
            <div className="flex flex-col items-center text-center py-24">
              <SearchX size={32} strokeWidth={1.25} className="text-bone-faint mb-5" />
              <p className="font-serif text-[24px] text-bone mb-3">No investigations yet</p>
              <p className="text-[15px] text-bone-dim mb-8">Start by verifying your first media file.</p>
              <button onClick={() => navigate('/verify')} className="bg-brass text-ground px-8 py-4 rounded-[3px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift">
                Verify Media
              </button>
            </div>
          )}

          {!loading && !error && history.length > 0 && (
            <div className="space-y-16 mt-12">
              {Object.entries(groups).map(([month, items], gi) => (
                <Reveal key={month} delay={Math.min(gi * 0.05, 0.2)}>
                  <div>
                    <div className="flex items-center gap-6 mb-2">
                      <h2 className="tl-hud-label !text-[12px] !text-bone">{month}</h2>
                      <div className="flex-1 h-px bg-line-strong" />
                      <span className="tl-figure text-[11px] text-bone-faint">{items.length} scans</span>
                    </div>
                    <div>
                      {items.map((item, i) => {
                        const date = new Date(item.uploaded_at || Date.now())
                        const openReport = () => navigate(`/report/${item.upload_id}`)
                        const fresh = isFresh(item)
                        return (
                          <div
                            key={item.upload_id || i}
                            role="button"
                            tabIndex={0}
                            aria-label={`View report for ${item.file_name || 'untitled scan'}${fresh ? ' (just completed)' : ''}`}
                            className={`group flex items-center gap-4 md:gap-6 py-4 border-b border-line cursor-pointer hover:bg-panel/50 transition-colors duration-300 -mx-4 px-4 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-brass ${fresh ? 'tl-fresh' : ''}`}
                            onClick={openReport}
                            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openReport() } }}
                          >
                            <CaseTag id={item.upload_id} className="hidden md:block w-24 shrink-0" />
                            <span className="text-bone-faint shrink-0">{getFileIcon(item.media_type)}</span>
                            <div className="min-w-0 flex-1">
                              <p className="font-serif text-[16px] text-bone truncate">{item.file_name || 'Untitled'}</p>
                              <p className="tl-figure text-[10px] text-bone-faint mt-0.5">
                                {date.toLocaleDateString()} · {date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                              </p>
                            </div>
                            {fresh && <span className="tl-hud-label !text-[9px] text-brass hidden sm:block">New</span>}
                            <span className="tl-figure text-[12px] text-bone-dim hidden sm:block w-12 text-right shrink-0">
                              {Math.round((item.confidence_score || 0) * 100)}%
                            </span>
                            <VerdictBadge verdict={item.verdict} className="shrink-0" />
                            <span className="text-brass opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100 transition-opacity duration-300 shrink-0">
                              <ArrowRight size={14} strokeWidth={1.75} />
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </Reveal>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
