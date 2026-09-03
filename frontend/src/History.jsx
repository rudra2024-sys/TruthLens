import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getHistory } from './client'

const ImageIcon = () => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <rect x="2" y="3" width="16" height="14" rx="2" /><circle cx="7" cy="8" r="1.5" /><path d="M2 14L7 9L11 13L14 10L18 14" />
  </svg>
)

const VideoIcon = () => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <rect x="2" y="4" width="16" height="12" rx="2" /><polygon points="8,7 8,13 14,10" fill="currentColor" />
  </svg>
)

const AudioIcon = () => (
  <svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M10 2V18" /><path d="M6 6C6 6 4 8 4 10C4 12 6 14 6 14" /><path d="M14 6C14 6 16 8 16 10C16 12 14 14 14 14" />
  </svg>
)

const ArrowRight = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 8H13" /><path d="M9 4L13 8L9 12" />
  </svg>
)

const getVerdictStyle = (v) => {
  if (v === 'authentic' || v === 'real') return { text: 'text-[#5A7A5A]', bg: 'bg-[rgba(90,122,90,0.08)]', border: 'border-[rgba(90,122,90,0.2)]', label: 'Authentic' }
  if (v === 'suspicious') return { text: 'text-[#B89A6A]', bg: 'bg-[rgba(184,154,106,0.08)]', border: 'border-[rgba(184,154,106,0.2)]', label: 'Suspicious' }
  return { text: 'text-[#9A5A5A]', bg: 'bg-[rgba(154,90,90,0.08)]', border: 'border-[rgba(154,90,90,0.2)]', label: 'Manipulated' }
}

const getFileIcon = (type) => {
  if (!type) return <ImageIcon />
  if (type.startsWith('image/')) return <ImageIcon />
  if (type.startsWith('video/')) return <VideoIcon />
  if (type.startsWith('audio/')) return <AudioIcon />
  return <ImageIcon />
}

const groupByDate = (items) => {
  const groups = {}
  items.forEach(item => {
    const date = new Date(item.created_at || item.date || Date.now())
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
    <div className="min-h-screen bg-[#F7F5F0] pt-[72px]">
      <div className="tl-grain" />

      {/* Header */}
      <section className="pt-16 md:pt-24 pb-12">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <p className="text-[11px] tracking-[0.2em] uppercase text-[#A67B5B] mb-4 animate-fade-in-up">Archive</p>
          <div className="flex items-end justify-between flex-wrap gap-6">
            <h1 className="font-serif text-[clamp(2.5rem,5vw,4rem)] leading-[1.1] text-[#1A1A1A] animate-fade-in-up animate-delay-1">
              Your<br /><span className="italic">history.</span>
            </h1>
            <p className="text-[15px] text-[#8A8580] max-w-[300px] animate-fade-in-up animate-delay-2">
              Every file you've analyzed, preserved in chronological order.
            </p>
          </div>
        </div>
      </section>

      {/* Content */}
      <section className="pb-24 md:pb-32">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          {loading && (
            <div className="flex items-center justify-center py-24">
              <div className="w-12 h-12 rounded-full border-2 border-[rgba(166,123,91,0.15)] border-t-[#A67B5B] animate-spin" />
            </div>
          )}

          {error && (
            <div className="text-center py-24">
              <p className="font-serif text-[20px] text-[#1A1A1A] mb-2">Unable to load history</p>
              <p className="text-[14px] text-[#9A5A5A]">{error}</p>
            </div>
          )}

          {!loading && !error && history.length === 0 && (
            <div className="text-center py-24">
              <p className="font-serif text-[24px] text-[#1A1A1A] mb-3">No scans yet</p>
              <p className="text-[15px] text-[#8A8580] mb-8">Start by verifying your first media file.</p>
              <button onClick={() => navigate('/verify')} className="bg-[#1A1A1A] text-[#F7F5F0] px-8 py-4 rounded-[4px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift">
                Verify Media
              </button>
            </div>
          )}

          {!loading && !error && history.length > 0 && (
            <div className="space-y-20">
              {Object.entries(groups).map(([month, items]) => (
                <div key={month}>
                  <div className="flex items-center gap-6 mb-8">
                    <h2 className="font-serif text-[24px] text-[#1A1A1A]">{month}</h2>
                    <div className="flex-1 h-px bg-[rgba(138,133,128,0.15)]" />
                    <span className="font-mono text-[12px] text-[#8A8580]">{items.length} scans</span>
                  </div>
                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {items.map((item, i) => {
                      const verdict = item.verdict || item.label || 'analyzed'
                      const s = getVerdictStyle(verdict)
                      const date = new Date(item.created_at || item.date || Date.now())
                      return (
                        <div
                          key={item.upload_id || i}
                          className="group bg-[#FAFAF8] rounded-[8px] p-6 card-hover cursor-pointer"
                          onClick={() => navigate(`/report/${item.upload_id}`)}
                        >
                          <div className="flex items-start justify-between mb-5">
                            <div className="flex items-center gap-3">
                              <div className="w-10 h-10 rounded-full bg-[rgba(166,123,91,0.06)] flex items-center justify-center text-[#8A8580]">
                                {getFileIcon(item.file_type)}
                              </div>
                              <div>
                                <p className="font-mono text-[10px] tracking-wider text-[#8A8580] uppercase">{date.toLocaleDateString()}</p>
                                <p className="font-mono text-[10px] text-[#B8B0A8]">{date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</p>
                              </div>
                            </div>
                            <span className={`text-[10px] font-medium tracking-[0.08em] uppercase px-3 py-1 rounded-full border ${s.bg} ${s.border} ${s.text}`}>
                              {s.label}
                            </span>
                          </div>
                          <p className="font-serif text-[18px] text-[#1A1A1A] mb-3 truncate">{item.filename || 'Untitled'}</p>
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="text-[11px] text-[#8A8580]">Confidence</span>
                              <span className="font-mono text-[12px] text-[#1A1A1A]">{Math.round((item.confidence || item.score || 0) * 100)}%</span>
                            </div>
                            <span className="text-[#A67B5B] opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                              <ArrowRight />
                            </span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  )
}
