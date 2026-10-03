import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Image as ImageIcon, Video as VideoIcon, AudioLines, ArrowRight, SearchX, AlertCircle } from 'lucide-react'
import { getHistory, getStats } from './api/client'
import VerdictBadge from './components/VerdictBadge'
import LoadingState from './components/LoadingState'
import CaseTag from './components/CaseTag'
import Reveal from './components/Reveal'

/**
 * A stats overview, built 2026-10-03 to replace the long-dead src/pages/Dashboard.jsx and Analytics.jsx
 * (confirmed unrouted and unimported anywhere, CLAUDE.md's dead-code list). Deliberately NOT a port of
 * either: both used a completely different, no-longer-existing colour system (mint/canvas/snow/stroke,
 * absent from tailwind.config.js's current ground/bone/brass/verdict* palette) and Dashboard.jsx's three
 * "pipeline" cards described stale, fabricated processing ("Noise residual + FFT spectrum" for image,
 * "Waveform / byte spectral cues" for audio) that violates CLAUDE.md section 8 rule 5 (no fabricated model
 * claims) — the real pipelines are ConvNeXt-Tiny+CLIP, Video Model v1, and Audio Model v1 (wav2vec2), and a
 * stats page has no good reason to re-describe them inaccurately when nothing here needs to at all. Plain
 * counts only, computed from genuine /stats and /history responses, no re-derived or invented numbers.
 */
const MEDIA_ICON = { image: ImageIcon, video: VideoIcon, audio: AudioLines }

function StatBlock({ label, value, accent = false }) {
  return (
    <div>
      <p className="tl-hud-label !text-[10px] mb-2">{label}</p>
      <p className={`tl-figure text-[32px] md:text-[40px] leading-none ${accent ? 'text-brass' : 'text-bone'}`}>
        {value}
      </p>
    </div>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [recent, setRecent] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    Promise.all([getStats(), getHistory(8)])
      .then(([s, h]) => { setStats(s.data); setRecent(h.data || []) })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const byMedia = stats?.by_media_type || {}

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />

      <section className="pt-16 md:pt-24 pb-12 border-b border-line">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <p className="tl-hud-label mb-4">Overview</p>
          <div className="flex items-end justify-between flex-wrap gap-6">
            <h1 className="font-serif text-display-l text-bone">
              The ledger,<br /><span className="italic text-brass">at a glance.</span>
            </h1>
            <p className="text-[15px] text-bone-dim max-w-[300px]">
              Totals and recent activity, read straight from your own scan history.
            </p>
          </div>
        </div>
      </section>

      <section className="pb-24 md:pb-32">
        <div className="max-w-[1100px] mx-auto px-6 md:px-12 lg:px-16">
          {loading && <LoadingState label="Tallying records" />}

          {error && (
            <div className="flex flex-col items-center text-center py-24">
              <AlertCircle size={32} strokeWidth={1.25} className="text-verdictDanger mb-4" />
              <p className="font-serif text-[20px] text-bone mb-2">Unable to load the overview</p>
              <p className="text-[14px] text-verdictDanger">{error}</p>
            </div>
          )}

          {!loading && !error && (stats?.total_scans ?? 0) === 0 && (
            <div className="flex flex-col items-center text-center py-24">
              <SearchX size={32} strokeWidth={1.25} className="text-bone-faint mb-5" />
              <p className="font-serif text-[24px] text-bone mb-3">No investigations yet</p>
              <p className="text-[15px] text-bone-dim mb-8">Start by verifying your first media file.</p>
              <button onClick={() => navigate('/verify')} className="bg-brass text-ground px-8 py-4 rounded-[3px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift">
                Verify Media
              </button>
            </div>
          )}

          {!loading && !error && (stats?.total_scans ?? 0) > 0 && (
            <>
              <Reveal>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-10 pt-2 pb-12 border-b border-line mb-12">
                  <StatBlock label="Total scans" value={stats.total_scans} accent />
                  <StatBlock label="Flagged fake" value={stats.fake_count} />
                  <StatBlock label="Verified real" value={stats.real_count} />
                  <StatBlock label="Uncertain" value={stats.uncertain_count} />
                </div>
              </Reveal>

              <Reveal delay={0.05}>
                <h2 className="tl-hud-label !text-[12px] !text-bone mb-6">Scans by media type</h2>
                <div className="grid sm:grid-cols-3 gap-4 mb-14">
                  {['image', 'video', 'audio'].map((type) => {
                    const Icon = MEDIA_ICON[type]
                    return (
                      <div key={type} className="border border-line rounded-[4px] p-5 flex items-center gap-4">
                        <Icon size={18} strokeWidth={1.5} className="text-bone-faint shrink-0" />
                        <div>
                          <p className="tl-figure text-[22px] text-bone leading-none">{byMedia[type] ?? 0}</p>
                          <p className="text-[12px] text-bone-dim capitalize mt-1">{type}</p>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </Reveal>

              <Reveal delay={0.1}>
                <div className="flex items-end justify-between gap-4 mb-4">
                  <h2 className="tl-hud-label !text-[12px] !text-bone">Recent activity</h2>
                  <button type="button" onClick={() => navigate('/history')} className="text-[12px] text-brass hover:text-bone transition-colors duration-300 inline-flex items-center gap-1 link-underline">
                    Full history <ArrowRight size={12} strokeWidth={1.75} />
                  </button>
                </div>
                <div>
                  {recent.map((item) => {
                    const Icon = MEDIA_ICON[item.media_type] || ImageIcon
                    const openReport = () => navigate(`/report/${item.upload_id}`)
                    return (
                      <div
                        key={item.upload_id}
                        role="button"
                        tabIndex={0}
                        aria-label={`View report for ${item.file_name || 'untitled scan'}`}
                        className="group flex items-center gap-4 md:gap-6 py-4 border-b border-line cursor-pointer hover:bg-panel/50 transition-colors duration-300 -mx-4 px-4 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-brass"
                        onClick={openReport}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openReport() } }}
                      >
                        <CaseTag id={item.upload_id} className="hidden md:block w-24 shrink-0" />
                        <Icon size={14} strokeWidth={1.5} className="text-bone-faint shrink-0" />
                        <p className="font-serif text-[15px] text-bone truncate flex-1">{item.file_name || 'Untitled'}</p>
                        {item.verdict ? <VerdictBadge verdict={item.verdict} className="shrink-0" /> : (
                          <span className="tl-hud-label !text-[9px] text-bone-faint shrink-0">pending</span>
                        )}
                        <span className="text-brass opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100 transition-opacity duration-300 shrink-0 hidden sm:block">
                          <ArrowRight size={14} strokeWidth={1.75} />
                        </span>
                      </div>
                    )
                  })}
                </div>
              </Reveal>
            </>
          )}
        </div>
      </section>
    </div>
  )
}
