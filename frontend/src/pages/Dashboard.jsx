import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowUpRight, ImageIcon, Video, Music } from 'lucide-react'
import { getStats, getHistory } from '../api/client'
import { Eyebrow, Stat, Badge, AnimatedCounter, Card } from '../components/ui'
import HeuristicNotice from '../components/StubBadge'

const MEDIA_ICON = { image: ImageIcon, video: Video, audio: Music }
const VERDICT_TONE = { FAKE: 'fake', REAL: 'real', UNCERTAIN: 'caution' }

export default function Dashboard({ onGoScan, onGoAnalytics }) {
  const [stats, setStats] = useState(null)
  const [recent, setRecent] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getStats(), getHistory(6)])
      .then(([s, h]) => { setStats(s.data); setRecent(h.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const pipelinesActive = stats
    ? ['image', 'video', 'audio'].filter(t => (stats.by_media_type?.[t] ?? 0) > 0).length
    : 0

  return (
    <div>
      <motion.section className="mb-14 md:mb-16 max-w-3xl animate-fadeUp">
        <p className="font-display font-extrabold text-mint text-sm tracking-[0.2em] uppercase mb-5">
          TruthLens
        </p>
        <h1 className="font-display font-extrabold text-4xl sm:text-5xl md:text-6xl leading-[1.05] tracking-tight text-snow mb-5">
          Detect. Decide.<br />
          <span className="text-mint">Keep the record.</span>
        </h1>
        <p className="text-soft text-base md:text-lg leading-relaxed mb-8 max-w-xl">
          Forensic review for images, video, and audio — every scan stored, every score tied to the file you uploaded.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={onGoScan}
            className="bg-mint text-canvas font-bold px-6 py-3.5 text-sm rounded-xl hover:brightness-110 transition shadow-glow"
          >
            Start a scan
          </button>
          <button
            type="button"
            onClick={onGoAnalytics}
            className="text-sm font-semibold text-soft hover:text-snow px-4 py-3.5 transition inline-flex items-center gap-1.5"
          >
            View analytics <ArrowUpRight className="w-4 h-4" />
          </button>
        </div>
      </motion.section>

      {error && (
        <div className="mb-8 rounded-xl px-4 py-3 text-sm border" style={{ background: 'rgba(255,77,106,0.1)', borderColor: 'rgba(255,77,106,0.3)', color: '#FF4D6A' }}>
          {error}
        </div>
      )}

      <section className="mb-14">
        <Eyebrow>/ 01 · Live from your database</Eyebrow>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8 pt-6 border-t border-stroke">
          <Stat
            label="Files processed"
            color="#3DFF9A"
            value={loading ? '—' : <AnimatedCounter value={stats?.total_scans ?? 0} />}
          />
          <Stat
            label="Flagged manipulated"
            color="#FF4D6A"
            value={loading ? '—' : <AnimatedCounter value={stats?.fake_count ?? 0} />}
          />
          <Stat
            label="Verified authentic"
            color="#3DFF9A"
            value={loading ? '—' : <AnimatedCounter value={stats?.real_count ?? 0} />}
          />
          <Stat
            label="Pipelines with scans"
            color="#F3F5F9"
            value={loading ? '—' : pipelinesActive}
          />
        </div>
      </section>

      <section className="mb-14">
        <Eyebrow>/ 02 · Detection pipelines</Eyebrow>
        <h2 className="font-display font-bold text-2xl md:text-3xl text-snow tracking-tight mb-2">
          Three media paths. One ledger.
        </h2>
        <p className="text-soft text-sm mb-8 max-w-lg">
          Counts come from uploads already in SQLite.
        </p>
        <div className="grid md:grid-cols-3 gap-4">
          {[
            { icon: ImageIcon, name: 'Image', detail: 'Noise residual + FFT spectrum', formats: 'PNG · JPG · WEBP', count: stats?.by_media_type?.image ?? 0 },
            { icon: Video, name: 'Video', detail: 'Container entropy + structure', formats: 'MP4 · MOV · WEBM', count: stats?.by_media_type?.video ?? 0 },
            { icon: Music, name: 'Audio', detail: 'Waveform / byte spectral cues', formats: 'WAV · MP3 · FLAC', count: stats?.by_media_type?.audio ?? 0 },
          ].map((p) => (
            <Card key={p.name} className="p-6 hover:border-mint/40 transition-colors">
              <p.icon className="w-5 h-5 text-mint mb-4" />
              <p className="font-display font-bold text-xl text-snow mb-1">{p.name}</p>
              <p className="text-sm text-soft mb-4">{p.detail}</p>
              <p className="text-xs text-soft/80 mb-5">{p.formats}</p>
              <p className="font-display text-3xl font-extrabold text-snow">
                {loading ? '—' : p.count}
                <span className="text-sm font-body font-normal text-soft ml-2">scanned</span>
              </p>
            </Card>
          ))}
        </div>
      </section>

      <section>
        <div className="flex items-end justify-between gap-4 mb-4">
          <div>
            <Eyebrow>/ 03 · Recent activity</Eyebrow>
            <h2 className="font-display font-bold text-2xl text-snow tracking-tight">What just ran</h2>
          </div>
          <button type="button" onClick={onGoAnalytics} className="text-sm text-mint hover:underline inline-flex items-center gap-1 font-semibold">
            All records <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>
        <div className="mb-4"><HeuristicNotice /></div>
        <Card>
          {!loading && recent.length === 0 && (
            <div className="p-12 text-center text-soft text-sm">No scans yet. Run your first detection.</div>
          )}
          <ul className="divide-y divide-stroke">
            {recent.map(item => {
              const MIcon = MEDIA_ICON[item.media_type] ?? ImageIcon
              return (
                <li key={item.upload_id} className="flex items-center gap-4 px-5 py-4">
                  <div className="w-9 h-9 rounded-lg bg-elev2 flex items-center justify-center shrink-0">
                    <MIcon className="w-4 h-4 text-soft" />
                  </div>
                  <p className="text-sm truncate flex-1 font-medium text-snow">{item.file_name}</p>
                  {item.verdict ? (
                    <Badge tone={VERDICT_TONE[item.verdict]}>
                      {item.verdict} · {Math.round(item.confidence_score * 100)}%
                    </Badge>
                  ) : (
                    <Badge tone="muted">pending</Badge>
                  )}
                </li>
              )
            })}
          </ul>
        </Card>
      </section>
    </div>
  )
}
