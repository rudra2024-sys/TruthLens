import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { ImageIcon, Video, Music, ExternalLink } from 'lucide-react'
import { getStats } from '../api/client'
import { Eyebrow, Badge, Card } from '../components/ui'
import HeuristicNotice from '../components/StubBadge'

const PIPELINES = [
  {
    key: 'image',
    icon: ImageIcon,
    name: 'Image forensics',
    purpose: 'Noise residual + FFT spectrum analysis',
    architecture: 'Pillow-based pixel statistics and frequency-domain energy ratios on the uploaded image bytes.',
    formats: 'PNG, JPG, WebP',
    paper: 'https://arxiv.org/abs/1905.11946',
    paperLabel: 'EfficientNet (architecture reference)',
  },
  {
    key: 'video',
    icon: Video,
    name: 'Video forensics',
    purpose: 'Container structure + byte entropy sampling',
    architecture: 'Multi-window entropy and packing density over the stored video file.',
    formats: 'MP4, MOV, WebM',
    paper: 'https://arxiv.org/abs/1610.02357',
    paperLabel: 'Xception (architecture reference)',
  },
  {
    key: 'audio',
    icon: Music,
    name: 'Audio forensics',
    purpose: 'Waveform / spectral byte cues',
    architecture: 'WAV PCM stats when available; otherwise sampled-byte entropy for compressed audio.',
    formats: 'WAV, MP3, FLAC',
    paper: 'https://arxiv.org/abs/2006.11477',
    paperLabel: 'wav2vec 2.0 (architecture reference)',
  },
]

export default function Models() {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getStats().then(r => setStats(r.data)).catch(e => setError(e.message))
  }, [])

  return (
    <div>
      <Eyebrow>/ 01 · Pipelines</Eyebrow>
      <h1 className="font-display font-extrabold text-3xl md:text-5xl tracking-tight text-snow mb-2">
        How we score media
      </h1>
      <p className="text-soft text-sm mb-6 max-w-xl">
        Live scan counts from your database. Status reflects content heuristics — not trained DNN accuracy.
      </p>
      <div className="mb-10"><HeuristicNotice /></div>

      {error && (
        <div className="mb-6 rounded-xl px-4 py-3 text-sm border" style={{ background: 'rgba(255,77,106,0.1)', borderColor: 'rgba(255,77,106,0.3)', color: '#FF4D6A' }}>
          {error}
        </div>
      )}

      <div className="space-y-4">
        {PIPELINES.map((m, i) => {
          const count = stats?.by_media_type?.[m.key] ?? 0
          return (
            <motion.div
              key={m.key}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06 }}
            >
              <Card className="p-6 md:p-7 flex flex-col md:flex-row md:items-start gap-6 justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-3 mb-3">
                    <m.icon className="w-5 h-5 text-mint" />
                    <p className="font-display font-bold text-xl text-snow">{m.name}</p>
                    <Badge tone="caution">Heuristic pipeline active</Badge>
                  </div>
                  <p className="text-sm text-soft mb-2">{m.purpose}</p>
                  <p className="text-sm text-snow/80 leading-relaxed max-w-2xl mb-4">{m.architecture}</p>
                  <p className="text-xs text-soft mb-3">Formats: {m.formats}</p>
                  <a href={m.paper} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 text-xs text-mint hover:underline font-semibold">
                    {m.paperLabel} <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
                <div className="md:text-right shrink-0">
                  <p className="font-display text-5xl font-extrabold text-mint">
                    {stats == null ? '—' : count}
                  </p>
                  <p className="text-xs text-soft mt-1">scans in DB</p>
                </div>
              </Card>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
