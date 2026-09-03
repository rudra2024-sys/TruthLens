import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import { ImageIcon, Video, Music, Download, RefreshCw, Search, ArrowUpDown } from 'lucide-react'
import HeuristicNotice from '../components/StubBadge'
import { getHistory, reportUrl } from '../api/client'
import { Eyebrow, Badge, Card } from '../components/ui'

const MEDIA_ICON = { image: ImageIcon, video: Video, audio: Music }
const VERDICT_TONE = { FAKE: 'fake', REAL: 'real', UNCERTAIN: 'caution' }
const PAGE_SIZE = 10

export default function HistoryPage({ onGoScan }) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('ALL')
  const [sortDesc, setSortDesc] = useState(true)
  const [page, setPage] = useState(1)

  const load = () => {
    setLoading(true)
    getHistory(100).then(r => setItems(r.data)).catch(e => setError(e.message)).finally(() => setLoading(false))
  }
  useEffect(load, [])

  const filtered = useMemo(() => {
    let out = items.filter(i => i.file_name.toLowerCase().includes(query.toLowerCase()))
    if (filter !== 'ALL') out = out.filter(i => i.verdict === filter)
    out.sort((a, b) => sortDesc
      ? new Date(b.uploaded_at) - new Date(a.uploaded_at)
      : new Date(a.uploaded_at) - new Date(b.uploaded_at))
    return out
  }, [items, query, filter, sortDesc])

  const paged = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))

  return (
    <div>
      <div className="flex items-end justify-between mb-8 flex-wrap gap-4">
        <div>
          <Eyebrow>/ 01 · Scan log</Eyebrow>
          <h1 className="font-display font-extrabold text-3xl md:text-5xl tracking-tight text-snow">History</h1>
          <p className="text-soft text-sm mt-2">Every upload and verdict stored in SQLite.</p>
        </div>
        <button
          type="button"
          onClick={load}
          className="flex items-center gap-2 text-sm text-soft hover:text-snow px-3.5 py-2 rounded-xl border border-stroke bg-elev"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      {error && (
        <div className="mb-6 rounded-xl px-4 py-3 text-sm border" style={{ background: 'rgba(255,77,106,0.1)', borderColor: 'rgba(255,77,106,0.3)', color: '#FF4D6A' }}>
          {error}
        </div>
      )}

      <div className="mb-4"><HeuristicNotice /></div>

      <Card className="mb-4 p-3 flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="w-4 h-4 text-soft absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            value={query}
            onChange={e => { setQuery(e.target.value); setPage(1) }}
            placeholder="Search by filename…"
            className="w-full bg-elev2 border border-stroke rounded-xl pl-9 pr-3 py-2.5 text-sm text-snow outline-none focus:border-mint placeholder:text-soft"
          />
        </div>
        <div className="flex gap-1">
          {['ALL', 'REAL', 'FAKE', 'UNCERTAIN'].map(f => (
            <button
              key={f}
              type="button"
              onClick={() => { setFilter(f); setPage(1) }}
              className={`px-3 py-2 text-xs font-semibold rounded-lg transition ${
                filter === f ? 'bg-mint text-canvas' : 'bg-elev2 text-soft hover:text-snow'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setSortDesc(s => !s)}
          className="flex items-center gap-1.5 px-3 py-2 text-xs rounded-lg border border-stroke text-soft hover:text-snow bg-elev2"
        >
          <ArrowUpDown className="w-3.5 h-3.5" /> {sortDesc ? 'Newest' : 'Oldest'}
        </button>
      </Card>

      <Card>
        {!loading && filtered.length === 0 && (
          <div className="py-16 text-center">
            <p className="text-soft text-sm mb-4">
              {items.length === 0 ? 'No scans yet.' : 'No results match your filters.'}
            </p>
            {items.length === 0 && (
              <button type="button" onClick={onGoScan} className="text-sm bg-mint text-canvas px-4 py-2.5 rounded-xl font-bold">
                Start first scan
              </button>
            )}
          </div>
        )}
        <ul className="divide-y divide-stroke">
          {paged.map((item, i) => {
            const MIcon = MEDIA_ICON[item.media_type] ?? ImageIcon
            return (
              <motion.li
                key={item.upload_id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.02 }}
                className="flex items-center gap-4 px-5 py-4"
              >
                <div className="w-9 h-9 rounded-lg bg-elev2 flex items-center justify-center shrink-0">
                  <MIcon className="w-4 h-4 text-soft" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-snow truncate">{item.file_name}</p>
                  <p className="text-[11px] text-soft">{new Date(item.uploaded_at).toLocaleString()}</p>
                </div>
                {item.verdict ? (
                  <div className="flex items-center gap-2">
                    <Badge tone={VERDICT_TONE[item.verdict]}>
                      {item.verdict} · {Math.round(item.confidence_score * 100)}%
                    </Badge>
                    <HeuristicNotice compact />
                  </div>
                ) : <Badge tone="muted">pending</Badge>}
                {item.verdict && (
                  <a href={reportUrl(item.upload_id)} download className="shrink-0 p-2 rounded-lg text-soft hover:text-mint hover:bg-elev2 transition">
                    <Download className="w-4 h-4" />
                  </a>
                )}
              </motion.li>
            )
          })}
        </ul>
      </Card>

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-8">
          {Array.from({ length: totalPages }).map((_, i) => (
            <button
              key={i}
              type="button"
              onClick={() => setPage(i + 1)}
              className={`w-9 h-9 rounded-lg text-xs font-semibold transition ${
                page === i + 1 ? 'bg-mint text-canvas' : 'bg-elev text-soft hover:text-snow border border-stroke'
              }`}
            >
              {i + 1}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
