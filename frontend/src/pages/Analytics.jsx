import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid,
  PieChart, Pie, Cell, BarChart, Bar,
} from 'recharts'
import { getHistory, getStats } from '../api/client'
import { Eyebrow, Stat, Card } from '../components/ui'
import HeuristicNotice from '../components/StubBadge'

const COLORS = { REAL: '#3DFF9A', FAKE: '#FF4D6A', UNCERTAIN: '#FFB020' }
const MEDIA_COLORS = { image: '#3DFF9A', video: '#F3F5F9', audio: '#8B94A8' }

function tipStyle() {
  return {
    background: '#161B26',
    border: '1px solid #262D3B',
    borderRadius: 12,
    fontSize: 12,
    color: '#F3F5F9',
  }
}

export default function Analytics() {
  const [items, setItems] = useState([])
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([getHistory(200), getStats()])
      .then(([h, s]) => { setItems(h.data); setStats(s.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const scansByDay = useMemo(() => {
    const map = {}
    items.forEach(i => {
      const day = new Date(i.uploaded_at).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' })
      map[day] = (map[day] || 0) + 1
    })
    return Object.entries(map).map(([day, count]) => ({ day, count })).slice(-14)
  }, [items])

  const verdictPie = useMemo(() => {
    const counts = { REAL: 0, FAKE: 0, UNCERTAIN: 0 }
    items.forEach(i => { if (i.verdict) counts[i.verdict] = (counts[i.verdict] || 0) + 1 })
    return Object.entries(counts).filter(([, v]) => v > 0).map(([name, value]) => ({ name, value }))
  }, [items])

  const mediaBar = useMemo(() => {
    const counts = { image: 0, video: 0, audio: 0 }
    items.forEach(i => { if (counts[i.media_type] !== undefined) counts[i.media_type]++ })
    return Object.entries(counts).map(([type, count]) => ({ type, count }))
  }, [items])

  const avgConfidence = useMemo(() => {
    const withScore = items.filter(i => i.confidence_score != null)
    if (!withScore.length) return null
    return Math.round(withScore.reduce((s, i) => s + i.confidence_score, 0) / withScore.length * 100)
  }, [items])

  const pipelinesWithScans = stats
    ? ['image', 'video', 'audio'].filter(t => (stats.by_media_type?.[t] ?? 0) > 0).length
    : 0

  return (
    <div>
      <Eyebrow>/ 01 · Analytics</Eyebrow>
      <h1 className="font-display font-extrabold text-3xl md:text-5xl tracking-tight text-snow mb-2">
        What the ledger shows
      </h1>
      <p className="text-soft text-sm mb-6 max-w-lg">
        Charts and totals derived only from scan records in the database.
      </p>
      <div className="mb-10"><HeuristicNotice /></div>

      {error && (
        <div className="mb-6 rounded-xl px-4 py-3 text-sm border" style={{ background: 'rgba(255,77,106,0.1)', borderColor: 'rgba(255,77,106,0.3)', color: '#FF4D6A' }}>
          {error}
        </div>
      )}

      {!loading && items.length === 0 && (
        <Card className="py-16 text-center text-soft text-sm">No scan data yet. Run a few scans to populate analytics.</Card>
      )}

      {items.length > 0 && (
        <>
          <div className="grid grid-cols-3 gap-6 md:gap-8 border-t border-stroke pt-8 mb-10">
            <Stat label="Total records" value={items.length} color="#3DFF9A" />
            <Stat label="Avg risk score" value={avgConfidence != null ? `${avgConfidence}%` : '—'} color="#F3F5F9" />
            <Stat label="Pipelines with scans" value={pipelinesWithScans} color="#3DFF9A" />
          </div>

          <div className="grid lg:grid-cols-3 gap-5">
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="lg:col-span-2">
              <Card className="p-6">
                <p className="text-sm font-semibold text-snow mb-4">Scans over time</p>
                <ResponsiveContainer width="100%" height={220}>
                  <AreaChart data={scansByDay}>
                    <defs>
                      <linearGradient id="scanGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3DFF9A" stopOpacity={0.35} />
                        <stop offset="100%" stopColor="#3DFF9A" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#262D3B" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="day" tick={{ fill: '#8B94A8', fontSize: 11 }} axisLine={{ stroke: '#262D3B' }} tickLine={false} />
                    <YAxis allowDecimals={false} tick={{ fill: '#8B94A8', fontSize: 11 }} axisLine={false} tickLine={false} width={28} />
                    <Tooltip contentStyle={tipStyle()} />
                    <Area type="monotone" dataKey="count" stroke="#3DFF9A" strokeWidth={2} fill="url(#scanGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
              <Card className="p-6 h-full">
                <p className="text-sm font-semibold text-snow mb-4">Verdict mix</p>
                <ResponsiveContainer width="100%" height={180}>
                  <PieChart>
                    <Pie data={verdictPie} dataKey="value" nameKey="name" innerRadius={48} outerRadius={72} paddingAngle={3}>
                      {verdictPie.map(entry => <Cell key={entry.name} fill={COLORS[entry.name]} />)}
                    </Pie>
                    <Tooltip contentStyle={tipStyle()} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex justify-center gap-4 mt-2 flex-wrap">
                  {verdictPie.map(e => (
                    <div key={e.name} className="flex items-center gap-1.5 text-[11px] text-soft">
                      <span className="w-2 h-2 rounded-sm" style={{ background: COLORS[e.name] }} />
                      {e.name} ({e.value})
                    </div>
                  ))}
                </div>
              </Card>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="lg:col-span-3">
              <Card className="p-6">
                <p className="text-sm font-semibold text-snow mb-4">Media type distribution</p>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={mediaBar}>
                    <CartesianGrid stroke="#262D3B" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="type" tick={{ fill: '#8B94A8', fontSize: 11 }} axisLine={{ stroke: '#262D3B' }} tickLine={false} />
                    <YAxis allowDecimals={false} tick={{ fill: '#8B94A8', fontSize: 11 }} axisLine={false} tickLine={false} width={28} />
                    <Tooltip contentStyle={tipStyle()} />
                    <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                      {mediaBar.map(e => <Cell key={e.type} fill={MEDIA_COLORS[e.type]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </motion.div>
          </div>
        </>
      )}
    </div>
  )
}
