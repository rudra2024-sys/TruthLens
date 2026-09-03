import { useEffect, useMemo, useState } from 'react'
import { BarChart3, Clock3, Cpu, ImageIcon, Music, Video } from 'lucide-react'
import { getHistory, getStats } from '../api/client'
import Panel from '../components/Panel'

export default function AnalyticsPage() {
  const [stats, setStats] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getStats(), getHistory(20)])
      .then(([statsRes, historyRes]) => {
        setStats(statsRes.data)
        setHistory(historyRes.data)
      })
      .catch(() => {
        setStats(null)
        setHistory([])
      })
      .finally(() => setLoading(false))
  }, [])

  const weeklyBars = useMemo(() => {
    const buckets = Array.from({ length: 7 }, (_, index) => ({ label: `D${index + 1}`, count: 0 }))
    history.forEach((item) => {
      const day = new Date(item.uploaded_at).getDay()
      buckets[day % 7].count += 1
    })
    return buckets.map((bucket) => ({ ...bucket, height: Math.max(22, bucket.count * 18 + 28) }))
  }, [history])

  const trend = useMemo(() => {
    const values = history.slice(0, 8).map((item) => Math.max(0, Math.round((item.confidence_score || 0) * 100)))
    return values.length ? values : []
  }, [history])

  return (
    <div className="space-y-6">
      <Panel className="border border-cyan-400/20 bg-slate-900/70 p-6">
        <p className="text-[11px] font-semibold uppercase tracking-[0.32em] text-cyan-300">Analytics</p>
        <h1 className="mt-2 font-display text-3xl font-semibold text-white">Detection intelligence overview</h1>
        <p className="mt-2 max-w-2xl text-sm leading-7 text-slate-400">Monitor weekly scan volume, confidence distribution, model performance, and the health of the forensic pipeline.</p>
      </Panel>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {[
          ['Total scans', loading ? '—' : (stats?.total_scans ?? 0).toLocaleString(), 'Live volume from the database'],
          ['Weekly scans', loading ? '—' : history.length.toLocaleString(), 'Recent activity from history'],
          ['Flagged fake', loading ? '—' : (stats?.fake_count ?? 0).toLocaleString(), 'Verdicts recorded as fake'],
          ['Verified real', loading ? '—' : (stats?.real_count ?? 0).toLocaleString(), 'Verdicts recorded as real'],
        ].map(([label, value, footer]) => (
          <Panel key={label} className="p-5">
            <p className="text-sm text-slate-400">{label}</p>
            <p className="mt-3 font-display text-3xl font-semibold text-white">{value}</p>
            <p className="mt-2 text-xs text-slate-500">{footer}</p>
          </Panel>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Panel className="p-6">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.32em] text-slate-500">Detection trends</p>
              <h2 className="mt-1 font-display text-xl font-semibold text-white">Weekly detection momentum</h2>
            </div>
            <div className="rounded-full border border-emerald-400/20 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-300">Live history</div>
          </div>
          <div className="flex h-56 items-end gap-3 rounded-[24px] border border-white/10 bg-slate-950/70 p-4">
            {weeklyBars.map((bar) => (
              <div key={bar.label} className="flex flex-1 flex-col items-center gap-2">
                <div className="w-full rounded-t-2xl bg-gradient-to-t from-cyan-500 to-violet-500" style={{ height: `${Math.max(10, bar.height)}%` }} />
                <span className="text-[10px] uppercase tracking-[0.3em] text-slate-500">{bar.label}</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel className="p-6">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.32em] text-slate-500">Verdict balance</p>
              <h2 className="mt-1 font-display text-xl font-semibold text-white">Recorded outcomes</h2>
            </div>
            <Cpu className="h-5 w-5 text-violet-300" />
          </div>
          <div className="space-y-4">
            {[
              ['Fake', stats?.fake_count ?? 0],
              ['Real', stats?.real_count ?? 0],
              ['Uncertain', stats?.uncertain_count ?? 0],
            ].map(([name, count]) => (
              <div key={name}>
                <div className="mb-2 flex items-center justify-between text-sm text-slate-400">
                  <span>{name}</span>
                  <span className="text-slate-100">{count}</span>
                </div>
                <div className="h-2 rounded-full bg-white/10">
                  <div className="h-2 rounded-full bg-gradient-to-r from-cyan-400 to-violet-500" style={{ width: `${Math.max(6, Math.min(100, count ? (count / Math.max(stats?.total_scans || 1, 1)) * 100 : 0))}%` }} />
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
        <Panel className="p-6">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.32em] text-slate-500">Media distribution</p>
              <h2 className="mt-1 font-display text-xl font-semibold text-white">By modality</h2>
            </div>
            <BarChart3 className="h-5 w-5 text-cyan-300" />
          </div>
          <div className="space-y-4">
            {[
              ['Image', stats?.by_media_type?.image ? Math.round((stats.by_media_type.image / Math.max(stats.total_scans, 1)) * 100) : 0, ImageIcon],
              ['Video', stats?.by_media_type?.video ? Math.round((stats.by_media_type.video / Math.max(stats.total_scans, 1)) * 100) : 0, Video],
              ['Audio', stats?.by_media_type?.audio ? Math.round((stats.by_media_type.audio / Math.max(stats.total_scans, 1)) * 100) : 0, Music],
            ].map(([name, value, Icon]) => (
              <div key={name} className="rounded-2xl border border-white/10 bg-slate-950/70 p-4">
                <div className="mb-2 flex items-center justify-between text-sm text-slate-400">
                  <span className="flex items-center gap-2"><Icon className="h-4 w-4" /> {name}</span>
                  <span className="text-slate-100">{value}%</span>
                </div>
                <div className="h-2 rounded-full bg-white/10">
                  <div className="h-2 rounded-full bg-gradient-to-r from-emerald-400 to-cyan-400" style={{ width: `${value}%` }} />
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel className="p-6">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.32em] text-slate-500">Processing speed</p>
              <h2 className="mt-1 font-display text-xl font-semibold text-white">Inference latency trend</h2>
            </div>
            <Clock3 className="h-5 w-5 text-amber-300" />
          </div>
          <div className="flex h-56 items-end gap-3 rounded-[24px] border border-white/10 bg-slate-950/70 p-4">
            {trend.length ? trend.map((height, index) => (
              <div key={index} className="flex flex-1 flex-col items-center gap-2">
                <div className="w-full rounded-t-2xl bg-gradient-to-t from-amber-500 to-rose-500" style={{ height: `${Math.max(10, height)}%` }} />
                <span className="text-[10px] uppercase tracking-[0.3em] text-slate-500">D{index + 1}</span>
              </div>
            )) : <div className="flex w-full items-center justify-center text-sm text-slate-400">No confidence history available yet.</div>}
          </div>
        </Panel>
      </div>
    </div>
  )
}
