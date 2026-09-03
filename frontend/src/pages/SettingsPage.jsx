import { Bell, Fingerprint, MonitorPlay, ShieldCheck, Sparkles } from 'lucide-react'
import Panel from '../components/Panel'

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <Panel className="border border-cyan-400/20 bg-slate-900/70 p-6">
        <p className="text-[11px] font-semibold uppercase tracking-[0.32em] text-cyan-300">Settings</p>
        <h1 className="mt-2 font-display text-3xl font-semibold text-white">Platform configuration</h1>
        <p className="mt-2 max-w-2xl text-sm leading-7 text-slate-400">Tune alerting, model preferences, and deployment behavior for your forensic review environment.</p>
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel className="p-6">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl border border-cyan-400/20 bg-cyan-500/10 p-2.5 text-cyan-300"><Bell className="h-5 w-5" /></div>
            <div>
              <h2 className="font-display text-xl font-semibold text-white">Alert preferences</h2>
              <p className="text-sm text-slate-400">Configure how the platform surfaces suspicious detections.</p>
            </div>
          </div>
          <div className="mt-6 space-y-3 text-sm text-slate-400">
            <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3"><span>High-risk detections</span><span className="text-emerald-300">Enabled</span></div>
            <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3"><span>Daily summary email</span><span className="text-emerald-300">Enabled</span></div>
            <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3"><span>Developer mode logs</span><span className="text-slate-100">Available</span></div>
          </div>
        </Panel>

        <Panel className="p-6">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl border border-violet-400/20 bg-violet-500/10 p-2.5 text-violet-300"><MonitorPlay className="h-5 w-5" /></div>
            <div>
              <h2 className="font-display text-xl font-semibold text-white">Inference defaults</h2>
              <p className="text-sm text-slate-400">Tune how scans are prioritized and processed.</p>
            </div>
          </div>
          <div className="mt-6 space-y-3 text-sm text-slate-400">
            <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3"><span>Auto-run all models</span><span className="text-emerald-300">On</span></div>
            <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3"><span>Confidence threshold</span><span className="text-slate-100">0.92</span></div>
            <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3"><span>Report format</span><span className="text-slate-100">PDF + JSON</span></div>
          </div>
        </Panel>
      </div>
    </div>
  )
}
