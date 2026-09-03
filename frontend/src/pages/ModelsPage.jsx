import { Activity, Cpu, Globe, Radar, Sparkles } from 'lucide-react'
import Panel from '../components/Panel'
import StubBadge from '../components/StubBadge'

const models = [
  {
    name: 'EfficientNet-B4',
    architecture: 'Convolutional Neural Network',
    purpose: 'Image authenticity classification',
    status: 'Online',
    description: 'A compact visual backbone tuned for artifact and compression analysis in images.',
  },
  {
    name: 'XceptionNet',
    architecture: 'Depthwise Separable CNN',
    purpose: 'Video frame anomaly detection',
    status: 'Online',
    description: 'Used for detecting frame inconsistencies, face irregularities, and temporal anomalies.',
  },
  {
    name: 'wav2vec2',
    architecture: 'Self-supervised audio transformer',
    purpose: 'Audio authenticity scoring',
    status: 'Online',
    description: 'Learns high-level audio embeddings to identify synthetic or manipulated speech patterns.',
  },
]

export default function ModelsPage() {
  return (
    <div className="space-y-6">
      <Panel className="border border-cyan-400/20 bg-slate-900/70 p-6">
        <p className="text-[11px] font-semibold uppercase tracking-[0.32em] text-cyan-300">Model suite</p>
        <h1 className="mt-2 font-display text-3xl font-semibold text-white">Production-grade detection models</h1>
        <p className="mt-2 max-w-2xl text-sm leading-7 text-slate-400">Each model is deployed as part of a multi-stage reasoning pipeline for image, video, and audio forensic analysis.</p>
        <div className="mt-4">
          <StubBadge />
        </div>
      </Panel>

      <div className="grid gap-6 xl:grid-cols-3">
        {models.map((model) => (
          <Panel key={model.name} className="p-6">
            <div className="flex items-center justify-between">
              <div className="rounded-2xl border border-cyan-400/20 bg-cyan-500/10 p-2.5 text-cyan-300">
                <Cpu className="h-5 w-5" />
              </div>
              <span className="rounded-full border border-emerald-400/20 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-300">{model.status}</span>
            </div>
            <h2 className="mt-5 font-display text-xl font-semibold text-white">{model.name}</h2>
            <p className="mt-2 text-sm leading-7 text-slate-400">{model.description}</p>
            <div className="mt-5 grid gap-3 text-sm text-slate-400">
              <div className="flex items-center justify-between"><span>Architecture</span><span className="text-slate-100">{model.architecture}</span></div>
              <div className="flex items-center justify-between"><span>Purpose</span><span className="text-slate-100">{model.purpose}</span></div>
              <div className="flex items-center justify-between"><span>Status</span><span className="text-slate-100">{model.status}</span></div>
            </div>
            <button className="mt-5 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-300 transition-colors hover:bg-white/10">
              <Globe className="h-4 w-4" />
              Research paper
            </button>
          </Panel>
        ))}
      </div>
    </div>
  )
}
