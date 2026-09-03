import { Info } from 'lucide-react'

export default function HeuristicNotice({ compact = false }) {
  if (compact) {
    return (
      <span
        className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-md"
        style={{ background: 'rgba(255,176,32,0.14)', color: '#FFB020' }}
      >
        <Info className="w-2.5 h-2.5" /> heuristics
      </span>
    )
  }
  return (
    <div
      className="flex items-start gap-3 rounded-xl px-4 py-3 border"
      style={{ background: 'rgba(255,176,32,0.08)', borderColor: 'rgba(255,176,32,0.25)' }}
    >
      <Info className="w-4 h-4 shrink-0 mt-0.5" style={{ color: '#FFB020' }} />
      <div>
        <p className="text-xs font-semibold" style={{ color: '#FFB020' }}>
          Content heuristics — not a trained DNN
        </p>
        <p className="text-[11px] text-soft mt-0.5 leading-relaxed">
          Scores come from this file&apos;s bytes (frequency, noise, entropy) — not EfficientNet, Xception, or wav2vec2.
        </p>
      </div>
    </div>
  )
}
