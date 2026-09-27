import React, { useState } from 'react'
import { ChevronDown } from 'lucide-react'
import { getExplanation } from '../api/client'
import Disclosure from './Disclosure'

const pct = (v) => `${(v * 100).toFixed(1)}%`

/** Legend for the heatmap colours: cool = little influence on the FAKE score, warm = strong influence. */
function HeatLegend() {
  return (
    <div className="flex items-center gap-3 mt-3" aria-hidden="true">
      <span className="tl-figure text-[10px] text-bone-faint">low influence</span>
      <div
        className="h-[6px] flex-1 max-w-[180px] rounded-[2px]"
        style={{ background: 'linear-gradient(90deg, rgb(0,0,143), rgb(0,128,255), rgb(0,255,128), rgb(255,255,0), rgb(255,0,0))' }}
      />
      <span className="tl-figure text-[10px] text-bone-faint">high influence</span>
    </div>
  )
}

function ImageExplanation({ data }) {
  return (
    <div>
      <div className="grid sm:grid-cols-2 gap-6">
        <figure>
          <img
            src={data.original}
            alt="Submitted image"
            className="w-full max-h-[420px] object-contain rounded-[3px] border border-line bg-ground"
          />
          <figcaption className="tl-hud-label !text-[9px] mt-2">Submitted image</figcaption>
        </figure>
        <figure>
          <img
            src={data.heatmap}
            alt="Grad-CAM heatmap over the submitted image"
            className="w-full max-h-[420px] object-contain rounded-[3px] border border-line bg-ground"
          />
          <figcaption className="tl-hud-label !text-[9px] mt-2">
            Grad-CAM — ConvNeXt-Tiny FAKE probability {pct(data.fake_probability)}
          </figcaption>
        </figure>
      </div>
      <HeatLegend />
    </div>
  )
}

function VideoExplanation({ data }) {
  const frames = data.frames || []
  const hot = frames.filter((f) => f.heatmap).sort((a, b) => b.probability - a.probability)
  return (
    <div>
      <p className="text-[13px] text-bone-dim mb-4">
        Mean-logit probability <span className="tl-figure text-bone">{pct(data.mean_probability)}</span> vs decision
        threshold <span className="tl-figure text-bone">{data.threshold}</span> —{' '}
        <span className="tl-figure text-bone">{data.frames_above_threshold}</span> of {frames.length} frames
        individually above it.
      </p>

      {/* Per-frame FAKE probability: the model's own score for each of the 16 sampled frames. */}
      <div
        className="relative h-[120px] border border-line rounded-[3px] p-2 bg-ground"
        role="img"
        aria-label={`Per-frame fake probability, ${data.frames_above_threshold} of ${frames.length} frames above the threshold`}
      >
        <div className="relative h-full">
          <div
            className="absolute left-0 right-0 border-t border-dashed border-verdictCaution/80"
            style={{ bottom: `${data.threshold * 100}%` }}
          />
          <div className="flex items-end gap-[3px] h-full">
            {frames.map((f) => (
              <div
                key={f.order}
                className={f.probability >= data.threshold ? 'bg-verdictDanger' : 'bg-brass-deep'}
                style={{ height: `${Math.max(1, f.probability * 100)}%`, flex: 1 }}
                title={`frame ${f.order + 1}${f.timestamp_s != null ? ` @ ${f.timestamp_s.toFixed(1)}s` : ''}: ${pct(f.probability)} fake`}
              />
            ))}
          </div>
        </div>
      </div>
      <p className="tl-hud-label !text-[9px] mt-2">P(fake) per frame · dashed line = decision threshold</p>

      <p className="tl-hud-label !text-[9px] mt-6 mb-3">Frames as seen by the model</p>
      <div className="grid grid-cols-4 sm:grid-cols-8 gap-2">
        {frames.map((f) => (
          <figure key={f.order}>
            <img
              src={f.crop}
              alt={`Frame ${f.order + 1} as seen by the model`}
              className="w-full aspect-square object-cover rounded-[2px] border border-line"
            />
            <figcaption className="tl-figure text-[10px] text-bone-faint text-center mt-1">
              {Math.round(f.probability * 100)}%
            </figcaption>
          </figure>
        ))}
      </div>

      {hot.length > 0 && (
        <>
          <p className="tl-hud-label !text-[9px] mt-6 mb-3">Most suspicious frames — Grad-CAM</p>
          <div className="grid grid-cols-3 gap-3">
            {hot.map((f) => (
              <figure key={f.order}>
                <img
                  src={f.heatmap}
                  alt={`Grad-CAM heatmap for frame ${f.order + 1}`}
                  className="w-full aspect-square object-cover rounded-[3px] border border-line"
                />
                <figcaption className="tl-figure text-[10px] text-bone-faint mt-1">
                  frame {f.order + 1}
                  {f.timestamp_s != null ? ` · ${f.timestamp_s.toFixed(1)}s` : ''} · {Math.round(f.probability * 100)}% fake
                </figcaption>
              </figure>
            ))}
          </div>
          <HeatLegend />
        </>
      )}
    </div>
  )
}

/**
 * On-demand explainability panel. Nothing is fetched until the user opens it (the backend recomputes the
 * attribution with the deployed model, ~1-3 s), and everything shown comes from the model's real output — the
 * backend returns `available: false` with a reason instead of anything invented.
 */
export default function ExplanationPanel({ uploadId, mediaType, className = '' }) {
  const [open, setOpen] = useState(false)
  const [state, setState] = useState({ status: 'idle', data: null, error: null })

  if (mediaType !== 'image' && mediaType !== 'video') return null

  const toggle = () => {
    const next = !open
    setOpen(next)
    if (next && state.status === 'idle') {
      setState({ status: 'loading', data: null, error: null })
      getExplanation(uploadId)
        .then((res) => setState({ status: 'done', data: res.data, error: null }))
        .catch((err) => setState({ status: 'error', data: null, error: err.message }))
    }
  }

  const { status, data, error } = state
  const panelId = `explanation-panel-${uploadId}`

  return (
    <div className={className}>
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        aria-controls={panelId}
        className="w-full flex items-center justify-between gap-3 py-1 tl-hud-label !text-bone hover:!text-brass transition-colors duration-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass rounded-[2px]"
      >
        <span>{open ? 'Hide Explanation' : 'Why this verdict? — View Explanation'}</span>
        <ChevronDown
          size={13}
          strokeWidth={1.75}
          style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.3s cubic-bezier(0.22,1,0.36,1)' }}
        />
      </button>
      <Disclosure open={open} id={panelId}>
        <div className="pt-4" aria-live="polite">
          {status === 'loading' && <p className="text-[13px] text-bone-dim">Analysing the specimen…</p>}
          {status === 'error' && <p className="text-[13px] text-verdictDanger">{error}</p>}
          {status === 'done' && !data.available && (
            <p className="text-[13px] text-bone-dim">{data.reason || 'No explanation is available for this file.'}</p>
          )}
          {status === 'done' && data.available && (
            <>
              <p className="tl-hud-label !text-[9px] mb-4">{data.method}</p>
              {data.media_type === 'image' ? <ImageExplanation data={data} /> : <VideoExplanation data={data} />}
              {data.note && <p className="text-[12px] text-bone-faint mt-5 max-w-[640px]">{data.note}</p>}
            </>
          )}
        </div>
      </Disclosure>
    </div>
  )
}
