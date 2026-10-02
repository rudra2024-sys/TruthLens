import React, { useEffect, useState } from 'react'
import { ChevronDown, AlertTriangle } from 'lucide-react'
import { getProvenance } from '../api/client'
import Disclosure from './Disclosure'

// Strength -> colour of the marker dot. Only explicit declarations are drawn as "strong".
const DOT = {
  strong: 'bg-verdictDanger',
  moderate: 'bg-verdictCaution',
  weak: 'bg-bone-faint',
  none: 'bg-line-strong',
}

function Facts({ rows }) {
  const shown = rows.filter(([, v]) => v !== null && v !== undefined && v !== '')
  if (shown.length === 0) return null
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 mt-4">
      {shown.map(([k, v]) => (
        <React.Fragment key={k}>
          <dt className="tl-hud-label !text-[9px] py-[2px]">{k}</dt>
          <dd className="tl-figure text-[12px] text-bone break-words">{String(v)}</dd>
        </React.Fragment>
      ))}
    </dl>
  )
}

const yn = (v) => (v === true ? 'yes' : v === false ? 'no' : null)

/**
 * Provenance evidence (C2PA Content Credentials, embedded metadata, ELA visual aid). Supplementary to the
 * verdict — it never changes it. Fetched automatically (cheap: no model runs). A conflict between the models'
 * verdict and an embedded AI declaration is shown up front; everything else sits behind a disclosure. Renders
 * nothing while loading or when the backend has nothing to say, so it can never block the result screen.
 */
export default function ProvenancePanel({ uploadId, mediaType, className = '' }) {
  const [data, setData] = useState(null)
  const [open, setOpen] = useState(false)
  const eligible = mediaType === 'image' || mediaType === 'video' || mediaType === 'audio'

  useEffect(() => {
    if (!eligible || !uploadId) return undefined
    let cancelled = false
    setData(null)
    getProvenance(uploadId)
      .then((res) => { if (!cancelled) setData(res.data) })
      .catch(() => { if (!cancelled) setData(null) }) // supplementary: fail silently rather than disturb the verdict view
    return () => { cancelled = true }
  }, [uploadId, eligible])

  if (!eligible || !data || !data.available) return null

  const panelId = `provenance-panel-${uploadId}`
  const c2pa = data.c2pa
  const meta = data.metadata
  const declared = data.level === 'declared_ai'

  return (
    <div className={className}>
      {data.conflict_note && (
        <div role="note" className="flex gap-3 border border-verdictCaution/60 bg-verdictCaution/10 rounded-[4px] p-4 mb-4">
          <AlertTriangle size={18} strokeWidth={1.75} className="text-verdictCaution shrink-0 mt-[2px]" aria-hidden="true" />
          <div>
            <p className="text-[13px] font-medium text-bone mb-1">Provenance conflicts with the detection verdict</p>
            <p className="text-[13px] text-bone-dim">{data.conflict_note}</p>
          </div>
        </div>
      )}

      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={panelId}
        className="w-full flex items-center justify-between gap-3 py-1 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass rounded-[2px] group"
      >
        <span className="flex flex-col gap-1">
          <span className="tl-hud-label !text-bone group-hover:!text-brass transition-colors duration-300">
            {open ? 'Hide Provenance' : 'Provenance & Metadata'}
          </span>
          <span className={`text-[13px] ${declared ? 'text-verdictCaution' : 'text-bone-dim'}`}>{data.headline}</span>
        </span>
        <ChevronDown
          size={13}
          strokeWidth={1.75}
          style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.3s cubic-bezier(0.22,1,0.36,1)' }}
        />
      </button>

      <Disclosure open={open} id={panelId}>
        <div className="pt-4">
          <ul className="space-y-4">
            {data.signals.map((s, i) => (
              <li key={`${s.title}-${i}`} className="flex gap-3">
                <span className={`mt-[7px] h-[8px] w-[8px] rounded-full shrink-0 ${DOT[s.strength] || DOT.none}`} aria-hidden="true" />
                <div>
                  <p className="text-[13px] text-bone">{s.title}</p>
                  {s.detail && <p className="text-[12px] text-bone-dim mt-[2px] max-w-[640px]">{s.detail}</p>}
                </div>
              </li>
            ))}
          </ul>

          {c2pa?.present && (
            <Facts
              rows={[
                ['Signature valid', yn(c2pa.signature_valid)],
                ['File unchanged since signing', yn(c2pa.content_intact)],
                ['Signer on C2PA trust list', yn(c2pa.signer_trusted)],
                ['Issuer', c2pa.issuer],
                ['Generator', c2pa.generator],
                ['Signed', c2pa.signed_at ? c2pa.signed_at.slice(0, 10) : null],
                ['Validation state', c2pa.validation_state],
              ]}
            />
          )}
          {meta && (meta.has_exif || meta.ai_param_fields?.length > 0) && (
            <Facts
              rows={[
                ['Camera', [meta.camera_make, meta.camera_model].filter(Boolean).join(' ') || null],
                ['Lens', meta.lens],
                ['Taken', meta.taken_at],
                ['Software', meta.software],
                ['Embedded AI fields', meta.ai_param_fields?.length ? meta.ai_param_fields.join(', ') : null],
              ]}
            />
          )}
          {data.container && (
            <Facts
              rows={[
                ['Resolution', data.container.width ? `${data.container.width} × ${data.container.height}` : null],
                ['Frame rate', data.container.fps ? `${data.container.fps.toFixed(1)} fps` : null],
                ['Encoder tag', data.container.encoder_tag],
                ['Duration', data.container.duration_s ? `${data.container.duration_s.toFixed(1)}s` : null],
                ['Sample rate', data.container.sample_rate ? `${data.container.sample_rate} Hz` : null],
                ['Channels', data.container.channels],
                ['Codec', data.container.codec],
              ]}
            />
          )}
          {data.watermark?.applicable && (
            <Facts
              rows={[
                ['SD/SDXL watermark bit match', typeof data.watermark.bit_match === 'number'
                  ? `${Math.round(data.watermark.bit_match * 48)}/48${data.watermark.present ? ' (exact match)' : ''}`
                  : null],
                ['SD/SDXL watermark best frame match', typeof data.watermark.best_bit_match === 'number'
                  ? `${Math.round(data.watermark.best_bit_match * 48)}/48${data.watermark.present ? ' (exact match)' : ''}`
                  : null],
                ['Frames checked for watermark', typeof data.watermark.frames_checked === 'number'
                  ? `${data.watermark.frames_matched} / ${data.watermark.frames_checked} matched`
                  : null],
              ]}
            />
          )}

          {data.ela?.applicable && data.ela.heatmap && (
            <figure className="mt-6 max-w-[360px]">
              <img src={data.ela.heatmap} alt="Error level analysis map" className="w-full rounded-[3px] border border-line" />
              <figcaption className="tl-hud-label !text-[9px] mt-2">
                Error level analysis — visual aid only, not a detector
              </figcaption>
            </figure>
          )}

          {data.caveat && <p className="text-[12px] text-bone-faint mt-6 max-w-[640px]">{data.caveat}</p>}
        </div>
      </Disclosure>
    </div>
  )
}
