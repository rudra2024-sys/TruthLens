import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { AlertCircle, Download, ArrowLeft, ChevronDown } from 'lucide-react'
import { getDetectionResult, getUpload, downloadReport } from './api/client'
import VerdictSeal from './components/VerdictSeal'
import VerdictBadge from './components/VerdictBadge'
import ConfidenceGauge from './components/ConfidenceGauge'
import ScanFrame from './components/ScanFrame'
import CaseTag from './components/CaseTag'
import Reveal from './components/Reveal'
import LoadingState from './components/LoadingState'
import Disclosure from './components/Disclosure'
import ExplanationPanel from './components/ExplanationPanel'
import ProvenancePanel from './components/ProvenancePanel'
import FeedbackPanel from './components/FeedbackPanel'
import { getVerdictInfo } from './lib/verdict'
import { stationTag } from './lib/stations'

/**
 * Evidence rows are labeled for what the backend actually computed, per
 * modality — never a specific model name the pipeline doesn't use. See
 * CLAUDE.md's Phase 3 audit: video is a byte-entropy heuristic, and the
 * audio DB columns (wav2vec_score/lcnn_score) are legacy names that the
 * active AASIST detector repurposes, not a wav2vec2/LCNN pipeline.
 */
/**
 * Low-confidence transparency note (added 2026-10-03, see CLAUDE.md section 21): when the face detector found
 * a usable face in fewer than half of Video Model v1's 16 sampled frames, most frames were scored from a
 * generic centre crop instead of a face close-up — the model saw less of what it was trained to look at.
 * Shown next to the evidence rows, not folded into the verdict itself.
 */
function LowFaceConfidenceNote({ result }) {
  const a = result.video_analysis
  if (!a || a.frames_with_face == null || !a.frames_analyzed) return null
  if (a.frames_with_face >= a.frames_analyzed / 2) return null
  return (
    <div role="note" className="flex gap-3 border border-verdictCaution/60 bg-verdictCaution/10 rounded-[4px] p-4 mb-4">
      <AlertCircle size={18} strokeWidth={1.75} className="text-verdictCaution shrink-0 mt-[2px]" aria-hidden="true" />
      <div>
        <p className="text-[13px] font-medium text-bone mb-1">Low face-detection confidence</p>
        <p className="text-[13px] text-bone-dim">
          The face detector found a usable face in only {a.frames_with_face} of {a.frames_analyzed} sampled
          frames. Most frames were scored from a generic centre crop instead of a face close-up — treat this
          verdict with extra caution.
        </p>
      </div>
    </div>
  )
}

function EvidenceRows({ result }) {
  const rows = []

  if (result.image_analysis) {
    const a = result.image_analysis
    if (a.fake_probability != null) rows.push(['FAKE probability', `${(a.fake_probability * 100).toFixed(1)}%`])
    if (a.real_probability != null) rows.push(['REAL probability', `${(a.real_probability * 100).toFixed(1)}%`])
    if (a.convnext_fake_probability != null) rows.push(['ConvNeXt-Tiny sub-score (FAKE)', `${(a.convnext_fake_probability * 100).toFixed(1)}%`])
    if (a.clip_fake_probability != null) rows.push(['CLIP second-opinion sub-score (FAKE)', `${(a.clip_fake_probability * 100).toFixed(1)}%`])
    if (a.efficientnet_score != null) rows.push(['Noise residual score (legacy)', `${(a.efficientnet_score * 100).toFixed(1)}%`])
    if (a.fft_score != null) rows.push(['FFT frequency score (legacy)', `${(a.fft_score * 100).toFixed(1)}%`])
  }
  if (result.video_analysis) {
    const a = result.video_analysis
    rows.push(['Video fake probability', `${(a.xception_score * 100).toFixed(1)}%`])
    if (a.face_voice_sync != null) rows.push(['Consistency score (heuristic)', `${(a.face_voice_sync * 100).toFixed(1)}%`])
    rows.push(['Frames analyzed', String(a.frames_analyzed)])
    if (a.frames_with_face != null) rows.push(['Frames with a detected face', `${a.frames_with_face} / ${a.frames_analyzed}`])
  }
  if (result.audio_analysis) {
    const a = result.audio_analysis
    rows.push(['AASIST spoof probability', `${(a.wav2vec_score * 100).toFixed(1)}%`])
    rows.push(['AASIST score variability', `${(a.lcnn_score * 100).toFixed(1)}%`])
  }

  if (rows.length === 0) return null

  return (
    <div className="space-y-0">
      {rows.map(([label, value], i) => (
        <div key={label} className="flex items-center gap-4 py-3 border-b border-line last:border-b-0">
          <span className="tl-figure text-[10px] text-bone-faint w-6">{String(i + 1).padStart(2, '0')}</span>
          <span className="text-[13px] text-bone-dim flex-1">{label}</span>
          <span className="tl-figure text-[13px] text-bone">{value}</span>
        </div>
      ))}
    </div>
  )
}

export default function ReportDetail() {
  const { id: uploadId } = useParams()
  const [result, setResult] = useState(null)
  const [upload, setUpload] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [logOpen, setLogOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    Promise.all([
      getDetectionResult(uploadId),
      getUpload(uploadId).catch(() => null), // filename is a nice-to-have, not fatal if it fails
    ])
      .then(([resultRes, uploadRes]) => {
        if (cancelled) return
        setResult(resultRes.data)
        setUpload(uploadRes?.data ?? null)
      })
      .catch((err) => { if (!cancelled) setError(err.message) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [uploadId])

  if (loading) {
    return <LoadingState label="Opening case file" layout="screen-offset" />
  }

  if (error || !result) {
    return (
      <div className="min-h-screen bg-ground pt-[72px] flex items-center justify-center px-6">
        <div className="text-center max-w-[420px]">
          <AlertCircle size={32} strokeWidth={1.25} className="text-verdictDanger mb-5 mx-auto" />
          <h1 className="font-serif text-[28px] text-bone mb-3">Report unavailable</h1>
          <p className="text-[15px] text-bone-dim mb-8">{error || 'No detection result found for this record.'}</p>
          <Link to="/history" className="inline-flex items-center gap-2 text-[13px] font-medium tracking-[0.06em] text-brass hover:text-bone transition-colors duration-300 link-underline">
            <ArrowLeft size={14} strokeWidth={1.75} /> Back to history
          </Link>
        </div>
      </div>
    )
  }

  const info = getVerdictInfo(result.verdict)

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />
      <div className="max-w-[900px] mx-auto px-6 py-16 md:py-20">
        <Reveal>
          <Link
            to="/history"
            className="inline-flex items-center gap-2 mb-6 text-[13px] font-medium tracking-[0.06em] text-bone-dim hover:text-brass transition-colors duration-300 link-underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass rounded-[2px]"
          >
            <ArrowLeft size={14} strokeWidth={1.75} /> Back to Case Archive
          </Link>
          <div className="flex items-center justify-between mb-3">
            <p className="tl-hud-label">{stationTag(5)} — Forensic Case File</p>
            <CaseTag id={result.upload_id} />
          </div>
          <h1 className="font-serif text-[36px] md:text-[42px] text-bone mb-1 break-all">
            {upload?.file_name || 'Detection Result'}
          </h1>
          <p className="text-[13px] text-bone-faint tl-figure">
            {new Date(result.detected_at).toLocaleString()}
          </p>
        </Reveal>

        <Reveal delay={0.1}>
          <div className="relative bg-panel border border-line rounded-[6px] p-8 md:p-12 mt-10 overflow-hidden">
            <ScanFrame gap={20} armSize={14} />

            {/* Verdict + Confidence — the headline of the document */}
            <div className="flex flex-col sm:flex-row items-center gap-10 mb-12 pb-12 border-b border-line">
              <VerdictSeal verdict={result.verdict} size={96} />
              <div className="flex-1 text-center sm:text-left">
                <p className="tl-hud-label mb-2">Verdict</p>
                <h2 className={`font-serif text-[32px] md:text-[36px] mb-2 ${info.textClass}`}>{info.label}</h2>
                <p className="text-[14px] text-bone-dim">{info.caption}</p>
              </div>
              <ConfidenceGauge value={result.confidence_score} accent={info.accent} size={128} />
            </div>

            {/* Specimen information */}
            <motion.div
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3, duration: 0.5 }}
              className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-10"
            >
              <div>
                <p className="tl-hud-label !text-[9px] mb-1">Media Type</p>
                <p className="tl-figure text-[13px] text-bone uppercase">{upload?.media_type || '—'}</p>
              </div>
              <div>
                <p className="tl-hud-label !text-[9px] mb-1">File Size</p>
                <p className="tl-figure text-[13px] text-bone">{upload?.file_size_kb ? `${upload.file_size_kb.toFixed(1)} KB` : '—'}</p>
              </div>
              <div>
                <p className="tl-hud-label !text-[9px] mb-1">Model</p>
                <p className="tl-figure text-[13px] text-bone truncate" title={result.model_used}>{result.model_used}</p>
              </div>
              <div>
                <p className="tl-hud-label !text-[9px] mb-1">Processing Time</p>
                <p className="tl-figure text-[13px] text-bone">{Math.round(result.processing_time_ms)} ms</p>
              </div>
            </motion.div>

            {/* Evidence — progressive disclosure: the verdict above is the
                conclusion of the examination, this is the detail behind it,
                revealed on request rather than dumped alongside it. */}
            {(result.image_analysis || result.video_analysis || result.audio_analysis) && (
              <motion.div
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4, duration: 0.5 }}
                className="mb-10"
              >
                <button
                  type="button"
                  onClick={() => setLogOpen((o) => !o)}
                  aria-expanded={logOpen}
                  aria-controls="inspection-log-panel"
                  className="w-full flex items-center justify-between gap-3 py-1 tl-hud-label !text-bone hover:!text-brass transition-colors duration-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass rounded-[2px]"
                >
                  <span>{logOpen ? 'Hide Inspection Log' : 'View Inspection Log'}</span>
                  <ChevronDown
                    size={13}
                    strokeWidth={1.75}
                    style={{ transform: logOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.3s cubic-bezier(0.22,1,0.36,1)' }}
                  />
                </button>
                <Disclosure open={logOpen} id="inspection-log-panel">
                  <div className="pt-4">
                    <LowFaceConfidenceNote result={result} />
                    <EvidenceRows result={result} />
                  </div>
                </Disclosure>
              </motion.div>
            )}

            {/* Provenance — supplementary evidence (C2PA credentials / embedded metadata); cheap, so automatic. */}
            <ProvenancePanel
              uploadId={result.upload_id}
              mediaType={upload?.media_type || (result.image_analysis ? 'image' : result.video_analysis ? 'video' : null)}
              className="mb-10"
            />

            {/* Explainability — fetched only when opened (backend recomputes it, ~1-3 s). */}
            <ExplanationPanel
              uploadId={result.upload_id}
              mediaType={upload?.media_type || (result.image_analysis ? 'image' : result.video_analysis ? 'video' : null)}
              className="mb-10"
            />

            <FeedbackPanel uploadId={result.upload_id} className="mb-10" />

            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5, duration: 0.5 }}>
              <button
                type="button"
                onClick={() => downloadReport(result.upload_id)}
                className="inline-flex items-center gap-2 px-6 py-3 bg-brass text-ground rounded-[3px] text-[12px] font-medium tracking-[0.08em] uppercase btn-lift"
              >
                <Download size={15} strokeWidth={1.75} /> Download PDF Report
              </button>
            </motion.div>
          </div>
        </Reveal>
      </div>
    </div>
  )
}
