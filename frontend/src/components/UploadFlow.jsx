import React, { useState, useRef, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Image as ImageIcon, Video as VideoIcon, AudioLines, Upload as UploadIcon, Download, Search, ChevronDown, RotateCcw } from 'lucide-react'
import { uploadMedia, runDetection, downloadReport, startDetectionJob, getJob, cancelJob, getDetectionResult } from '../api/client'
import { ACCEPTED_INPUT_ACCEPT, ACCEPTED_FORMAT_CHIPS } from '../lib/acceptedFormats'
import VerdictBadge from './VerdictBadge'
import VerdictSeal from './VerdictSeal'
import ConfidenceGauge from './ConfidenceGauge'
import ScanFrame from './ScanFrame'
import ForensicProcess from './ForensicProcess'
import CaseTag from './CaseTag'
import PixelGridScan from './scan/PixelGridScan'
import WaveformScan from './scan/WaveformScan'
import FilmstripScan from './scan/FilmstripScan'
import Reveal from './Reveal'
import Disclosure from './Disclosure'
import SpecimenDossier from './SpecimenDossier'
import ExplanationPanel from './ExplanationPanel'
import ProvenancePanel from './ProvenancePanel'
import FeedbackPanel from './FeedbackPanel'
import { getVerdictInfo } from '../lib/verdict'

/**
 * The forensic intake desk — the primary instrument of the product. Shared
 * by Home's compact hero slot and the dedicated Verify page. `compact`
 * controls sizing for the two contexts; `showRecent` renders the
 * localStorage-backed "Recent" strip (a client convenience cache — History/
 * Reports remain the real API-backed source of truth) below it.
 */

const ArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 8H13" /><path d="M9 4L13 8L9 12" />
  </svg>
)

const kindOf = (type) => {
  if (!type) return 'other'
  if (type.startsWith('image')) return 'image'
  if (type.startsWith('video')) return 'video'
  if (type.startsWith('audio')) return 'audio'
  return 'other'
}

const getFileIcon = (type) => {
  const k = kindOf(type)
  if (k === 'image') return <ImageIcon size={18} strokeWidth={1.5} />
  if (k === 'video') return <VideoIcon size={18} strokeWidth={1.5} />
  if (k === 'audio') return <AudioLines size={18} strokeWidth={1.5} />
  return <UploadIcon size={18} strokeWidth={1.5} />
}

const fmtSize = (bytes) => {
  if (!bytes) return '—'
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

class JobCancelledError extends Error {}

const POLL_INTERVAL_MS = 600
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

/** Real progress of a background scan (video). The numbers come from the server's job, not from a timer. */
function JobProgress({ job, onCancel }) {
  const pct = Math.round((job.progress || 0) * 100)
  const queued = job.state === 'queued'
  return (
    <div className="w-full max-w-[260px]" role="status" aria-live="polite">
      <div className="flex items-center gap-3">
        <div
          className="flex-1 h-[3px] bg-line-strong rounded-full overflow-hidden"
          role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label="Analysis progress"
        >
          <div className="h-full bg-brass rounded-full transition-[width] duration-300" style={{ width: `${pct}%` }} />
        </div>
        <span className="tl-figure text-[11px] text-bone-dim w-9 text-right">{pct}%</span>
      </div>
      <p className="tl-figure text-[11px] text-bone-faint mt-2">
        {queued ? `Waiting in queue${job.queue_position ? ` — position ${job.queue_position}` : ''}` : job.stage}
      </p>
      <button
        type="button"
        onClick={onCancel}
        className="mt-3 text-[11px] font-medium tracking-[0.06em] text-bone-dim hover:text-bone transition-colors duration-300 link-underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass"
      >
        Cancel analysis
      </button>
    </div>
  )
}

const SCAN_LABEL = { image: 'Pixel Scan', video: 'Captured Frames', audio: 'Waveform', other: 'Specimen' }

const SpecimenPreview = ({ file, active }) => {
  const kind = kindOf(file?.type)
  return (
    <div className="relative w-full aspect-[4/3] bg-ground border border-line rounded-[4px] overflow-hidden">
      {kind === 'image' && <PixelGridScan file={file} active={active} />}
      {kind === 'video' && <FilmstripScan file={file} active={active} />}
      {kind === 'audio' && <WaveformScan file={file} active={active} />}
      {kind === 'other' && (
        <div className="w-full h-full flex items-center justify-center text-bone-faint">
          <UploadIcon size={28} strokeWidth={1.25} />
        </div>
      )}
    </div>
  )
}

export default function UploadFlow({ compact = false, showRecent = false, heading = null }) {
  const [isDragging, setIsDragging] = useState(false)
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [analyzing, setAnalyzing] = useState(false)
  const [detectDone, setDetectDone] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [recentFiles, setRecentFiles] = useState([])
  const [inspectOpen, setInspectOpen] = useState(false)
  const [job, setJob] = useState(null)          // live state of a background (video) scan
  const jobRef = useRef(null)                     // { id, cancelled } - lets the polling loop and Cancel talk
  const fileInputRef = useRef(null)
  const navigate = useNavigate()

  useEffect(() => {
    if (!showRecent) return
    const saved = localStorage.getItem('truthlens_recent')
    if (saved) {
      try { setRecentFiles(JSON.parse(saved).slice(0, 4)) } catch {}
    }
  }, [showRecent])

  // Leaving the page stops the polling; the scan itself keeps running server-side and lands in the history.
  useEffect(() => () => { if (jobRef.current) jobRef.current.cancelled = true }, [])

  /** Video is slow: run it as a background job and poll its real progress instead of holding one long request. */
  const runVideoJob = async (uploadId) => {
    const started = await startDetectionJob(uploadId)
    let current = started.data
    jobRef.current = { id: current.job_id, cancelled: false }
    setJob(current)
    let failures = 0
    while (current.state === 'queued' || current.state === 'running') {
      await sleep(POLL_INTERVAL_MS)
      if (jobRef.current?.cancelled) throw new JobCancelledError()
      try {
        current = (await getJob(jobRef.current.id)).data
        failures = 0
        setJob(current)
      } catch (err) {
        if (err.status === 404) return getDetectionResult(uploadId)   // server restarted: the result may still exist
        if (++failures >= 5) throw err                                  // ride out brief network hiccups
      }
    }
    if (current.state === 'failed') throw new Error(current.error || 'Analysis failed.')
    if (current.state === 'cancelled') throw new JobCancelledError()
    return getDetectionResult(uploadId)
  }

  const cancelScan = async () => {
    const cur = jobRef.current
    if (cur) {
      cur.cancelled = true
      try { await cancelJob(cur.id) } catch { /* it may already have finished; nothing more to do */ }
    }
    setUploading(false); setAnalyzing(false); setDetectDone(false); setJob(null)
    setFile(null); setError(null); setResult(null)
  }

  const handleDragOver = useCallback((e) => { e.preventDefault(); setIsDragging(true) }, [])
  const handleDragLeave = useCallback((e) => { e.preventDefault(); setIsDragging(false) }, [])
  const handleDrop = useCallback((e) => {
    e.preventDefault(); setIsDragging(false)
    const f = e.dataTransfer.files[0]; if (f) processFile(f)
  }, [])
  const handleFileSelect = useCallback((e) => {
    const f = e.target.files[0]; if (f) processFile(f)
  }, [])

  const processFile = async (selectedFile) => {
    setFile(selectedFile); setError(null); setResult(null)
    setUploading(true); setProgress(0); setDetectDone(false); setInspectOpen(false); setJob(null)
    try {
      const uploadRes = await uploadMedia(selectedFile, (p) => setProgress(p))
      const uploadId = uploadRes.data.upload_id
      setUploading(false); setAnalyzing(true)
      const detectRes = kindOf(selectedFile.type) === 'video' ? await runVideoJob(uploadId) : await runDetection(uploadId)
      setDetectDone(true)
      setTimeout(() => {
        setAnalyzing(false)
        const res = {
          ...detectRes.data, uploadId,
          filename: selectedFile.name,
          fileType: selectedFile.type,
          date: new Date().toISOString(),
        }
        setResult(res)
        if (showRecent) {
          const saved = JSON.parse(localStorage.getItem('truthlens_recent') || '[]')
          const updated = [res, ...saved].slice(0, 10)
          localStorage.setItem('truthlens_recent', JSON.stringify(updated))
          setRecentFiles(updated.slice(0, 4))
        }
      }, 500)
    } catch (err) {
      if (err instanceof JobCancelledError) return          // the user cancelled: cancelScan already reset the UI
      setUploading(false); setAnalyzing(false); setJob(null); setError(err.message)
    }
  }

  const reset = () => { setFile(null); setResult(null); setError(null); setProgress(0); setDetectDone(false); setInspectOpen(false); setJob(null) }

  const info = result ? getVerdictInfo(result.verdict || result.label) : null

  const radius = compact ? 'rounded-[4px]' : 'rounded-[6px]'
  const minHeight = compact ? 'min-h-[480px]' : 'min-h-[440px] md:min-h-[500px]'
  const frameGapIdle = compact ? 24 : 32
  const frameArmIdle = compact ? 12 : 16
  const frameGapResult = compact ? 20 : 28
  const frameArmResult = compact ? 14 : 16
  const sealSize = compact ? 96 : 104
  const gaugeSize = compact ? 124 : 132

  const idle = !result && !uploading && !analyzing

  const core = (
    <>
      {heading && idle && <div className="animate-fade-in-up">{heading}</div>}
      {!result ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !uploading && !analyzing && fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && !uploading && !analyzing) fileInputRef.current?.click() }}
          aria-label="Upload media to verify"
          className={`relative bg-panel tl-ticks ${radius} ${minHeight} flex flex-col items-center justify-center cursor-pointer transition-colors duration-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-brass ${
            isDragging ? 'border-2 border-dashed border-brass bg-[rgba(200,147,97,0.04)]' : 'border border-dashed border-line-strong'
          } ${uploading || analyzing ? 'cursor-default' : ''}`}
        >
          <input ref={fileInputRef} type="file" accept={ACCEPTED_INPUT_ACCEPT} onChange={handleFileSelect} className="hidden" />
          <ScanFrame gap={frameGapIdle} armSize={frameArmIdle} color="rgba(200,147,97,0.3)" />

          {(uploading || analyzing) && (
            <div className="flex flex-col md:flex-row items-center gap-8 w-full max-w-[520px] px-6">
              <div className="w-full md:w-[220px] shrink-0">
                <SpecimenPreview file={file} active={analyzing} />
              </div>
              <div className="flex flex-col items-center md:items-start gap-4 flex-1">
                <div className="text-center md:text-left">
                  <p className="tl-hud-label mb-1">{uploading ? 'Receiving specimen' : 'Under inspection'}</p>
                  <p className="font-serif text-[19px] text-bone truncate max-w-[260px]">{file?.name}</p>
                  <p className="tl-figure text-[11px] text-bone-faint mt-1">{fmtSize(file?.size)} · {file?.type || 'unknown type'}</p>
                </div>
                {uploading && (
                  <div className="flex items-center gap-3 w-full max-w-[220px]">
                    <div className="flex-1 h-[3px] bg-line-strong rounded-full overflow-hidden">
                      <div className="h-full bg-brass rounded-full transition-[width] duration-300" style={{ width: `${progress}%` }} />
                    </div>
                    <span className="tl-figure text-[11px] text-bone-dim w-9 text-right">{progress}%</span>
                  </div>
                )}
                {analyzing && (job
                  ? <JobProgress job={job} onCancel={(e) => { e?.stopPropagation?.(); cancelScan() }} />
                  : <ForensicProcess done={detectDone} />)}
              </div>
            </div>
          )}

          {error && !uploading && !analyzing && (
            <div className="flex flex-col items-center gap-6 px-8 text-center">
              <div className="w-16 h-16 rounded-full bg-[rgba(209,101,101,0.1)] flex items-center justify-center">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#D16565" strokeWidth="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 8V12"/><path d="M12 16H12.01"/></svg>
              </div>
              <div className="space-y-2">
                <p className="font-serif text-[20px] text-bone">Analysis unavailable</p>
                <p className="text-[14px] text-verdictDanger max-w-[320px]">{error}</p>
              </div>
              <button onClick={(e) => { e.stopPropagation(); reset() }} className="text-[13px] font-medium tracking-[0.06em] text-brass hover:text-bone transition-colors duration-300 link-underline">
                Try again
              </button>
            </div>
          )}

          {!uploading && !analyzing && !error && (
            <div className="flex flex-col items-center gap-8 text-center px-8">
              <div className="w-24 h-24 rounded-full bg-[rgba(200,147,97,0.06)] border border-line flex items-center justify-center transition-transform duration-500 hover:scale-105">
                <UploadIcon size={36} strokeWidth={1.25} className="text-bone-faint" />
              </div>
              <div className="space-y-3">
                <p className="font-serif text-[28px] md:text-[32px] text-bone">Place the specimen</p>
                <p className="text-[15px] text-bone-dim leading-relaxed max-w-[320px]">or click anywhere in this area to browse</p>
              </div>
              <div className="space-y-2">
                <p className="tl-hud-label !text-[9px]">Accepted specimen types</p>
                <div className="flex items-center gap-3 flex-wrap justify-center">
                  {ACCEPTED_FORMAT_CHIPS.map((fmt) => (
                    <span key={fmt} className="text-[10px] font-mono tracking-[0.05em] uppercase text-bone-faint px-3 py-1.5 border border-line rounded-[3px]">{fmt}</span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className={`relative bg-panel ${radius} p-10 md:p-14 overflow-hidden border border-line`}
        >
          <ScanFrame gap={frameGapResult} armSize={frameArmResult} color="rgba(200,147,97,0.3)" />
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 sm:gap-0 mb-8">
            <p className="tl-hud-label !text-brass">Case Resolved</p>
            <div className="flex items-baseline gap-3">
              <CaseTag id={result.uploadId} />
              <span className="tl-figure text-[10px] text-bone-faint">
                Logged {new Date(result.date).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          </div>

          <div className="flex flex-col md:flex-row items-center gap-10 mb-10">
            <VerdictSeal verdict={result.verdict || result.label} size={sealSize} />
            <div className="flex-1 text-center md:text-left">
              <h3 className="font-serif text-[26px] md:text-[30px] text-bone mb-2 break-all">{file?.name}</h3>
              <p className={`text-[14px] mb-3 ${info.textClass}`}>{info.caption}</p>
              <VerdictBadge verdict={result.verdict || result.label} />
            </div>
            <ConfidenceGauge value={result.confidence_score} accent={info.accent} size={gaugeSize} />
          </div>

          {/* Examination Controls — the case isn't over just because the
              verdict rendered: the specimen can still be inspected again,
              or replaced outright, without leaving this card. */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.45, duration: 0.5 }}
            className="flex items-center justify-between flex-wrap gap-4 pt-8 border-t border-line"
          >
            <p className="tl-hud-label !text-[9px]">Examination Controls</p>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setInspectOpen((o) => !o)}
                aria-expanded={inspectOpen}
                aria-controls="specimen-inspect-panel"
                className="flex items-center gap-2 text-[11px] font-medium tracking-[0.08em] uppercase bg-panel-raised border border-line-strong text-bone px-4 py-2.5 rounded-[3px] btn-lift"
              >
                <Search size={13} strokeWidth={1.75} />
                {inspectOpen ? 'Hide Specimen' : 'Inspect Specimen'}
                <ChevronDown
                  size={12}
                  strokeWidth={1.75}
                  style={{ transform: inspectOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.3s cubic-bezier(0.22,1,0.36,1)' }}
                />
              </button>
              <button
                type="button"
                onClick={reset}
                className="flex items-center gap-2 text-[11px] font-medium tracking-[0.08em] uppercase text-bone-dim hover:text-bone transition-colors duration-300 px-4 py-2.5 rounded-[3px] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass"
              >
                <RotateCcw size={13} strokeWidth={1.75} />
                Replace Specimen
              </button>
            </div>
          </motion.div>

          <Disclosure open={inspectOpen} id="specimen-inspect-panel">
            <div className="grid md:grid-cols-2 gap-8 pt-8">
              <div>
                <p className="tl-hud-label !text-[9px] mb-3">{SCAN_LABEL[kindOf(file?.type)]} — captured from your file</p>
                <SpecimenPreview file={file} active={false} />
              </div>
              <SpecimenDossier file={file} />
            </div>
          </Disclosure>

          {/* Provenance (image/video): C2PA credentials + embedded metadata. Supplementary to the verdict;
              a conflict with the model verdict is shown up front. Cheap, so fetched automatically. */}
          <ProvenancePanel uploadId={result.uploadId} mediaType={kindOf(file?.type)} className="mt-8" />

          {/* Explainability (image/video): fetched only when opened. */}
          <ExplanationPanel uploadId={result.uploadId} mediaType={kindOf(file?.type)} className="mt-8" />

          {/* Feedback: was the verdict right? Optional, changeable, withdrawable. */}
          <FeedbackPanel uploadId={result.uploadId} className="mt-8" />

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-10 mt-10"
          >
            <div>
              <p className="text-[10px] tracking-[0.15em] uppercase text-bone-faint mb-1">File Type</p>
              <p className="tl-figure text-[13px] text-bone">{file?.type?.split('/')[1]?.toUpperCase() || 'UNKNOWN'}</p>
            </div>
            <div>
              <p className="text-[10px] tracking-[0.15em] uppercase text-bone-faint mb-1">Size</p>
              <p className="tl-figure text-[13px] text-bone">{fmtSize(file?.size)}</p>
            </div>
            <div>
              <p className="text-[10px] tracking-[0.15em] uppercase text-bone-faint mb-1">Model</p>
              <p className="tl-figure text-[13px] text-bone truncate">{result.model_used || '--'}</p>
            </div>
            <div>
              <p className="text-[10px] tracking-[0.15em] uppercase text-bone-faint mb-1">Processing Time</p>
              <p className="tl-figure text-[13px] text-bone">{result.processing_time_ms ? `${Math.round(result.processing_time_ms)} ms` : '--'}</p>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.65, duration: 0.5 }}
            className="flex items-center gap-4 flex-wrap"
          >
            <button onClick={() => navigate(`/report/${result.uploadId}`)} className="flex items-center gap-2 bg-brass text-ground px-6 py-3 rounded-[4px] text-[12px] font-medium tracking-[0.08em] uppercase btn-lift">
              View Full Report <ArrowRight />
            </button>
            <button type="button" onClick={() => downloadReport(result.uploadId)} className="flex items-center gap-2 text-[12px] font-medium tracking-[0.06em] text-bone-dim hover:text-bone transition-colors duration-300 link-underline px-4 py-3">
              <Download size={15} strokeWidth={1.75} /> Download PDF
            </button>
          </motion.div>
        </motion.div>
      )}
    </>
  )

  return (
    <>
      {compact ? core : (
        <div className="relative z-10 max-w-[900px] mx-auto px-6 w-full">{core}</div>
      )}

      {showRecent && recentFiles.length > 0 && !result && (
        <section className="py-16 md:py-24 bg-panel border-t border-line">
          <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
            <Reveal>
              <div className="flex items-baseline justify-between mb-10">
                <p className="tl-hud-label">Recent specimens</p>
                <button onClick={() => navigate('/history')} className="text-[12px] font-medium tracking-[0.06em] text-bone-dim hover:text-bone transition-colors duration-300 link-underline">
                  View all history
                </button>
              </div>
            </Reveal>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {recentFiles.map((item, i) => (
                <Reveal key={i} delay={i * 0.06}>
                  <div
                    role="button"
                    tabIndex={0}
                    aria-label={`View report for ${item.filename || 'untitled scan'}`}
                    className="group bg-panel-raised border border-line rounded-[4px] p-6 card-hover cursor-pointer h-full focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass"
                    onClick={() => navigate(`/report/${item.uploadId}`)}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); navigate(`/report/${item.uploadId}`) } }}
                  >
                    <div className="flex items-center justify-between gap-3 mb-4">
                      <div className="text-bone-faint">{getFileIcon(item.fileType)}</div>
                      <VerdictBadge verdict={item.verdict || item.label} />
                    </div>
                    <p className="font-serif text-[16px] text-bone mb-1 truncate">{item.filename}</p>
                    <p className="tl-figure text-[11px] text-bone-faint">{new Date(item.date).toLocaleDateString()}</p>
                  </div>
                </Reveal>
              ))}
            </div>
          </div>
        </section>
      )}
    </>
  )
}
