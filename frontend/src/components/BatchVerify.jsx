import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { AlertCircle, AudioLines, Clock, Download, FolderOpen, Image as ImageIcon, Loader2, Upload as UploadIcon,
  Video as VideoIcon, Files } from 'lucide-react'
import { cancelJob, downloadBatchExport, findScannedByHash, getDetectionResult, getJob, runDetection, startDetectionJob, uploadMedia }
  from '../api/client'
import { ACCEPTED_INPUT_ACCEPT } from '../lib/acceptedFormats'
import { MAX_BATCH_FILES, entriesFromDrop, entriesFromFileList, selectBatchFiles } from '../lib/batchFiles'
import { createPacer, withRetryOn429 } from '../lib/pacer'
import { sha256Hex } from '../lib/fileHash'
import VerdictBadge from './VerdictBadge'

// Production allows 20 upload/detect requests a minute per client (nginx.prod.conf); local dev has no limit.
const PACE_MS = Number(import.meta.env.VITE_BATCH_PACE_MS ?? (import.meta.env.DEV ? 0 : 3400))
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const KIND_ICON = { image: ImageIcon, video: VideoIcon, audio: AudioLines }
const STATUS_TEXT = {
  waiting: 'Waiting', uploading: 'Uploading', scanning: 'Scanning', done: 'Done', failed: 'Failed', cancelled: 'Cancelled',
}

function formatEta(seconds) {
  if (!isFinite(seconds) || seconds < 0) return ''
  if (seconds < 60) return 'under a minute'
  return `about ${Math.round(seconds / 60)} min`
}

export default function BatchVerify() {
  const [items, setItems] = useState([])
  const [notice, setNotice] = useState('')
  const [running, setRunning] = useState(false)
  const [finished, setFinished] = useState(false)
  const [exporting, setExporting] = useState('')
  const [exportError, setExportError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [startedAt, setStartedAt] = useState(0)
  const [, setTick] = useState(0)
  const cancelled = useRef(false)
  const jobIds = useRef(new Set())
  const pacer = useRef(null)
  const fileInput = useRef(null)
  const folderInput = useRef(null)

  // Closing the page stops the queue (files already handed to the server still finish and appear in History).
  useEffect(() => () => { cancelled.current = true }, [])
  useEffect(() => {
    if (!running) return undefined
    const t = setInterval(() => setTick((n) => n + 1), 1000)
    return () => clearInterval(t)
  }, [running])

  const patch = useCallback((id, changes) =>
    setItems((cur) => cur.map((it) => (it.id === id ? { ...it, ...changes } : it))), [])

  const addEntries = useCallback((entries) => {
    setItems((cur) => {
      const { accepted, skippedType, skippedOverLimit } = selectBatchFiles(entries, cur.length)
      const msgs = []
      if (skippedType) msgs.push(`${skippedType} file${skippedType === 1 ? '' : 's'} skipped (not an image, video or audio file we accept).`)
      if (skippedOverLimit) msgs.push(`Only ${MAX_BATCH_FILES} files fit in one batch; ${skippedOverLimit} left out.`)
      if (!accepted.length && !msgs.length) msgs.push('No files found.')
      setNotice(msgs.join(' '))
      const base = Date.now()
      return [...cur, ...accepted.map((a, i) => ({
        id: `${base}-${cur.length + i}`, file: a.file, path: a.path, kind: a.kind, status: 'waiting',
        uploadId: null, verdict: null, confidence: null, error: null,
      }))]
    })
    setFinished(false)
  }, [])

  const onDrop = async (e) => {
    e.preventDefault(); setDragging(false)
    if (running) return
    try { addEntries(await entriesFromDrop(e.dataTransfer)) } catch { setNotice('Could not read the dropped items.') }
  }
  const onPick = (e) => { if (!running) addEntries(entriesFromFileList(e.target.files)); e.target.value = '' }

  const paced = (fn) => withRetryOn429(async () => { await pacer.current(); return fn() })

  async function scanOne(item) {
    // Same bytes already scanned by this user: reuse that result instead of uploading and scanning again.
    const hash = await sha256Hex(item.file)
    const earlier = hash ? await paced(() => findScannedByHash(hash)) : null
    if (earlier) {
      patch(item.id, { status: 'done', uploadId: earlier.upload_id, verdict: earlier.verdict,
        confidence: earlier.confidence_score, reused: true })
      return
    }
    patch(item.id, { status: 'uploading' })
    const up = await paced(() => uploadMedia(item.file))
    const uploadId = up.data.upload_id
    patch(item.id, { status: 'scanning', uploadId })
    let result
    if (item.kind === 'video') {
      const started = await paced(() => startDetectionJob(uploadId))
      let job = started.data
      jobIds.current.add(job.job_id)
      try {
        while (job.state === 'queued' || job.state === 'running') {
          if (cancelled.current) { try { await cancelJob(job.job_id) } catch { /* may have just finished */ } break }
          await sleep(1200)
          job = (await getJob(job.job_id)).data
        }
      } finally { jobIds.current.delete(job.job_id) }
      if (cancelled.current && job.state !== 'done') { patch(item.id, { status: 'cancelled' }); return }
      if (job.state === 'failed') throw new Error(job.error || 'Analysis failed.')
      if (job.state === 'cancelled') { patch(item.id, { status: 'cancelled' }); return }
      result = (await paced(() => getDetectionResult(uploadId))).data
    } else {
      result = (await paced(() => runDetection(uploadId))).data
    }
    patch(item.id, { status: 'done', verdict: result.verdict, confidence: result.confidence_score })
  }

  async function lane(queue, workers) {
    const worker = async () => {
      for (;;) {
        if (cancelled.current) return
        const item = queue.shift()
        if (!item) return
        try { await scanOne(item) } catch (err) { patch(item.id, { status: 'failed', error: err.message || 'Scan failed.' }) }
      }
    }
    await Promise.all(Array.from({ length: workers }, worker))
  }

  const start = async () => {
    cancelled.current = false
    pacer.current = createPacer({ refillMs: PACE_MS })
    setRunning(true); setFinished(false); setExportError(''); setStartedAt(Date.now())
    const todo = items.filter((it) => it.status === 'waiting')
    // Images and audio two at a time; videos one at a time (the server scans one video at a time anyway).
    await Promise.all([lane(todo.filter((i) => i.kind !== 'video'), 2), lane(todo.filter((i) => i.kind === 'video'), 1)])
    setItems((cur) => cur.map((it) => (it.status === 'waiting' ? { ...it, status: 'cancelled' } : it)))
    setRunning(false); setFinished(true)
  }

  const cancel = () => { cancelled.current = true }
  const clear = () => { setItems([]); setNotice(''); setFinished(false); setExportError('') }

  const counts = useMemo(() => {
    const c = { total: items.length, done: 0, failed: 0, cancelled: 0, REAL: 0, FAKE: 0, UNCERTAIN: 0 }
    items.forEach((it) => {
      if (it.status === 'done') { c.done++; c[it.verdict] = (c[it.verdict] || 0) + 1 }
      if (it.status === 'failed') c.failed++
      if (it.status === 'cancelled') c.cancelled++
    })
    return c
  }, [items])
  const settled = counts.done + counts.failed + counts.cancelled
  const waitingCount = items.filter((i) => i.status === 'waiting').length
  const eta = running && counts.done > 0
    ? formatEta(((Date.now() - startedAt) / 1000 / counts.done) * (items.length - settled)) : ''

  const exportAs = async (format) => {
    setExporting(format); setExportError('')
    try {
      await downloadBatchExport(items.map((it) => ({
        upload_id: it.status === 'done' || (it.status === 'failed' && it.uploadId) ? it.uploadId : null,
        file_name: it.path.slice(0, 300),
        error: it.status === 'failed' ? it.error : it.status === 'cancelled' ? 'Cancelled before it was scanned' : null,
      })), format)
    } catch (err) { setExportError(err.message) } finally { setExporting('') }
  }

  const empty = items.length === 0

  return (
    <div className="w-full">
      <input ref={fileInput} type="file" multiple accept={ACCEPTED_INPUT_ACCEPT} onChange={onPick} className="hidden" />
      <input ref={folderInput} type="file" webkitdirectory="" directory="" multiple onChange={onPick} className="hidden" />

      {!running && (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`relative border border-dashed rounded-[6px] bg-panel text-center transition-colors duration-300 ${
            dragging ? 'border-brass bg-[rgba(200,147,97,0.06)]' : 'border-line'} ${empty ? 'p-10 md:p-14' : 'p-6'}`}
        >
          {empty && (
            <div className="w-16 h-16 mx-auto mb-5 rounded-full bg-[rgba(200,147,97,0.06)] border border-line flex items-center justify-center">
              <Files size={26} strokeWidth={1.25} className="text-bone-faint" />
            </div>
          )}
          <p className="font-serif text-[22px] md:text-[26px] text-bone mb-2">
            {empty ? 'Drop several files, or a whole folder' : 'Add more files'}
          </p>
          <p className="text-[14px] text-bone-dim mb-5">
            Up to {MAX_BATCH_FILES} images, videos and audio files per batch. Folders are searched, other file types are skipped.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <button type="button" onClick={() => fileInput.current?.click()}
              className="inline-flex items-center gap-2 px-5 py-2.5 border border-line-strong text-[12px] tracking-[0.1em] uppercase text-bone hover:border-brass hover:text-brass transition-colors">
              <UploadIcon size={14} strokeWidth={1.5} /> Choose files
            </button>
            <button type="button" onClick={() => folderInput.current?.click()}
              className="inline-flex items-center gap-2 px-5 py-2.5 border border-line-strong text-[12px] tracking-[0.1em] uppercase text-bone hover:border-brass hover:text-brass transition-colors">
              <FolderOpen size={14} strokeWidth={1.5} /> Choose a folder
            </button>
          </div>
        </div>
      )}

      {notice && (
        <div role="note" className="flex gap-3 border border-verdictCaution/60 bg-verdictCaution/10 rounded-[4px] p-3 mt-4">
          <AlertCircle size={16} strokeWidth={1.75} className="text-verdictCaution shrink-0 mt-[2px]" aria-hidden="true" />
          <p className="text-[13px] text-bone-dim">{notice}</p>
        </div>
      )}

      {!empty && (
        <div className="mt-6">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <p className="tl-hud-label">
              {running ? `Scanning: ${settled} of ${items.length} done${eta ? ` · ${eta} left` : ''}`
                : finished ? `Finished: ${counts.done} scanned${counts.failed ? `, ${counts.failed} failed` : ''}${counts.cancelled ? `, ${counts.cancelled} cancelled` : ''}`
                : `${items.length} file${items.length === 1 ? '' : 's'} ready`}
            </p>
            <div className="flex flex-wrap items-center gap-3">
              {!running && waitingCount > 0 && (
                <button type="button" onClick={start}
                  className="px-6 py-2.5 bg-brass text-ground text-[12px] font-medium tracking-[0.1em] uppercase hover:opacity-90 transition-opacity">
                  Scan {waitingCount} file{waitingCount === 1 ? '' : 's'}
                </button>
              )}
              {running && (
                <button type="button" onClick={cancel} disabled={cancelled.current}
                  className="px-5 py-2.5 border border-line-strong text-[12px] tracking-[0.1em] uppercase text-bone hover:border-verdictDanger hover:text-verdictDanger transition-colors disabled:opacity-50">
                  {cancelled.current ? 'Stopping…' : 'Cancel'}
                </button>
              )}
              {!running && (
                <button type="button" onClick={clear} className="text-[12px] tracking-[0.1em] uppercase text-bone-dim hover:text-bone transition-colors">
                  Clear
                </button>
              )}
            </div>
          </div>

          {running && (
            <>
              <div className="h-[3px] bg-line rounded-full overflow-hidden mb-2" role="progressbar" aria-valuemin={0}
                aria-valuemax={items.length} aria-valuenow={settled} aria-label="Batch progress">
                <div className="h-full bg-brass transition-all duration-500" style={{ width: `${(settled / items.length) * 100}%` }} />
              </div>
              <p className="text-[12px] text-bone-faint mb-3">
                {PACE_MS > 0 ? 'Files are sent at a steady pace to stay within the server\'s request limit. ' : ''}
                Keep this page open: closing it stops the rest of the batch.
              </p>
            </>
          )}

          <div className="border border-line rounded-[4px] overflow-hidden bg-panel">
            <ul className="divide-y divide-line max-h-[460px] overflow-y-auto">
              {items.map((it) => {
                const Icon = KIND_ICON[it.kind] || Files
                return (
                  <li key={it.id} className="flex items-center gap-3 px-4 py-3">
                    <Icon size={16} strokeWidth={1.5} className="text-bone-faint shrink-0" aria-hidden="true" />
                    <span className="flex-1 min-w-0 text-[14px] text-bone truncate" title={it.path}>{it.path}</span>
                    <span className="shrink-0 text-right min-w-[150px]">
                      {it.status === 'done' ? (
                        <span className="inline-flex items-center gap-3">
                          <VerdictBadge verdict={it.verdict} />
                          <span className="tl-figure text-[12px] text-bone-dim w-9 text-right">{Math.round(it.confidence * 100)}%</span>
                          {it.reused && <span className="text-[11px] text-bone-faint" title="Identical file scanned before; earlier result reused">Scanned before</span>}
                        </span>
                      ) : it.status === 'failed' ? (
                        <span className="text-[12px] text-verdictDanger" title={it.error}>Failed</span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 text-[12px] text-bone-faint">
                          {(it.status === 'uploading' || it.status === 'scanning') ? <Loader2 size={12} className="animate-spin" />
                            : <Clock size={12} />}
                          {STATUS_TEXT[it.status]}
                        </span>
                      )}
                    </span>
                    <span className="shrink-0 w-[52px] text-right">
                      {it.status === 'done' && it.uploadId && (
                        <Link to={`/report/${it.uploadId}`} className="text-[12px] text-brass hover:underline">Report</Link>
                      )}
                    </span>
                  </li>
                )
              })}
            </ul>
          </div>

          {items.some((i) => i.status === 'failed') && (
            <ul className="mt-3 space-y-1">
              {items.filter((i) => i.status === 'failed').slice(0, 5).map((i) => (
                <li key={i.id} className="text-[12px] text-bone-dim"><span className="text-verdictDanger">{i.path}:</span> {i.error}</li>
              ))}
            </ul>
          )}

          {finished && (counts.done > 0 || counts.failed > 0) && (
            <div className="mt-6 border border-line rounded-[4px] bg-panel p-5">
              <p className="text-[14px] text-bone mb-1">
                {counts.REAL} authentic · {counts.FAKE} manipulated · {counts.UNCERTAIN} uncertain
              </p>
              <p className="text-[12px] text-bone-faint mb-4">Download every result in one file. The summary PDF explains the results in plain words.</p>
              <div className="flex flex-wrap gap-3">
                <button type="button" onClick={() => exportAs('csv')} disabled={!!exporting}
                  className="inline-flex items-center gap-2 px-5 py-2.5 bg-brass text-ground text-[12px] font-medium tracking-[0.1em] uppercase hover:opacity-90 transition-opacity disabled:opacity-50">
                  {exporting === 'csv' ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} strokeWidth={1.75} />} Download CSV
                </button>
                <button type="button" onClick={() => exportAs('pdf')} disabled={!!exporting}
                  className="inline-flex items-center gap-2 px-5 py-2.5 border border-line-strong text-[12px] tracking-[0.1em] uppercase text-bone hover:border-brass hover:text-brass transition-colors disabled:opacity-50">
                  {exporting === 'pdf' ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} strokeWidth={1.75} />} Download summary PDF
                </button>
              </div>
              {exportError && <p className="text-[12px] text-verdictDanger mt-3" role="alert">{exportError}</p>}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
