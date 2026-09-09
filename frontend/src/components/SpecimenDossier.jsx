import React, { useEffect, useState } from 'react'

/**
 * Real, browser-derived specimen metadata — read directly off the File
 * object (and, for image/video, a cheap metadata-only probe of the file
 * itself: natural dimensions, media duration). This is never a substitute
 * for the backend's own evidence, and is always labeled separately from it
 * (see the "read from your file" caption) so the two are never confused.
 *
 * Dimensions/duration are read via a plain <img>/<video>/<audio> metadata
 * load, never a full decode — no canvas, no per-frame work, one small
 * one-time read per upload.
 */
const kindOf = (type) => {
  if (!type) return 'other'
  if (type.startsWith('image')) return 'image'
  if (type.startsWith('video')) return 'video'
  if (type.startsWith('audio')) return 'audio'
  return 'other'
}

const fmtSize = (bytes) => {
  if (!bytes) return '—'
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`
}

const fmtDuration = (s) => {
  if (!Number.isFinite(s)) return null
  const m = Math.floor(s / 60)
  const r = Math.floor(s % 60)
  return `${m}:${String(r).padStart(2, '0')}`
}

export default function SpecimenDossier({ file }) {
  const [dims, setDims] = useState(null)
  const [duration, setDuration] = useState(null)

  useEffect(() => {
    setDims(null)
    setDuration(null)
    if (!file) return
    const kind = kindOf(file.type)
    if (kind === 'other') return

    let cancelled = false
    const url = URL.createObjectURL(file)
    let probe

    if (kind === 'image') {
      probe = new Image()
      probe.onload = () => {
        if (cancelled) return
        setDims({ w: probe.naturalWidth, h: probe.naturalHeight })
      }
      probe.src = url
    } else if (kind === 'video') {
      probe = document.createElement('video')
      probe.preload = 'metadata'
      probe.onloadedmetadata = () => {
        if (cancelled) return
        setDims({ w: probe.videoWidth, h: probe.videoHeight })
        setDuration(probe.duration)
      }
      probe.src = url
    } else if (kind === 'audio') {
      probe = document.createElement('audio')
      probe.preload = 'metadata'
      probe.onloadedmetadata = () => {
        if (cancelled) return
        setDuration(probe.duration)
      }
      probe.src = url
    }

    return () => {
      cancelled = true
      URL.revokeObjectURL(url)
    }
  }, [file])

  if (!file) return null

  const durationLabel = fmtDuration(duration)
  const rows = [
    ['Filename', file.name || 'untitled'],
    ['Format', file.type || 'unknown'],
    ['Size', fmtSize(file.size)],
  ]
  if (dims) rows.push(['Dimensions', `${dims.w} × ${dims.h}px`])
  if (durationLabel) rows.push(['Duration', durationLabel])

  return (
    <div>
      <p className="tl-hud-label !text-[9px] mb-3">Specimen Dossier — read from your file</p>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-4">
        {rows.map(([label, value]) => (
          <div key={label} className="min-w-0">
            <p className="text-[10px] tracking-[0.14em] uppercase text-bone-faint mb-1">{label}</p>
            <p className="tl-figure text-[12px] text-bone truncate" title={String(value)}>{value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
