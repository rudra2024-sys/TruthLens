import React, { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import useReducedMotion from '../../hooks/useReducedMotion'

/**
 * A REAL waveform decoded from the uploaded audio file via the Web Audio
 * API — actual amplitude data, not a fabricated squiggle. If decoding fails
 * (unsupported codec, browser quirk), falls back to a plain pulse rather
 * than drawing fake samples.
 */
export default function WaveformScan({ file, active }) {
  const [bars, setBars] = useState(null)
  const [failed, setFailed] = useState(false)
  const reduced = useReducedMotion()
  const ctxRef = useRef(null)

  useEffect(() => {
    if (!file) return
    let cancelled = false
    const AudioCtx = window.AudioContext || window.webkitAudioContext
    if (!AudioCtx) { setFailed(true); return }
    const ctx = new AudioCtx()
    ctxRef.current = ctx
    file.arrayBuffer()
      .then((buf) => ctx.decodeAudioData(buf))
      .then((audioBuf) => {
        if (cancelled) return
        const data = audioBuf.getChannelData(0)
        const BUCKETS = 64
        const size = Math.floor(data.length / BUCKETS)
        const peaks = []
        for (let i = 0; i < BUCKETS; i++) {
          let max = 0
          for (let j = 0; j < size; j++) {
            const v = Math.abs(data[i * size + j] || 0)
            if (v > max) max = v
          }
          peaks.push(max)
        }
        const norm = Math.max(...peaks, 0.01)
        setBars(peaks.map((p) => Math.max(0.06, p / norm)))
      })
      .catch(() => { if (!cancelled) setFailed(true) })
      .finally(() => { ctx.close().catch(() => {}) })
    return () => { cancelled = true; ctx.close().catch(() => {}) }
  }, [file])

  if (failed || !bars) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <span className={`w-2 h-2 rounded-full bg-brass ${reduced ? '' : 'animate-pulse'}`} />
      </div>
    )
  }

  return (
    <div className="relative w-full h-full flex items-center gap-[2px] px-2 overflow-hidden">
      {bars.map((h, i) => (
        <motion.span
          key={i}
          className="flex-1 bg-brass/70 rounded-full"
          style={{ height: `${h * 100}%` }}
          initial={{ scaleY: 0 }}
          animate={{ scaleY: 1 }}
          transition={{ duration: 0.4, delay: reduced ? 0 : i * 0.008 }}
        />
      ))}
      {active && !reduced && (
        <motion.div
          className="absolute top-0 bottom-0 w-[2px] bg-bone"
          style={{ boxShadow: '0 0 10px 2px rgba(237,234,227,0.5)' }}
          initial={{ left: '0%' }}
          animate={{ left: ['0%', '100%'] }}
          transition={{ duration: 2.4, repeat: Infinity, ease: 'linear' }}
        />
      )}
    </div>
  )
}
