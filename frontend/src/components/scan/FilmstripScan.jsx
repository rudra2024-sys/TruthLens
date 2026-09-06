import React, { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import useReducedMotion from '../../hooks/useReducedMotion'

const FRAME_COUNT = 6

/**
 * A REAL filmstrip captured from the uploaded video via canvas — actual
 * frames from the actual file, not stock imagery. Falls back to a plain
 * video preview (still the real file) if frame capture isn't possible for
 * this codec/browser combination.
 */
export default function FilmstripScan({ file, active }) {
  const [url, setUrl] = useState(null)
  const [frames, setFrames] = useState([])
  const [duration, setDuration] = useState(null)
  const [captureFailed, setCaptureFailed] = useState(false)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const reduced = useReducedMotion()

  useEffect(() => {
    if (!file) return
    const objUrl = URL.createObjectURL(file)
    setUrl(objUrl)
    return () => URL.revokeObjectURL(objUrl)
  }, [file])

  useEffect(() => {
    if (!url) return
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return
    let cancelled = false

    const captureAt = (t) => new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        video.removeEventListener('seeked', onSeeked)
        reject(new Error('seek timed out'))
      }, 3000)
      const onSeeked = () => {
        clearTimeout(timeout)
        video.removeEventListener('seeked', onSeeked)
        try {
          const ctx = canvas.getContext('2d')
          canvas.width = 96; canvas.height = 64
          ctx.drawImage(video, 0, 0, 96, 64)
          resolve(canvas.toDataURL('image/jpeg', 0.6))
        } catch (e) { reject(e) }
      }
      video.addEventListener('seeked', onSeeked)
      video.currentTime = t
    })

    const onMeta = async () => {
      if (cancelled) return
      setDuration(video.duration)
      const dur = video.duration
      if (!isFinite(dur) || dur <= 0) { setCaptureFailed(true); return }
      const shots = []
      try {
        for (let i = 0; i < FRAME_COUNT; i++) {
          const t = (dur * (i + 0.5)) / FRAME_COUNT
          const shot = await captureAt(t)
          if (cancelled) return
          shots.push(shot)
          setFrames([...shots])
        }
      } catch {
        if (!cancelled) setCaptureFailed(true)
      }
    }

    video.addEventListener('loadedmetadata', onMeta)
    return () => { cancelled = true; video.removeEventListener('loadedmetadata', onMeta) }
  }, [url])

  const fmt = (s) => {
    if (!isFinite(s)) return '—:—'
    const m = Math.floor(s / 60), r = Math.floor(s % 60)
    return `${m}:${String(r).padStart(2, '0')}`
  }

  return (
    <div className="relative w-full h-full flex flex-col gap-2 p-2">
      <video ref={videoRef} src={url} muted preload="metadata" className="hidden" />
      <canvas ref={canvasRef} className="hidden" />

      {frames.length > 0 && !captureFailed ? (
        <div className="relative flex-1 flex gap-1 overflow-hidden rounded-[3px]">
          {Array.from({ length: FRAME_COUNT }).map((_, i) => (
            <div key={i} className="flex-1 bg-panel-raised overflow-hidden">
              {frames[i] && <img src={frames[i]} alt="" className="w-full h-full object-cover" />}
            </div>
          ))}
          {active && !reduced && (
            <motion.div
              className="absolute top-0 bottom-0 w-[2px] bg-brass"
              style={{ boxShadow: '0 0 10px 2px rgba(200,147,97,0.6)' }}
              initial={{ left: '0%' }}
              animate={{ left: ['0%', '100%'] }}
              transition={{ duration: 2.6, repeat: Infinity, ease: 'linear' }}
            />
          )}
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <span className={`w-2 h-2 rounded-full bg-brass ${reduced ? '' : 'animate-pulse'}`} />
        </div>
      )}
      {duration != null && (
        <p className="tl-figure text-[9px] text-bone-faint text-center">{fmt(duration)} duration</p>
      )}
    </div>
  )
}
