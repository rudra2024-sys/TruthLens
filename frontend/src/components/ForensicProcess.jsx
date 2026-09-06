import React, { useEffect, useRef, useState } from 'react'

const STAGES = ['CALIBRATING', 'INSPECTING', 'EXTRACTING SIGNAL', 'CORRELATING', 'ASSEMBLING EVIDENCE', 'RESOLVING']

/**
 * The visual procedure shown while the real `runDetection` request is in
 * flight. This is a single network call — the backend does not report
 * discrete stages — so these labels are a conceptual presentation of "an
 * inspection is underway," not a claim that the server is literally
 * executing each named step. It advances on a timer and holds at the
 * second-to-last stage until the real response arrives (`done`), at which
 * point it resolves immediately. Elapsed time shown is real (measured
 * client-side), never simulated.
 */
export default function ForensicProcess({ done = false }) {
  const [index, setIndex] = useState(0)
  const [elapsedMs, setElapsedMs] = useState(0)
  const startRef = useRef(Date.now())

  useEffect(() => {
    if (done) { setIndex(STAGES.length - 1); return }
    const id = setInterval(() => {
      setIndex((i) => Math.min(i + 1, STAGES.length - 2))
    }, 900)
    return () => clearInterval(id)
  }, [done])

  useEffect(() => {
    const id = setInterval(() => setElapsedMs(Date.now() - startRef.current), 100)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="w-full max-w-[280px]" role="status" aria-live="polite">
      <div className="flex items-center justify-between mb-3">
        <span className="tl-hud-label !text-brass">{STAGES[index]}</span>
        <span className="tl-figure text-[10px] text-bone-faint">{(elapsedMs / 1000).toFixed(1)}s</span>
      </div>
      <div className="flex gap-1">
        {STAGES.map((s, i) => (
          <span key={s} className={`h-[2px] flex-1 rounded-full transition-colors duration-500 ${i <= index ? 'bg-brass' : 'bg-line-strong'}`} />
        ))}
      </div>
    </div>
  )
}
