import React, { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import useReducedMotion from '../../hooks/useReducedMotion'
import { seededPoints } from '../../lib/seededRandom'

/**
 * Real image preview under inspection — the sampling grid and marker points
 * are a visualization of the scan PROCESS (sample coordinates), never a
 * claim about what was found. Points are seeded from the file's own
 * name+size so they're stable, not fabricated per-render randomness.
 */
export default function PixelGridScan({ file, active }) {
  const [url, setUrl] = useState(null)
  const reduced = useReducedMotion()
  const points = useMemo(() => seededPoints(`${file?.name}-${file?.size}`, 5), [file])

  useEffect(() => {
    if (!file) return
    const objUrl = URL.createObjectURL(file)
    setUrl(objUrl)
    return () => URL.revokeObjectURL(objUrl)
  }, [file])

  if (!url) return null

  return (
    <div className="relative w-full h-full overflow-hidden rounded-[4px] bg-ground">
      <img src={url} alt="" className="w-full h-full object-cover opacity-70" />
      <div
        className="absolute inset-0"
        style={{
          backgroundImage:
            'linear-gradient(rgba(200,147,97,0.16) 1px, transparent 1px), linear-gradient(90deg, rgba(200,147,97,0.16) 1px, transparent 1px)',
          backgroundSize: '10% 10%',
        }}
      />
      {active && points.map((p, i) => (
        <motion.span
          key={i}
          className="absolute w-1.5 h-1.5 rounded-full bg-brass"
          style={{ left: `${p.x}%`, top: `${p.y}%`, boxShadow: '0 0 6px 1px rgba(200,147,97,0.6)' }}
          initial={{ opacity: 0, scale: 0 }}
          animate={reduced ? { opacity: 0.9, scale: 1 } : { opacity: [0, 1, 0.6], scale: [0, 1.3, 1] }}
          transition={{ duration: 0.5, delay: 0.3 + i * 0.35 }}
        />
      ))}
      {active && !reduced && (
        <motion.div
          className="absolute left-0 right-0 h-[2px]"
          style={{ background: 'linear-gradient(90deg, transparent, rgba(200,147,97,0.9) 50%, transparent)', boxShadow: '0 0 12px 2px rgba(200,147,97,0.5)' }}
          initial={{ top: '0%' }}
          animate={{ top: ['0%', '100%', '0%'] }}
          transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
        />
      )}
    </div>
  )
}
