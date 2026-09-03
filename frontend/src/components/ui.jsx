import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { ShieldAlert, ShieldCheck, AlertTriangle } from 'lucide-react'

export const fadeUp = {
  hidden: { opacity: 0, y: 22 },
  show: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.07, duration: 0.55, ease: [0.22, 1, 0.36, 1] },
  }),
}

export const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08, delayChildren: 0.06 } },
}

export const itemUp = {
  hidden: { opacity: 0, y: 18, filter: 'blur(4px)' },
  show: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] },
  },
}

export function PageFade({ children, pageKey }) {
  return (
    <motion.div
      key={pageKey}
      initial={{ opacity: 0, y: 16, filter: 'blur(6px)' }}
      animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      exit={{ opacity: 0, y: -10, filter: 'blur(4px)' }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  )
}

export function Eyebrow({ children }) {
  return (
    <motion.p
      variants={itemUp}
      className="text-mint text-[11px] font-semibold tracking-[0.22em] uppercase mb-3 inline-flex items-center gap-2"
    >
      <span className="w-1.5 h-1.5 rounded-full bg-mint shadow-[0_0_10px_#3DFF9A]" />
      {children}
    </motion.p>
  )
}

export function Stat({ label, value, color = '#F3F5F9' }) {
  return (
    <motion.div variants={itemUp} className="group">
      <p
        className="font-display text-4xl md:text-5xl font-extrabold tracking-tight transition-transform duration-300 group-hover:scale-[1.04] origin-left"
        style={{ color }}
      >
        {value}
      </p>
      <p className="text-sm text-soft mt-2">{label}</p>
    </motion.div>
  )
}

export function Badge({ children, tone = 'mint' }) {
  const styles = {
    mint:    { background: 'rgba(61,255,154,0.12)', color: '#3DFF9A' },
    real:    { background: 'rgba(61,255,154,0.12)', color: '#3DFF9A' },
    fake:    { background: 'rgba(255,77,106,0.14)', color: '#FF4D6A' },
    caution: { background: 'rgba(255,176,32,0.14)', color: '#FFB020' },
    muted:   { background: 'rgba(139,148,168,0.12)', color: '#8B94A8' },
  }
  return (
    <span
      className="inline-flex items-center gap-1.5 text-[11px] font-semibold px-2.5 py-1 rounded-lg backdrop-blur-sm"
      style={styles[tone] || styles.mint}
    >
      {children}
    </span>
  )
}

export function VerdictMark({ verdict }) {
  const cfg = {
    FAKE:      { icon: ShieldAlert,   color: '#FF4D6A', label: 'Manipulated' },
    REAL:      { icon: ShieldCheck,   color: '#3DFF9A', label: 'Authentic' },
    UNCERTAIN: { icon: AlertTriangle, color: '#FFB020', label: 'Needs review' },
  }[verdict] ?? { icon: AlertTriangle, color: '#8B94A8', label: verdict || 'Pending' }
  const Icon = cfg.icon
  return (
    <motion.div
      initial={{ opacity: 0, x: 8 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex items-center gap-2.5"
      style={{ color: cfg.color }}
    >
      <Icon className="w-5 h-5" />
      <span className="font-display font-bold text-xl text-snow">{cfg.label}</span>
    </motion.div>
  )
}

export function AnimatedCounter({ value, decimals = 0, suffix = '' }) {
  const [display, setDisplay] = useState(0)
  useEffect(() => {
    let raf, start
    const step = (ts) => {
      if (!start) start = ts
      const t = Math.min((ts - start) / 1000, 1)
      const eased = 1 - Math.pow(1 - t, 3)
      setDisplay(value * eased)
      if (t < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [value])
  return <>{display.toFixed(decimals)}{suffix}</>
}

export function Gauge({ score, verdict, size = 140 }) {
  const pct = Math.round(score * 100)
  const color = { FAKE: '#FF4D6A', REAL: '#3DFF9A', UNCERTAIN: '#FFB020' }[verdict] || '#8B94A8'
  const r = 54, c = 2 * Math.PI * r
  const offset = c - (pct / 100) * c
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <div
        className="absolute inset-4 rounded-full blur-2xl opacity-40"
        style={{ background: color }}
      />
      <svg viewBox="0 0 120 120" className="-rotate-90 relative z-10" style={{ width: size, height: size }}>
        <circle cx="60" cy="60" r={r} fill="none" stroke="#262D3B" strokeWidth="9" />
        <motion.circle
          cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="9"
          strokeDasharray={c} strokeLinecap="round"
          initial={{ strokeDashoffset: c }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
          style={{ filter: `drop-shadow(0 0 6px ${color})` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center z-10">
        <span className="text-3xl font-display font-extrabold" style={{ color }}>
          <AnimatedCounter value={pct} suffix="%" />
        </span>
        <span className="text-[10px] text-soft uppercase tracking-wider mt-1">risk</span>
      </div>
    </div>
  )
}

export function ScoreRow({ label, score }) {
  const pct = Math.round(score * 100)
  const bar = pct > 70 ? '#FF4D6A' : pct > 40 ? '#FFB020' : '#3DFF9A'
  return (
    <div className="mb-3.5">
      <div className="flex justify-between text-xs mb-1.5">
        <span className="text-soft">{label}</span>
        <span className="font-semibold text-snow">{pct}%</span>
      </div>
      <div className="w-full h-1.5 rounded-full bg-stroke overflow-hidden">
        <motion.div
          className="h-1.5 rounded-full"
          style={{ background: bar, boxShadow: `0 0 12px ${bar}66` }}
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
    </div>
  )
}

/** Spotlight card — mouse-follow glow */
export function Card({ className = '', children, interactive = true, ...rest }) {
  const ref = useRef(null)
  const [spot, setSpot] = useState({ x: 50, y: 50, on: false })

  const onMove = (e) => {
    if (!interactive || !ref.current) return
    const r = ref.current.getBoundingClientRect()
    setSpot({
      x: ((e.clientX - r.left) / r.width) * 100,
      y: ((e.clientY - r.top) / r.height) * 100,
      on: true,
    })
  }

  return (
    <motion.div
      ref={ref}
      onMouseMove={onMove}
      onMouseLeave={() => setSpot(s => ({ ...s, on: false }))}
      whileHover={interactive ? { y: -3, transition: { duration: 0.25 } } : undefined}
      className={`relative bg-elev/90 border border-stroke rounded-2xl overflow-hidden backdrop-blur-sm ${className}`}
      style={{
        boxShadow: spot.on
          ? '0 20px 50px rgba(0,0,0,0.35), 0 0 0 1px rgba(61,255,154,0.08)'
          : '0 8px 30px rgba(0,0,0,0.2)',
      }}
      {...rest}
    >
      {interactive && (
        <div
          className="pointer-events-none absolute inset-0 transition-opacity duration-300"
          style={{
            opacity: spot.on ? 1 : 0,
            background: `radial-gradient(420px circle at ${spot.x}% ${spot.y}%, rgba(61,255,154,0.1), transparent 45%)`,
          }}
        />
      )}
      <div className="relative z-10">{children}</div>
    </motion.div>
  )
}

export function PrimaryButton({ children, className = '', ...rest }) {
  return (
    <motion.button
      type="button"
      whileHover={{ scale: 1.03, y: -1 }}
      whileTap={{ scale: 0.97 }}
      className={`relative tl-pulse-ring bg-mint text-canvas font-bold text-sm px-6 py-3.5 rounded-xl shadow-glow ${className}`}
      {...rest}
    >
      <span className="relative z-10">{children}</span>
    </motion.button>
  )
}

/** Decorative AI core visual for hero */
export function AiCore({ className = '' }) {
  return (
    <div className={`relative w-full max-w-[340px] aspect-square mx-auto ${className}`}>
      <div className="absolute inset-[18%] rounded-full bg-mint/20 blur-3xl tl-float" />
      <div className="absolute inset-0 tl-spin-slow">
        <div className="absolute inset-[8%] rounded-full border border-mint/20 border-dashed" />
        <div className="absolute top-[8%] left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-mint shadow-[0_0_12px_#3DFF9A]" />
      </div>
      <div className="absolute inset-[14%] tl-spin-rev">
        <div className="absolute inset-0 rounded-full border border-stroke" />
        <div className="absolute bottom-[10%] right-[18%] w-1.5 h-1.5 rounded-full bg-snow/70" />
      </div>
      <div className="absolute inset-[28%] rounded-full border border-mint/40 bg-elev2/80 backdrop-blur-md flex items-center justify-center tl-float-delay shadow-[0_0_40px_rgba(61,255,154,0.25)]">
        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-mint to-mint/30 flex items-center justify-center">
          <div className="w-8 h-8 rounded-full bg-canvas/80" />
        </div>
      </div>
      <div className="absolute inset-[42%] rounded-full tl-shimmer opacity-60" />
    </div>
  )
}
