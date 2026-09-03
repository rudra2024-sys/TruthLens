import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { getStats } from '../api/client'

const TABS = [
  { id: 'dashboard', label: 'Overview' },
  { id: 'scan',      label: 'Scan' },
  { id: 'history',   label: 'History' },
  { id: 'models',    label: 'Pipelines' },
  { id: 'analytics', label: 'Analytics' },
]

export default function NavShell({ active, onChange, children }) {
  const [scanCount, setScanCount] = useState(null)

  useEffect(() => {
    getStats()
      .then(r => setScanCount(r.data.total_scans))
      .catch(() => setScanCount(null))
  }, [active])

  return (
    <div className="min-h-screen tl-mesh text-snow font-body relative">
      <div className="fixed inset-0 tl-grid pointer-events-none opacity-90" />
      <div className="tl-noise" />

      {/* Ambient orbs */}
      <motion.div
        className="tl-orb w-[420px] h-[420px] bg-mint/15 top-[-120px] right-[-80px]"
        animate={{ x: [0, 30, 0], y: [0, 20, 0] }}
        transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="tl-orb w-[320px] h-[320px] bg-mint/10 bottom-[10%] left-[-100px]"
        animate={{ x: [0, 24, 0], y: [0, -18, 0] }}
        transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut' }}
      />

      <header className="relative z-30 sticky top-0 border-b border-stroke/60 bg-canvas/70 backdrop-blur-2xl">
        <div className="max-w-6xl mx-auto px-5 md:px-8 h-16 flex items-center gap-8">
          <motion.button
            type="button"
            onClick={() => onChange('dashboard')}
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            className="font-display font-extrabold text-xl tracking-tight text-snow"
          >
            Truth<span className="text-mint">Lens</span>
          </motion.button>

          <nav className="hidden md:flex items-center gap-0.5 flex-1">
            {TABS.map(({ id, label }) => (
              <button
                key={id}
                type="button"
                onClick={() => onChange(id)}
                className={`relative px-3.5 py-2 text-sm transition-colors ${
                  active === id ? 'text-snow font-semibold' : 'text-soft hover:text-snow'
                }`}
              >
                {label}
                {active === id && (
                  <motion.span
                    layoutId="navline"
                    className="absolute left-3 right-3 -bottom-[1px] h-0.5 bg-mint rounded-full shadow-[0_0_8px_#3DFF9A]"
                    transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                  />
                )}
              </button>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3">
            {scanCount != null && (
              <motion.p
                key={scanCount}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                className="hidden sm:block text-xs text-soft"
              >
                <span className="text-snow font-semibold tabular-nums">{scanCount}</span> scans
              </motion.p>
            )}
            <motion.button
              type="button"
              onClick={() => onChange('scan')}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              className="bg-mint text-canvas text-sm font-bold px-4 py-2 rounded-xl shadow-glow"
            >
              New Scan
            </motion.button>
          </div>
        </div>

        <div className="md:hidden flex gap-1.5 px-4 pb-3 overflow-x-auto">
          {TABS.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              onClick={() => onChange(id)}
              className={`shrink-0 px-3 py-1.5 text-xs rounded-lg font-medium transition ${
                active === id ? 'bg-mint text-canvas' : 'bg-elev2 text-soft'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      <main className="relative z-10 max-w-6xl mx-auto px-5 md:px-8 py-10 md:py-14">
        {children}
      </main>
    </div>
  )
}
