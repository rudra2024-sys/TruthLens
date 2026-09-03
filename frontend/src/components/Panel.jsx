import { motion } from 'framer-motion'

export default function Panel({ children, className = '', animate = true }) {
  return (
    <motion.div
      initial={animate ? { opacity: 0, y: 16 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: 'easeOut' }}
      className={`rounded-[28px] border border-white/10 bg-slate-900/70 shadow-[0_20px_80px_rgba(2,6,23,0.45)] backdrop-blur-xl ${className}`}
    >
      {children}
    </motion.div>
  )
}
