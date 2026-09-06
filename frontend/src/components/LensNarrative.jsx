import React, { useRef, useState } from 'react'
import {
  motion, useScroll, useTransform, useMotionValueEvent, useMotionTemplate, useSpring, AnimatePresence,
} from 'framer-motion'
import { Fingerprint, ScanSearch, Radio, GitMerge, ClipboardList, Gauge, ShieldCheck, ChevronDown } from 'lucide-react'
import useReducedMotion from '../hooks/useReducedMotion'
import useIsLargeScreen from '../hooks/useIsLargeScreen'
import LensGlass from './LensGlass'
import Reveal from './Reveal'
import VerdictSeal from './VerdictSeal'

/**
 * TruthLens' forensic examination narrative.
 *
 * Architecture note (this replaced a single 600vh position:sticky section
 * whose ~20 simultaneous useTransform subscriptions plus a backdrop-blur
 * nav were the measured source of scroll jank — not the concept). This
 * version keeps exactly ONE thing continuously scroll-scrubbed: the lens
 * object's position/scale/rotation, which is what needs to read as one
 * physical instrument moving through the scene. Phase CONTENT is discrete:
 * `useMotionValueEvent` watches scroll progress and updates a plain state
 * index only when the active phase actually changes (not every frame), so
 * at most one phase's content tree is mounted and animating via
 * AnimatePresence at a time, instead of every phase's content existing
 * simultaneously at variable opacity. The nav's blur was replaced with a
 * flat panel color for the same reason (see index.css .nav-panel).
 */

const PHASES = [
  { key: 'input', label: 'Input', weight: 95, icon: Fingerprint },
  { key: 'inspection', label: 'Inspection', weight: 120, icon: ScanSearch },
  { key: 'signal', label: 'Signal Extraction', weight: 95, icon: Radio },
  { key: 'crossmodal', label: 'Cross-Modal Analysis', weight: 95, icon: GitMerge },
  { key: 'evidence', label: 'Evidence', weight: 100, icon: ClipboardList },
  { key: 'confidence', label: 'Confidence', weight: 95, icon: Gauge },
  { key: 'verdict', label: 'Verdict', weight: 120, icon: ShieldCheck },
]
const TOTAL_WEIGHT = PHASES.reduce((s, p) => s + p.weight, 0)
// Cumulative [start, end) of each phase as a 0-1 fraction of total scroll.
const RANGES = (() => {
  let acc = 0
  return PHASES.map((p) => {
    const start = acc / TOTAL_WEIGHT
    acc += p.weight
    return [start, acc / TOTAL_WEIGHT]
  })
})()
const CENTERS = RANGES.map(([s, e]) => (s + e) / 2)

// Lens transform per phase, sampled at each phase's center and interpolated
// smoothly between — the one continuous physical journey. cx/cy in vw/vh,
// scale relative to the 200px base, rotate in degrees.
const CX_VW = [14, -16, -28, 24, 30, -6, 0]
const CY_VH = [-2, 6, -14, 4, -10, -4, -24]
const SCALE = [1.0, 0.8, 0.5, 0.46, 0.4, 1.15, 2.3]
const ROTATE = [-10, -2, 14, -10, 6, -4, 0]
const BASE_LENS_PX = 200

const SpecimenTag = ({ children }) => (
  <span className="tl-hud-label inline-flex items-center gap-2 !text-[9px]">
    <span className="w-1 h-1 rounded-full bg-brass" />
    {children}
  </span>
)

/* ------------------------------------------------------------------ */
/* Per-phase OUTSIDE-the-lens compositions — each materially different, */
/* not a repeated [text][object] template.                             */
/* ------------------------------------------------------------------ */

const InputPhase = () => (
  <div className="h-full flex items-center px-6 md:px-12 lg:px-16">
    <div className="max-w-[560px]">
      <h1 className="font-serif text-display-xl text-bone mb-6">
        The instrument<br />for what's <span className="italic text-brass">real.</span>
      </h1>
      <p className="text-[17px] leading-[1.7] text-bone-dim max-w-[440px] mb-8">
        Drop in an image, a video, or a piece of audio. What follows is a real examination —
        signal by signal — not a spinner and a guess.
      </p>
      <div className="flex items-center gap-3 tl-hud-label">
        <span className="w-8 h-px bg-brass" />
        Scroll to begin the examination
      </div>
    </div>
  </div>
)

const InspectionPhase = () => (
  <div className="h-full flex items-end md:items-center justify-end px-6 md:px-12 lg:px-16 pb-24 md:pb-0">
    <div className="max-w-[420px] text-right">
      <h2 className="font-serif text-display-l text-bone mb-4">
        Every pixel, examined.
      </h2>
      <p className="text-[15px] leading-[1.7] text-bone-dim">
        The lens moves across the file's own surface — compression seams, sensor noise,
        the places a synthesis pipeline leaves a seam.
      </p>
    </div>
  </div>
)

const SIGNAL_LABELS = ['Compression artifact', 'Frequency deviation', 'Pixel structure', 'Metadata trace']
const SignalPhase = () => (
  <div className="h-full flex flex-col items-center justify-center px-6 md:px-12 lg:px-16">
    <h2 className="font-serif text-display-l text-bone mb-10 text-center">
      Signals, drawn out.
    </h2>
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-8 w-full max-w-[720px]">
      {SIGNAL_LABELS.map((label, i) => (
        <Reveal key={label} delay={i * 0.12} y={10}>
          <div className="border-t border-line pt-3">
            <p className="tl-figure text-[11px] text-brass mb-1">{String(i + 1).padStart(2, '0')}</p>
            <p className="text-[13px] text-bone-dim leading-snug">{label}</p>
          </div>
        </Reveal>
      ))}
    </div>
  </div>
)

const CrossModalPhase = () => {
  const nodes = [
    [18, 18], [82, 22], [14, 82], [84, 80],
  ]
  return (
    <div className="h-full flex flex-col items-center justify-center px-6 md:px-12 lg:px-16">
      <h2 className="font-serif text-display-l text-bone mb-10 text-center max-w-[600px]">
        Independent signals.<br /><span className="italic text-brass">One convergence.</span>
      </h2>
      <svg viewBox="0 0 100 100" className="w-full max-w-[380px] aspect-square" aria-hidden="true">
        {nodes.map(([x, y], i) => (
          <line key={i} x1={x} y1={y} x2={50} y2={50} stroke="rgba(200,147,97,0.35)" strokeWidth="0.6" />
        ))}
        {nodes.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="2.4" className="fill-bone-faint" />
        ))}
        <circle cx="50" cy="50" r="4.5" fill="none" className="stroke-brass" strokeWidth="1.4" />
        <circle cx="50" cy="50" r="1.6" className="fill-brass" />
      </svg>
    </div>
  )
}

const EVIDENCE_ROWS = [
  ['Signal consistency', 'Nominal'],
  ['Compression profile', 'Irregular'],
  ['Frequency response', 'Flagged'],
  ['Structural continuity', 'Nominal'],
]
const EvidencePhase = () => (
  <div className="h-full flex items-center justify-start px-6 md:px-12 lg:px-16">
    <div className="w-full max-w-[440px] bg-panel/60 border border-line rounded-[4px] p-6 md:p-8 tl-ticks">
      <div className="flex items-baseline justify-between mb-6">
        <SpecimenTag>Evidence log</SpecimenTag>
        <span className="tl-figure text-[10px] text-bone-faint">04 / 04 measured</span>
      </div>
      {EVIDENCE_ROWS.map(([label, value], i) => (
        <Reveal key={label} delay={i * 0.1} y={8}>
          <div className="flex items-center justify-between py-3 border-b border-line last:border-b-0">
            <span className="text-[13px] text-bone-dim">{label}</span>
            <span className="tl-figure text-[12px] text-bone uppercase tracking-wide">{value}</span>
          </div>
        </Reveal>
      ))}
    </div>
  </div>
)

const ConfidencePhase = () => (
  <div className="h-full flex items-center justify-start px-6 md:px-12 lg:px-16">
    <div className="max-w-[440px]">
      <h2 className="font-serif text-display-l text-bone mb-5">
        Not one signal.<br />All of them.
      </h2>
      <p className="text-[15px] leading-[1.7] text-bone-dim">
        Every extracted signal resolves into a single, honestly-scored confidence —
        shown, never just asserted.
      </p>
    </div>
  </div>
)

const VerdictPhase = () => (
  <div className="h-full flex flex-col items-center justify-end pb-16 md:pb-12 px-6 md:px-12 lg:px-16 text-center">
    <h2 className="font-serif text-display-l text-bone">
      Analysis complete.
    </h2>
    <p className="text-[15px] leading-[1.7] text-bone-dim max-w-[440px] mt-4">
      A clear, confidence-scored conclusion — never a guess dressed up as certainty.
      Your own result appears the same way.
    </p>
  </div>
)

const OUTSIDE = [InputPhase, InspectionPhase, SignalPhase, CrossModalPhase, EvidencePhase, ConfidencePhase, VerdictPhase]

/* ------------------------------------------------------------------ */
/* Inside-the-lens content — kept minimal; the lens magnifies the      */
/* scan field texture, not an illustration.                            */
/* ------------------------------------------------------------------ */

const ScanField = ({ active }) => (
  <div
    className="absolute inset-0"
    style={{
      backgroundImage:
        'radial-gradient(rgba(200,147,97,0.9) 1px, transparent 1.2px)',
      backgroundSize: '7px 7px',
      filter: active ? 'invert(1) brightness(1.4)' : 'none',
    }}
  />
)

const VerdictInside = () => (
  <div className="absolute inset-0 flex items-center justify-center">
    <div className="scale-[0.62]">
      <VerdictSeal verdict="UNCERTAIN" size={96} />
    </div>
  </div>
)

/* ------------------------------------------------------------------ */
/* Desktop: discrete phase sections, one continuously scroll-scrubbed  */
/* lens.                                                                */
/* ------------------------------------------------------------------ */

const DesktopLensExperience = () => {
  const containerRef = useRef(null)
  const { scrollYProgress } = useScroll({ target: containerRef, offset: ['start start', 'end end'] })
  const [phase, setPhase] = useState(0)

  useMotionValueEvent(scrollYProgress, 'change', (v) => {
    const idx = RANGES.findIndex(([s, e]) => v >= s && v < e)
    const clamped = idx === -1 ? (v >= 1 ? PHASES.length - 1 : 0) : idx
    setPhase((prev) => (prev === clamped ? prev : clamped))
  })

  // Raw scroll-derived targets, run through a light spring so the lens
  // trails and settles like a physical object being moved rather than
  // snapping exactly to the scrollbar — inertia, not decoration.
  const cxRaw = useTransform(scrollYProgress, CENTERS, CX_VW)
  const cyRaw = useTransform(scrollYProgress, CENTERS, CY_VH)
  const scaleRaw = useTransform(scrollYProgress, CENTERS, SCALE)
  const rotateRaw = useTransform(scrollYProgress, CENTERS, ROTATE)
  const springCfg = { stiffness: 120, damping: 18, mass: 0.6 }
  const cx = useSpring(cxRaw, springCfg)
  const cy = useSpring(cyRaw, springCfg)
  const scale = useSpring(scaleRaw, { ...springCfg, damping: 14 })
  const rotate = useSpring(rotateRaw, springCfg)
  const lensTransform = useMotionTemplate`translate(calc(-50% + ${cx}vw), calc(-50% + ${cy}vh)) scale(${scale})`

  const inspectionActive = phase === 1
  const Outside = OUTSIDE[phase]

  return (
    <section ref={containerRef} className="relative bg-ground" style={{ height: `${TOTAL_WEIGHT}vh` }}>
      <div className="sticky top-0 h-screen overflow-hidden pt-[72px]">
        <div className="absolute inset-0 tl-inspection-grid opacity-70" />

        <div className="absolute inset-0">
          <AnimatePresence mode="wait">
            <motion.div
              key={PHASES[phase].key}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
              className="absolute inset-0"
            >
              <Outside />
            </motion.div>
          </AnimatePresence>
        </div>

        <motion.div
          className="absolute left-1/2 top-1/2 z-30 pointer-events-none"
          style={{ width: BASE_LENS_PX, height: BASE_LENS_PX, transform: lensTransform, willChange: 'transform' }}
        >
          <motion.div className="absolute inset-0" style={{ rotate }}>
            <LensGlass size={BASE_LENS_PX}>
              <ScanField active={inspectionActive} />
              {phase === 6 && <VerdictInside />}
            </LensGlass>
          </motion.div>
        </motion.div>

        <div className="absolute bottom-8 left-0 right-0 z-40 px-6 md:px-12 lg:px-16">
          <div className="max-w-[1400px] mx-auto">
            <div className="relative h-px bg-line-strong mb-4">
              <motion.div className="absolute top-0 left-0 h-full bg-brass origin-left" style={{ scaleX: scrollYProgress }} />
            </div>
            <div className="flex items-center gap-2 tl-hud-label flex-wrap">
              {PHASES.map((p, i) => (
                <React.Fragment key={p.key}>
                  {i > 0 && <span className="opacity-30">/</span>}
                  <span className={i === phase ? 'text-brass' : 'opacity-40'}>{p.label}</span>
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>

        <a
          href="#verify-now"
          className="absolute top-20 right-6 md:right-12 z-40 flex items-center gap-1.5 tl-hud-label !text-brass hover:!text-bone transition-colors duration-300"
        >
          Skip to verify
          <ChevronDown size={12} strokeWidth={2} />
        </a>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ */
/* Mobile/tablet: same seven phases, flowing — a deliberate mobile      */
/* composition, not the desktop layout stacked. Each phase gets a      */
/* compact version of its own distinct visual, no pinning/scroll-scrub.*/
/* ------------------------------------------------------------------ */

const MobilePhaseShell = ({ index, children }) => {
  const { icon: Icon, label } = PHASES[index]
  const lensSize = Math.round(72 + index * 10)
  return (
    <div className="relative py-20 px-6 overflow-hidden bg-ground">
      <div className="absolute inset-0 tl-inspection-grid opacity-70" />
      <div className="relative max-w-[520px] mx-auto">
        <Reveal>
          <div className="flex items-center gap-3 mb-8">
            <div className="w-9 h-9 rounded-full bg-panel border border-line flex items-center justify-center">
              <Icon size={15} strokeWidth={1.5} className="text-brass" />
            </div>
            <span className="tl-hud-label">{String(index + 1).padStart(2, '0')} — {label}</span>
          </div>
        </Reveal>
        {children}
        <Reveal delay={0.15} className="flex justify-center mt-10">
          <LensGlass size={lensSize}>
            <ScanField active={index === 1} />
            {index === 6 && <VerdictInside />}
          </LensGlass>
        </Reveal>
      </div>
    </div>
  )
}

const MobileLensExperience = () => (
  <section className="relative">
    {OUTSIDE.map((Comp, i) => (
      <MobilePhaseShell key={PHASES[i].key} index={i}>
        <Comp />
      </MobilePhaseShell>
    ))}
  </section>
)

/* ------------------------------------------------------------------ */
/* Reduced-motion fallback                                             */
/* ------------------------------------------------------------------ */

const StaticLensExperience = () => (
  <section className="relative py-32 bg-ground">
    <div className="max-w-[900px] mx-auto px-6 text-center">
      <h1 className="font-serif text-display-xl text-bone mb-8">
        The instrument for what's <span className="italic text-brass">real.</span>
      </h1>
      <LensGlass size={140} className="mx-auto mb-16">
        <ScanField />
      </LensGlass>
      <div className="grid sm:grid-cols-3 md:grid-cols-7 gap-6 text-left">
        {PHASES.map((p) => (
          <div key={p.key}>
            <p className="tl-hud-label mb-2">{p.label}</p>
          </div>
        ))}
      </div>
    </div>
  </section>
)

const UploadBridge = ({ children, narrow = false }) => (
  <div id="verify-now" className="relative bg-panel py-24 md:py-28 px-6 md:px-12 lg:px-16 overflow-hidden border-t border-line">
    <div className="absolute inset-0 tl-inspection-grid opacity-40" />
    <div className={`relative mx-auto text-center ${narrow ? 'max-w-[560px]' : 'max-w-[900px]'}`}>
      <p className="tl-hud-label mb-4">Enter the instrument</p>
      <h2 className="font-serif text-display-m text-bone mb-10">Your media. Your verdict.</h2>
      {children}
    </div>
  </div>
)

export default function LensNarrative({ uploadSlot }) {
  const reduced = useReducedMotion()
  const isLarge = useIsLargeScreen()

  if (reduced) {
    return (
      <>
        <StaticLensExperience />
        <UploadBridge>{uploadSlot}</UploadBridge>
      </>
    )
  }

  if (!isLarge) {
    return (
      <>
        <MobileLensExperience />
        <UploadBridge narrow>{uploadSlot}</UploadBridge>
      </>
    )
  }

  return (
    <>
      <DesktopLensExperience />
      <UploadBridge>{uploadSlot}</UploadBridge>
    </>
  )
}
