import React, { useRef } from 'react'
import { Link } from 'react-router-dom'
import { motion, useScroll, useTransform } from 'framer-motion'
import useReducedMotion from './hooks/useReducedMotion'
import Logo from './Logo'
import UploadFlow from './components/UploadFlow'
import Reveal from './components/Reveal'
import LensNarrative from './components/LensNarrative'

const ArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 8H13" /><path d="M9 4L13 8L9 12" />
  </svg>
)

const MissionSection = () => {
  const reduced = useReducedMotion()
  const ref = useRef(null)
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start end', 'end start'] })
  const markY = useTransform(scrollYProgress, [0, 1], [reduced ? 0 : 60, reduced ? 0 : -60])

  return (
  <section ref={ref} className="relative py-32 md:py-40 bg-panel tl-noise-dark overflow-hidden border-t border-line">
    <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16 relative z-10">
      <div className="grid lg:grid-cols-12 gap-12 lg:gap-8">
        <div className="lg:col-span-7">
          <Reveal>
            <h2 className="font-serif text-[clamp(2rem,4vw,3.5rem)] leading-[1.15] text-bone mb-8">
              In an age where seeing<br />is no longer believing,<br /><span className="italic text-brass">we restore clarity.</span>
            </h2>
          </Reveal>
          <Reveal delay={0.08}>
            <p className="text-[17px] leading-[1.8] text-bone-dim max-w-[480px]">
              Deepfakes, synthetic media, and AI-generated content are reshaping our relationship with truth. TruthLens exists to give you confidence in what you see, hear, and share.
            </p>
          </Reveal>
        </div>
        <div className="lg:col-span-5 lg:pl-8 flex flex-col justify-center">
          <div className="space-y-10">
            {[
              { num: '01', title: 'Precision', desc: 'Every pixel, every frequency, every frame examined with algorithmic rigor.' },
              { num: '02', title: 'Transparency', desc: 'We show you exactly what was found and how confident the system is. No black boxes.' },
              { num: '03', title: 'Privacy', desc: 'Your media is analyzed securely and never stored without your consent.' },
            ].map((item, i) => (
              <Reveal key={item.num} delay={i * 0.08}>
                <div className="group">
                  <div className="flex items-baseline gap-4 mb-2">
                    <span className="font-mono text-[11px] text-brass tracking-wider">{item.num}</span>
                    <h3 className="font-serif text-[22px] text-bone group-hover:text-brass transition-colors duration-500">{item.title}</h3>
                  </div>
                  <p className="text-[14px] leading-[1.7] text-bone-dim pl-10">{item.desc}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </div>
    </div>
    <motion.div
      style={{ y: markY }}
      className="absolute bottom-[-5%] right-[5%] font-serif text-[30vw] leading-none text-bone opacity-[0.03] select-none pointer-events-none"
    >
      01
    </motion.div>
  </section>
  )
}

const CTASection = () => {
  const reduced = useReducedMotion()
  const ref = useRef(null)
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start end', 'end start'] })
  const ringRotate = useTransform(scrollYProgress, [0, 1], [0, reduced ? 0 : 40])
  const ringScale = useTransform(scrollYProgress, [0, 0.5, 1], [0.92, 1, 0.92])

  return (
  <section ref={ref} className="relative py-32 md:py-40 bg-ground overflow-hidden border-t border-line">
    <div className="absolute inset-0 tl-inspection-grid opacity-50" />
    <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16 relative">
      <div className="max-w-[700px]">
        <Reveal>
          <h2 className="font-serif text-[clamp(2.5rem,5vw,4rem)] leading-[1.1] text-bone mb-8">
            Ready to see<br />the truth?
          </h2>
        </Reveal>
        <Reveal delay={0.08}>
          <p className="text-[18px] leading-[1.7] text-bone-dim mb-10 max-w-[480px]">
            Join researchers, journalists, and everyday users who trust TruthLens to verify what matters.
          </p>
        </Reveal>
        <Reveal delay={0.16}>
          <div className="flex items-center gap-6">
            <Link to="/verify" className="group flex items-center gap-3 bg-brass text-ground px-8 py-4 rounded-[4px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift">
              Start Verifying
              <span className="transition-transform duration-300 group-hover:translate-x-1"><ArrowRight /></span>
            </Link>
            <Link to="/about" className="text-[13px] font-medium tracking-[0.06em] text-bone-dim hover:text-bone transition-colors duration-300 link-underline">Learn more</Link>
          </div>
        </Reveal>
      </div>
    </div>
    <motion.div
      style={{ rotate: ringRotate, scale: ringScale }}
      className="absolute top-1/2 right-[10%] -translate-y-1/2 hidden lg:block"
    >
      <div className="w-64 h-64 rounded-full border border-brass-deep flex items-center justify-center">
        <div className="w-48 h-48 rounded-full border border-line-strong flex items-center justify-center">
          <Logo size={40} className="text-brass-deep" />
        </div>
      </div>
    </motion.div>
  </section>
  )
}

export default function Home() {
  return (
    <div className="min-h-screen bg-ground">
      <div className="tl-grain" />
      <LensNarrative uploadSlot={<UploadFlow compact />} />
      <MissionSection />
      <CTASection />
    </div>
  )
}
