import React from 'react'
import { Link } from 'react-router-dom'
import Logo from './Logo'
import Reveal from './components/Reveal'

const ArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 8H13" /><path d="M9 4L13 8L9 12" />
  </svg>
)

export default function About() {
  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />

      {/* Hero */}
      <section className="relative pt-16 md:pt-32 pb-20 md:pb-32 overflow-hidden tl-inspection-grid">
        <div className="absolute top-[10%] left-[-10%] font-serif text-[25vw] leading-none text-bone opacity-[0.02] select-none pointer-events-none">
          About
        </div>
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16 relative z-10">
          <div className="grid lg:grid-cols-12 gap-12">
            <div className="lg:col-span-7">
              <h1 className="font-serif text-display-xl text-bone mb-8">
                We built TruthLens<br />
                because truth<br />
                <span className="italic text-brass">deserves protection.</span>
              </h1>
            </div>
            <div className="lg:col-span-5 lg:pt-20">
              <p className="text-[17px] leading-[1.8] text-bone-dim">
                In 2023, a team of researchers, engineers, and designers came together with a shared
                concern: the tools to deceive were becoming more powerful than the tools to detect deception.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* The Problem */}
      <section className="py-24 md:py-32 bg-panel tl-noise-dark relative overflow-hidden border-y border-line">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16 relative z-10">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <Reveal>
              <div>
                <h2 className="font-serif text-display-l text-bone mb-8">
                  The line between<br />
                  <span className="italic text-brass">real and synthetic</span><br />
                  has never been thinner.
                </h2>
              </div>
            </Reveal>
            <Reveal delay={0.1}>
              <div className="space-y-8">
                <p className="text-[16px] leading-[1.8] text-bone-dim">
                  Deepfake videos can place anyone in any situation. AI-generated images are indistinguishable
                  from photographs. Synthetic audio can clone a voice with just a few seconds of sample data.
                </p>
                <p className="text-[16px] leading-[1.8] text-bone-dim">
                  For journalists, this means verifying sources becomes harder. For researchers, it means
                  data integrity is at risk. For everyday people, it means not knowing what to trust.
                </p>
              </div>
            </Reveal>
          </div>
        </div>
        <div className="absolute bottom-[-5%] right-[5%] font-serif text-[25vw] leading-none text-bone opacity-[0.02] select-none pointer-events-none">02</div>
      </section>

      {/* Our Approach */}
      <section className="py-24 md:py-32">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <div className="grid lg:grid-cols-12 gap-12">
            <Reveal className="lg:col-span-4">
              <div>
                <h2 className="font-serif text-display-l text-bone">
                  How we<br />
                  <span className="italic text-brass">see through.</span>
                </h2>
              </div>
            </Reveal>
            <div className="lg:col-span-8 lg:pl-12">
              <div className="space-y-12">
                {[
                  {
                    title: 'Multi-Modal Analysis',
                    desc: "We do not look at just one signal. Our system examines visual artifacts, audio inconsistencies, metadata anomalies, and compression traces simultaneously. Each layer tells part of the story.",
                  },
                  {
                    title: 'Explainable Results',
                    desc: "Black-box AI breeds distrust. That is why every TruthLens report includes confidence breakdowns and plain-language explanations of what we found and why.",
                  },
                  {
                    title: 'Continuous Learning',
                    desc: 'Synthetic media evolves daily. Our models are refined against the latest deepfake techniques to stay ahead of emerging manipulation methods.',
                  },
                ].map((item, i) => (
                  <Reveal key={i} delay={i * 0.08}>
                    <div className="group flex gap-6">
                      <span className="tl-figure text-[11px] text-bone-faint pt-2 w-6 shrink-0">{String(i + 1).padStart(2, '0')}</span>
                      <div>
                        <h3 className="font-serif text-[22px] text-bone group-hover:text-brass transition-colors duration-500 mb-3">
                          {item.title}
                        </h3>
                        <p className="text-[15px] leading-[1.7] text-bone-dim max-w-[560px]">
                          {item.desc}
                        </p>
                      </div>
                    </div>
                  </Reveal>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Who We Serve */}
      <section className="py-24 md:py-32 border-t border-line">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <Reveal>
            <div className="mb-16">
              <h2 className="font-serif text-display-l text-bone">
                Built for those who<br />
                <span className="italic text-brass">verify before they trust.</span>
              </h2>
            </div>
          </Reveal>
          <div className="grid md:grid-cols-3 gap-8 lg:gap-12">
            {[
              {
                title: 'Journalists',
                desc: 'Verify sources, authenticate images, and maintain editorial integrity in an era of synthetic media.',
              },
              {
                title: 'Researchers',
                desc: 'Ensure data authenticity in academic studies, forensic investigations, and scientific publications.',
              },
              {
                title: 'Everyone',
                desc: "Before you share that viral video or retweet that shocking image — know what you are spreading.",
              },
            ].map((item, i) => (
              <Reveal key={i} delay={i * 0.08}>
                <div className="group">
                  <div className="w-12 h-px bg-brass mb-6" />
                  <h3 className="font-serif text-[22px] text-bone mb-4 group-hover:text-brass transition-colors duration-500">
                    {item.title}
                  </h3>
                  <p className="text-[15px] leading-[1.7] text-bone-dim">{item.desc}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 md:py-32 bg-panel tl-noise-dark relative overflow-hidden border-t border-line">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16 relative z-10">
          <Reveal>
            <div className="max-w-[700px]">
              <h2 className="font-serif text-display-l text-bone mb-8">
                Start seeing<br />
                <span className="italic text-brass">clearly.</span>
              </h2>
              <p className="text-[17px] leading-[1.8] text-bone-dim mb-10 max-w-[480px]">
                TruthLens is free to use for individual verifications. No account required for your first scan.
              </p>
              <div className="flex items-center gap-6">
                <Link to="/verify" className="group flex items-center gap-3 bg-brass text-ground px-8 py-4 rounded-[3px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift">
                  Verify Now
                  <span className="transition-transform duration-300 group-hover:translate-x-1"><ArrowRight /></span>
                </Link>
                <Link to="/history" className="text-[13px] font-medium tracking-[0.06em] text-bone-dim hover:text-bone transition-colors duration-300 link-underline">
                  View history
                </Link>
              </div>
            </div>
          </Reveal>
        </div>
        <div className="absolute bottom-[-10%] right-[5%] font-serif text-[25vw] leading-none text-bone opacity-[0.02] select-none pointer-events-none">TL</div>
      </section>
    </div>
  )
}
