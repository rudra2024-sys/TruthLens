import React from 'react'
import { Link } from 'react-router-dom'
import Logo from './Logo'

const ArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 8H13" /><path d="M9 4L13 8L9 12" />
  </svg>
)

export default function About() {
  return (
    <div className="min-h-screen bg-[#F7F5F0] pt-[72px]">
      <div className="tl-grain" />

      {/* Hero */}
      <section className="relative pt-16 md:pt-32 pb-20 md:pb-32 overflow-hidden">
        <div className="absolute top-[10%] left-[-10%] font-serif text-[25vw] leading-none text-[#1A1A1A] opacity-[0.02] select-none pointer-events-none">
          About
        </div>
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16 relative z-10">
          <div className="grid lg:grid-cols-12 gap-12">
            <div className="lg:col-span-7">
              <p className="text-[11px] tracking-[0.2em] uppercase text-[#A67B5B] mb-6 animate-fade-in-up">Our Story</p>
              <h1 className="font-serif text-[clamp(2.5rem,5vw,4.5rem)] leading-[1.05] text-[#1A1A1A] mb-8 animate-fade-in-up animate-delay-1">
                We built TruthLens<br />
                because truth<br />
                <span className="italic text-[#A67B5B]">deserves protection.</span>
              </h1>
            </div>
            <div className="lg:col-span-5 lg:pt-20">
              <p className="text-[17px] leading-[1.8] text-[#8A8580] animate-fade-in-up animate-delay-2">
                In 2023, a team of researchers, engineers, and designers came together with a shared 
                concern: the tools to deceive were becoming more powerful than the tools to detect deception.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* The Problem */}
      <section className="py-24 md:py-32 bg-[#1A1A1A] tl-noise-dark relative overflow-hidden">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16 relative z-10">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <p className="text-[11px] tracking-[0.2em] uppercase text-[#A67B5B] mb-6">The Challenge</p>
              <h2 className="font-serif text-[clamp(2rem,4vw,3rem)] leading-[1.15] text-[#F7F5F0] mb-8">
                The line between<br />
                <span className="italic">real and synthetic</span><br />
                has never been thinner.
              </h2>
            </div>
            <div className="space-y-8">
              <p className="text-[16px] leading-[1.8] text-[#8A8580]">
                Deepfake videos can place anyone in any situation. AI-generated images are indistinguishable 
                from photographs. Synthetic audio can clone a voice with just a few seconds of sample data.
              </p>
              <p className="text-[16px] leading-[1.8] text-[#8A8580]">
                For journalists, this means verifying sources becomes harder. For researchers, it means 
                data integrity is at risk. For everyday people, it means not knowing what to trust.
              </p>
            </div>
          </div>
        </div>
        <div className="absolute bottom-[-5%] right-[5%] font-serif text-[25vw] leading-none text-[#F7F5F0] opacity-[0.02] select-none pointer-events-none">02</div>
      </section>

      {/* Our Approach */}
      <section className="py-24 md:py-32">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <div className="grid lg:grid-cols-12 gap-12">
            <div className="lg:col-span-4">
              <p className="text-[11px] tracking-[0.2em] uppercase text-[#A67B5B] mb-4">Technology</p>
              <h2 className="font-serif text-[clamp(2rem,4vw,3rem)] leading-[1.15] text-[#1A1A1A]">
                How we<br />
                <span className="italic">see through.</span>
              </h2>
            </div>
            <div className="lg:col-span-8 lg:pl-12">
              <div className="space-y-12">
                {[
                  {
                    title: 'Multi-Modal Analysis',
                    desc: "We do not look at just one signal. Our system examines visual artifacts, audio inconsistencies, metadata anomalies, and compression traces simultaneously. Each layer tells part of the story.",
                  },
                  {
                    title: 'Explainable Results',
                    desc: "Black-box AI breeds distrust. That is why every TruthLens report includes heatmaps, confidence breakdowns, and plain-language explanations of what we found and why.",
                  },
                  {
                    title: 'Continuous Learning',
                    desc: 'Synthetic media evolves daily. Our models are retrained on the latest deepfake techniques, ensuring we stay ahead of emerging manipulation methods.',
                  },
                ].map((item, i) => (
                  <div key={i} className="group">
                    <div className="flex items-baseline gap-4 mb-3">
                      <span className="font-mono text-[11px] text-[#A67B5B]">0{i + 1}</span>
                      <h3 className="font-serif text-[22px] text-[#1A1A1A] group-hover:text-[#A67B5B] transition-colors duration-500">
                        {item.title}
                      </h3>
                    </div>
                    <p className="text-[15px] leading-[1.7] text-[#8A8580] pl-10 max-w-[560px]">
                      {item.desc}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Trust Stats */}
      <section className="py-20 md:py-28 bg-[#F0EDE6]">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-12">
            {[
              { value: '99.7%', label: 'Detection accuracy' },
              { value: '2.4M+', label: 'Files analyzed' },
              { value: '<3s', label: 'Average analysis time' },
              { value: '150+', label: 'Countries served' },
            ].map((stat, i) => (
              <div key={i} className="text-center md:text-left">
                <p className="font-serif text-[clamp(2rem,4vw,3rem)] text-[#1A1A1A] mb-2">{stat.value}</p>
                <p className="text-[11px] tracking-[0.1em] uppercase text-[#8A8580]">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Who We Serve */}
      <section className="py-24 md:py-32">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <div className="mb-16">
            <p className="text-[11px] tracking-[0.2em] uppercase text-[#A67B5B] mb-4">Users</p>
            <h2 className="font-serif text-[clamp(2rem,4vw,3rem)] leading-[1.15] text-[#1A1A1A]">
              Built for those who<br />
              <span className="italic">verify before they trust.</span>
            </h2>
          </div>
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
              <div key={i} className="group">
                <div className="w-12 h-px bg-[#A67B5B] mb-6" />
                <h3 className="font-serif text-[22px] text-[#1A1A1A] mb-4 group-hover:text-[#A67B5B] transition-colors duration-500">
                  {item.title}
                </h3>
                <p className="text-[15px] leading-[1.7] text-[#8A8580]">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 md:py-32 bg-[#1A1A1A] tl-noise-dark relative overflow-hidden">
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16 relative z-10">
          <div className="max-w-[700px]">
            <h2 className="font-serif text-[clamp(2rem,4vw,3.5rem)] leading-[1.1] text-[#F7F5F0] mb-8">
              Start seeing<br />
              <span className="italic text-[#A67B5B]">clearly.</span>
            </h2>
            <p className="text-[17px] leading-[1.8] text-[#8A8580] mb-10 max-w-[480px]">
              TruthLens is free to use for individual verifications. No account required for your first scan.
            </p>
            <div className="flex items-center gap-6">
              <Link to="/verify" className="group flex items-center gap-3 bg-[#A67B5B] text-[#F7F5F0] px-8 py-4 rounded-[4px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift">
                Verify Now
                <span className="transition-transform duration-300 group-hover:translate-x-1"><ArrowRight /></span>
              </Link>
              <Link to="/history" className="text-[13px] font-medium tracking-[0.06em] text-[#8A8580] hover:text-[#F7F5F0] transition-colors duration-300 link-underline">
                View history
              </Link>
            </div>
          </div>
        </div>
        <div className="absolute bottom-[-10%] right-[5%] font-serif text-[25vw] leading-none text-[#F7F5F0] opacity-[0.02] select-none pointer-events-none">TL</div>
      </section>
    </div>
  )
}
