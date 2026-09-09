import React from 'react'
import UploadFlow from './components/UploadFlow'
import { stationTag } from './lib/stations'

const heading = (
  <div className="text-center mb-12">
    <p className="tl-hud-label !text-brass mb-4">{stationTag(2)} — Intake Desk</p>
    <h1 className="font-serif text-display-l text-bone mb-6">
      Place it under<br />
      <span className="italic text-brass">the lens.</span>
    </h1>
    <p className="text-[16px] text-bone-dim max-w-[420px] mx-auto">
      Images, videos, and audio files. Every layer is inspected for signs of manipulation.
    </p>
  </div>
)

export default function Verify() {
  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />

      <section className="relative min-h-[calc(100vh-72px)] flex flex-col items-center justify-center py-16 md:py-24 tl-inspection-grid">
        <div className="absolute inset-0 bg-gradient-to-b from-ground via-ground to-panel/40" />

        <div className="relative z-10 max-w-[900px] mx-auto px-6 w-full animate-fade-in-up animate-delay-1">
          <UploadFlow showRecent heading={heading} />
        </div>
      </section>
    </div>
  )
}
