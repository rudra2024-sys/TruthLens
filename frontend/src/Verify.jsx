import React, { useState } from 'react'
import UploadFlow from './components/UploadFlow'
import BatchVerify from './components/BatchVerify'
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

const MODES = [
  { key: 'single', label: 'Single file' },
  { key: 'batch', label: 'Batch (up to 100)' },
]

export default function Verify() {
  const [mode, setMode] = useState('single')

  return (
    <div className="min-h-screen bg-ground pt-[72px]">
      <div className="tl-grain" />

      <section className="relative min-h-[calc(100vh-72px)] flex flex-col items-center justify-center py-16 md:py-24 tl-inspection-grid">
        <div className="absolute inset-0 bg-gradient-to-b from-ground via-ground to-panel/40" />

        <div className="relative z-10 max-w-[900px] mx-auto px-6 w-full animate-fade-in-up animate-delay-1">
          <div role="tablist" aria-label="Scan mode" className="flex justify-center mb-8">
            <div className="inline-flex border border-line rounded-[4px] overflow-hidden">
              {MODES.map(({ key, label }) => (
                <button
                  key={key} type="button" role="tab" aria-selected={mode === key} onClick={() => setMode(key)}
                  className={`px-5 py-2.5 text-[12px] tracking-[0.1em] uppercase transition-colors duration-300 ${
                    mode === key ? 'bg-[rgba(200,147,97,0.12)] text-brass' : 'text-bone-dim hover:text-bone'}`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {mode === 'batch' && (
            <div className="text-center mb-10">
              <p className="tl-hud-label !text-brass mb-4">{stationTag(2)} — Intake Desk</p>
              <h1 className="font-serif text-display-l text-bone mb-6">
                Place them under<br />
                <span className="italic text-brass">the lens.</span>
              </h1>
              <p className="text-[16px] text-bone-dim max-w-[460px] mx-auto">
                Scan many files in one go and download every result as a spreadsheet or a plain-words PDF.
              </p>
            </div>
          )}

          {mode === 'single' ? <UploadFlow showRecent heading={heading} /> : <BatchVerify />}
        </div>
      </section>
    </div>
  )
}
