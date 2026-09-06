import React from 'react'
import { Link } from 'react-router-dom'
import { ScanSearch } from 'lucide-react'
import Reveal from './components/Reveal'
import ScanFrame from './components/ScanFrame'

export default function NotFound() {
  return (
    <div className="min-h-screen bg-ground pt-[72px] flex items-center justify-center px-6 tl-inspection-grid">
      <Reveal>
        <div className="relative text-center max-w-[440px] bg-panel border border-line rounded-[6px] p-12">
          <ScanFrame gap={16} armSize={12} />
          <ScanSearch size={30} strokeWidth={1.25} className="text-bone-faint mx-auto mb-6" />
          <p className="tl-hud-label mb-3">Signal Lost · 404</p>
          <h1 className="font-serif text-[32px] text-bone mb-4">Nothing found here.</h1>
          <p className="text-[15px] text-bone-dim mb-8">
            The page you're looking for doesn't exist or may have moved.
          </p>
          <Link
            to="/"
            className="inline-block px-6 py-3 bg-brass text-ground rounded-[3px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift"
          >
            Back to home
          </Link>
        </div>
      </Reveal>
    </div>
  )
}
