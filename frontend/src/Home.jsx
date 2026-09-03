import React, { useState, useRef, useCallback } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { uploadMedia, runDetection } from './client'
import Logo from './Logo'

const ArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 8H13" /><path d="M9 4L13 8L9 12" />
  </svg>
)

const UploadIcon = ({ className = '' }) => (
  <svg width="48" height="48" viewBox="0 0 48 48" fill="none" className={className}>
    <path d="M24 8V32" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    <path d="M16 20L24 12L32 20" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M8 36H40" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
)

const UploadZone = () => {
  const [isDragging, setIsDragging] = useState(false)
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)
  const navigate = useNavigate()

  const handleDragOver = useCallback((e) => { e.preventDefault(); setIsDragging(true) }, [])
  const handleDragLeave = useCallback((e) => { e.preventDefault(); setIsDragging(false) }, [])
  const handleDrop = useCallback((e) => {
    e.preventDefault(); setIsDragging(false)
    const f = e.dataTransfer.files[0]; if (f) processFile(f)
  }, [])
  const handleFileSelect = useCallback((e) => {
    const f = e.target.files[0]; if (f) processFile(f)
  }, [])

  const processFile = async (selectedFile) => {
    setFile(selectedFile); setError(null); setResult(null)
    setUploading(true); setProgress(0)
    try {
      const uploadRes = await uploadMedia(selectedFile, (p) => setProgress(p))
      const uploadId = uploadRes.data.upload_id
      setUploading(false); setAnalyzing(true)
      const detectRes = await runDetection(uploadId)
      setAnalyzing(false)
      setResult({ ...detectRes.data, uploadId, filename: selectedFile.name })
    } catch (err) {
      setUploading(false); setAnalyzing(false); setError(err.message)
    }
  }

  const getVerdictStyle = (v) => {
    if (v === 'authentic' || v === 'real') return { text: 'text-[#5A7A5A]', bg: 'bg-[rgba(90,122,90,0.08)]', border: 'border-[rgba(90,122,90,0.2)]', label: 'Authentic' }
    if (v === 'suspicious') return { text: 'text-[#B89A6A]', bg: 'bg-[rgba(184,154,106,0.08)]', border: 'border-[rgba(184,154,106,0.2)]', label: 'Suspicious' }
    return { text: 'text-[#9A5A5A]', bg: 'bg-[rgba(154,90,90,0.08)]', border: 'border-[rgba(154,90,90,0.2)]', label: 'Manipulated' }
  }

  if (result) {
    const confidence = ((result.confidence_score ?? 0) * 100).toFixed(2)
    const verdict = result.verdict || result.label || 'analyzed'
    const style = getVerdictStyle(verdict)
    return (
      <div className="bg-[#FAFAF8] rounded-[12px] p-10 md:p-14 shadow-[0_4px_24px_rgba(26,26,26,0.06)]">
        <div className="flex items-start justify-between mb-10 flex-wrap gap-4">
          <div>
            <p className="text-[11px] tracking-[0.15em] uppercase text-[#8A8580] mb-3">Analysis Complete</p>
            <h3 className="font-serif text-[28px] md:text-[32px] text-[#1A1A1A]">{file?.name}</h3>
          </div>
          <div className={`px-5 py-2 rounded-full border ${style.bg} ${style.border} ${style.text}`}>
            <span className="text-[12px] font-medium tracking-[0.08em] uppercase">{style.label}</span>
          </div>
        </div>
        <div className="mb-10">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[13px] text-[#8A8580]">Confidence</span>
            <span className="font-mono text-[14px] text-[#1A1A1A]">{confidence}%</span>
          </div>
          <div className="h-[3px] bg-[rgba(138,133,128,0.15)] rounded-full overflow-hidden">
            <div className="h-full bg-[#A67B5B] rounded-full transition-all duration-1000" style={{ width: `${confidence}%` }} />
          </div>
        </div>
        <div className="flex items-center gap-4 flex-wrap">
          <button onClick={() => navigate(`/report/${result.uploadId}`)} className="flex items-center gap-2 bg-[#1A1A1A] text-[#F7F5F0] px-6 py-3 rounded-[4px] text-[12px] font-medium tracking-[0.08em] uppercase btn-lift">
            View Full Report <ArrowRight />
          </button>
          <a href={`/api/v1/report/${result.uploadId}`} download className="flex items-center gap-2 text-[12px] font-medium tracking-[0.06em] text-[#8A8580] hover:text-[#1A1A1A] transition-colors duration-300 link-underline px-4 py-3">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"><path d="M8 2V10"/><path d="M4 8L8 12L12 8"/><path d="M2 14H14"/></svg>
            Download PDF
          </a>
          <button onClick={() => { setFile(null); setResult(null); setError(null) }} className="text-[12px] font-medium tracking-[0.06em] text-[#8A8580] hover:text-[#1A1A1A] transition-colors duration-300 ml-auto">
            Analyze another
          </button>
        </div>
      </div>
    )
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => !uploading && !analyzing && fileInputRef.current?.click()}
      className={`relative bg-[#FAFAF8] rounded-[12px] min-h-[480px] flex flex-col items-center justify-center cursor-pointer transition-all duration-500 ${
        isDragging ? 'border-2 border-dashed border-[#A67B5B] bg-[rgba(166,123,91,0.03)]' : 'border border-dashed border-[rgba(138,133,128,0.25)]'
      } ${uploading || analyzing ? 'cursor-default' : ''}`}
    >
      <input ref={fileInputRef} type="file" accept="image/*,video/*,audio/*" onChange={handleFileSelect} className="hidden" />
      <div className="absolute top-6 left-6 w-3 h-3 border-l border-t border-[rgba(166,123,91,0.2)]" />
      <div className="absolute top-6 right-6 w-3 h-3 border-r border-t border-[rgba(166,123,91,0.2)]" />
      <div className="absolute bottom-6 left-6 w-3 h-3 border-l border-b border-[rgba(166,123,91,0.2)]" />
      <div className="absolute bottom-6 right-6 w-3 h-3 border-r border-b border-[rgba(166,123,91,0.2)]" />

      {uploading && (
        <div className="flex flex-col items-center gap-6 w-full max-w-[320px] px-6">
          <div className="w-16 h-16 rounded-full border-2 border-[rgba(166,123,91,0.15)] border-t-[#A67B5B] animate-spin" />
          <div className="text-center space-y-2">
            <p className="font-serif text-[20px] text-[#1A1A1A]">Uploading</p>
            <p className="text-[13px] text-[#8A8580]">{file?.name}</p>
          </div>
          <div className="w-full h-[2px] bg-[rgba(138,133,128,0.15)] rounded-full overflow-hidden">
            <div className="h-full tl-progress-shimmer rounded-full transition-all duration-300" style={{ width: `${progress}%` }} />
          </div>
          <p className="font-mono text-[12px] text-[#8A8580]">{progress}%</p>
        </div>
      )}

      {analyzing && (
        <div className="flex flex-col items-center gap-6">
          <div className="relative w-20 h-20">
            <div className="absolute inset-0 rounded-full border border-[rgba(166,123,91,0.2)]" />
            <div className="absolute inset-2 rounded-full border border-[rgba(166,123,91,0.3)] tl-scan-ring" />
            <div className="absolute inset-0 flex items-center justify-center">
              <Logo size={28} className="text-[#A67B5B]" />
            </div>
          </div>
          <div className="text-center space-y-2">
            <p className="font-serif text-[24px] text-[#1A1A1A]">Analyzing authenticity</p>
            <p className="text-[14px] text-[#8A8580] italic">This may take a moment</p>
          </div>
        </div>
      )}

      {error && !uploading && !analyzing && (
        <div className="flex flex-col items-center gap-6 px-8 text-center">
          <div className="w-16 h-16 rounded-full bg-[rgba(154,90,90,0.08)] flex items-center justify-center">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#9A5A5A" strokeWidth="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 8V12"/><path d="M12 16H12.01"/></svg>
          </div>
          <div className="space-y-2">
            <p className="font-serif text-[20px] text-[#1A1A1A]">Something went wrong</p>
            <p className="text-[14px] text-[#9A5A5A] max-w-[280px]">{error}</p>
          </div>
          <button onClick={(e) => { e.stopPropagation(); setError(null); setFile(null) }} className="text-[13px] font-medium tracking-[0.06em] text-[#A67B5B] hover:text-[#7A5A3D] transition-colors duration-300 link-underline">
            Try again
          </button>
        </div>
      )}

      {!uploading && !analyzing && !error && (
        <div className="flex flex-col items-center gap-8 text-center px-8">
          <div className="w-20 h-20 rounded-full bg-[rgba(166,123,91,0.06)] flex items-center justify-center transition-transform duration-500 hover:scale-105">
            <UploadIcon className="text-[#B8B0A8]" />
          </div>
          <div className="space-y-3">
            <p className="font-serif text-display-m text-[#1A1A1A]">Drop your media here</p>
            <p className="text-[15px] text-[#8A8580] leading-relaxed max-w-[300px]">or click to browse. We'll analyze its authenticity with precision.</p>
          </div>
          <div className="flex items-center gap-3 flex-wrap justify-center">
            {['JPG', 'PNG', 'MP4', 'WAV', 'MP3'].map((fmt) => (
              <span key={fmt} className="text-[10px] font-mono tracking-[0.05em] uppercase text-[#B8B0A8] px-3 py-1.5 border border-[rgba(138,133,128,0.15)] rounded-[4px]">{fmt}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

const HeroSection = ({ onUploadClick }) => (
  <section className="relative min-h-screen flex items-center pt-[72px] overflow-hidden">
    <div className="absolute inset-0 bg-gradient-to-br from-[#F7F5F0] via-[#FAFAF8] to-[#F0EDE6]" />
    <div className="absolute top-[15%] right-[-5%] font-serif text-[20vw] leading-none text-[#1A1A1A] opacity-[0.03] select-none pointer-events-none">TL</div>
    <div className="relative z-10 max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16 w-full">
      <div className="grid lg:grid-cols-12 gap-8 lg:gap-4 items-center">
        <div className="lg:col-span-5 xl:col-span-5 space-y-8">
          <div className="space-y-6">
            <p className="text-[12px] font-medium tracking-[0.15em] uppercase text-[#A67B5B] animate-fade-in-up">Digital Authenticity Studio</p>
            <h1 className="font-serif text-display-xl text-[#1A1A1A] animate-fade-in-up animate-delay-1">
              See what<br />others<br /><span className="italic text-[#A67B5B]">cannot.</span>
            </h1>
            <p className="text-[18px] leading-[1.7] text-[#8A8580] max-w-[380px] animate-fade-in-up animate-delay-2">
              We analyze images, video, and audio to reveal the truth hidden beneath the surface. Precision meets intuition.
            </p>
          </div>
          <div className="flex items-center gap-6 pt-4 animate-fade-in-up animate-delay-3">
            <button onClick={onUploadClick} className="group flex items-center gap-3 bg-[#1A1A1A] text-[#F7F5F0] px-8 py-4 rounded-[4px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift">
              Verify Media
              <span className="transition-transform duration-300 group-hover:translate-x-1"><ArrowRight /></span>
            </button>
            <Link to="/about" className="text-[13px] font-medium tracking-[0.06em] text-[#8A8580] hover:text-[#1A1A1A] transition-colors duration-300 link-underline">Our mission</Link>
          </div>
          <div className="flex items-center gap-8 pt-8 animate-fade-in-up animate-delay-4">
            <div><p className="font-serif text-[28px] text-[#1A1A1A]">99.7%</p><p className="text-[11px] tracking-[0.1em] uppercase text-[#8A8580] mt-1">Accuracy</p></div>
            <div className="w-px h-10 bg-[rgba(138,133,128,0.2)]" />
            <div><p className="font-serif text-[28px] text-[#1A1A1A]">2.4M+</p><p className="text-[11px] tracking-[0.1em] uppercase text-[#8A8580] mt-1">Files analyzed</p></div>
            <div className="w-px h-10 bg-[rgba(138,133,128,0.2)]" />
            <div><p className="font-serif text-[28px] text-[#1A1A1A]">&lt;3s</p><p className="text-[11px] tracking-[0.1em] uppercase text-[#8A8580] mt-1">Per analysis</p></div>
          </div>
        </div>
        <div className="lg:col-span-7 xl:col-span-7 lg:pl-8">
          <div className="relative animate-fade-in-up animate-delay-2">
            <div className="absolute -top-4 -left-4 w-24 h-24 border-l border-t border-[rgba(166,123,91,0.3)]" />
            <div className="absolute -bottom-4 -right-4 w-24 h-24 border-r border-b border-[rgba(166,123,91,0.3)]" />
            <UploadZone />
          </div>
        </div>
      </div>
    </div>
    <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 animate-fade-in-up animate-delay-5">
      <span className="text-[10px] tracking-[0.2em] uppercase text-[#B8B0A8]">Scroll</span>
      <div className="w-px h-8 bg-gradient-to-b from-[#B8B0A8] to-transparent" />
    </div>
  </section>
)

const MissionSection = () => (
  <section className="relative py-32 md:py-40 bg-[#1A1A1A] tl-noise-dark overflow-hidden">
    <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16 relative z-10">
      <div className="grid lg:grid-cols-12 gap-12 lg:gap-8">
        <div className="lg:col-span-7">
          <p className="text-[11px] tracking-[0.2em] uppercase text-[#A67B5B] mb-8">Our Purpose</p>
          <h2 className="font-serif text-[clamp(2rem,4vw,3.5rem)] leading-[1.15] text-[#F7F5F0] mb-8">
            In an age where seeing<br />is no longer believing,<br /><span className="italic text-[#A67B5B]">we restore clarity.</span>
          </h2>
          <p className="text-[17px] leading-[1.8] text-[#8A8580] max-w-[480px]">
            Deepfakes, synthetic media, and AI-generated content are reshaping our relationship with truth. TruthLens exists to give you confidence in what you see, hear, and share.
          </p>
        </div>
        <div className="lg:col-span-5 lg:pl-8 flex flex-col justify-center">
          <div className="space-y-10">
            {[
              { num: '01', title: 'Precision', desc: 'Every pixel, every frequency, every frame examined with algorithmic rigor.' },
              { num: '02', title: 'Transparency', desc: 'We show you exactly how we reached our conclusion. No black boxes.' },
              { num: '03', title: 'Privacy', desc: 'Your media is analyzed securely and never stored without your consent.' },
            ].map((item) => (
              <div key={item.num} className="group">
                <div className="flex items-baseline gap-4 mb-2">
                  <span className="font-mono text-[11px] text-[#A67B5B] tracking-wider">{item.num}</span>
                  <h3 className="font-serif text-[22px] text-[#F7F5F0] group-hover:text-[#A67B5B] transition-colors duration-500">{item.title}</h3>
                </div>
                <p className="text-[14px] leading-[1.7] text-[#8A8580] pl-10">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
    <div className="absolute bottom-[-5%] right-[5%] font-serif text-[30vw] leading-none text-[#F7F5F0] opacity-[0.02] select-none pointer-events-none">01</div>
  </section>
)

const ProcessSection = () => {
  const steps = [
    { num: '01', title: 'Upload', desc: 'Drag and drop any image, video, or audio file. We support all major formats.' },
    { num: '02', title: 'Analyze', desc: 'Our models examine metadata, visual artifacts, audio signatures, and deepfake traces.' },
    { num: '03', title: 'Understand', desc: 'Receive a clear verdict with detailed explainability — heatmaps, confidence scores, and reasoning.' },
  ]
  return (
    <section className="relative py-32 md:py-40 bg-[#F7F5F0]">
      <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
        <div className="mb-20">
          <p className="text-[11px] tracking-[0.2em] uppercase text-[#A67B5B] mb-4">How it works</p>
          <h2 className="font-serif text-[clamp(2rem,4vw,3rem)] leading-[1.15] text-[#1A1A1A]">
            Three steps to<br /><span className="italic">certainty.</span>
          </h2>
        </div>
        <div className="grid md:grid-cols-3 gap-8 lg:gap-12">
          {steps.map((step, i) => (
            <div key={step.num} className="group relative">
              {i < steps.length - 1 && <div className="hidden md:block absolute top-8 left-[60%] w-[80%] h-px bg-[rgba(138,133,128,0.15)]" />}
              <div className="space-y-6">
                <div className="flex items-center gap-4">
                  <span className="font-mono text-[11px] tracking-wider text-[#A67B5B]">{step.num}</span>
                  <div className="w-12 h-px bg-[rgba(166,123,91,0.3)]" />
                </div>
                <h3 className="font-serif text-[28px] text-[#1A1A1A] group-hover:text-[#A67B5B] transition-colors duration-500">{step.title}</h3>
                <p className="text-[15px] leading-[1.7] text-[#8A8580]">{step.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

const CTASection = () => (
  <section className="relative py-32 md:py-40 bg-[#F0EDE6] overflow-hidden">
    <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
      <div className="max-w-[700px]">
        <h2 className="font-serif text-[clamp(2.5rem,5vw,4rem)] leading-[1.1] text-[#1A1A1A] mb-8">
          Ready to see<br />the truth?
        </h2>
        <p className="text-[18px] leading-[1.7] text-[#8A8580] mb-10 max-w-[480px]">
          Join thousands of journalists, researchers, and everyday users who trust TruthLens to verify what matters.
        </p>
        <div className="flex items-center gap-6">
          <Link to="/verify" className="group flex items-center gap-3 bg-[#1A1A1A] text-[#F7F5F0] px-8 py-4 rounded-[4px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift">
            Start Verifying
            <span className="transition-transform duration-300 group-hover:translate-x-1"><ArrowRight /></span>
          </Link>
          <Link to="/about" className="text-[13px] font-medium tracking-[0.06em] text-[#8A8580] hover:text-[#1A1A1A] transition-colors duration-300 link-underline">Learn more</Link>
        </div>
      </div>
    </div>
    <div className="absolute top-1/2 right-[10%] -translate-y-1/2 hidden lg:block">
      <div className="w-64 h-64 rounded-full border border-[rgba(166,123,91,0.1)] flex items-center justify-center">
        <div className="w-48 h-48 rounded-full border border-[rgba(166,123,91,0.15)] flex items-center justify-center">
          <Logo size={40} className="text-[rgba(166,123,91,0.3)]" />
        </div>
      </div>
    </div>
  </section>
)

export default function Home() {
  const uploadRef = useRef(null)
  const scrollToUpload = () => uploadRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })

  return (
    <div className="min-h-screen bg-[#F7F5F0]">
      <div className="tl-grain" />
      <HeroSection onUploadClick={scrollToUpload} />
      <div ref={uploadRef} />
      <MissionSection />
      <ProcessSection />
      <CTASection />
    </div>
  )
}
