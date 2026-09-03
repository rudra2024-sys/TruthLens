import React, { useState, useRef, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { uploadMedia, runDetection } from './api/client'
import Logo from './Logo'

const ArrowRight = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M3 8H13" /><path d="M9 4L13 8L9 12" />
  </svg>
)

const DownloadIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M8 2V10" /><path d="M4 8L8 12L12 8" /><path d="M2 14H14" />
  </svg>
)

const ImageIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <rect x="2" y="3" width="16" height="14" rx="2" /><circle cx="7" cy="8" r="1.5" /><path d="M2 14L7 9L11 13L14 10L18 14" />
  </svg>
)

const VideoIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <rect x="2" y="4" width="16" height="12" rx="2" /><polygon points="8,7 8,13 14,10" fill="currentColor" />
  </svg>
)

const AudioIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
    <path d="M10 2V18" /><path d="M6 6C6 6 4 8 4 10C4 12 6 14 6 14" /><path d="M14 6C14 6 16 8 16 10C16 12 14 14 14 14" />
    <path d="M2 10H4" /><path d="M16 10H18" />
  </svg>
)

const UploadIcon = ({ className = '' }) => (
  <svg width="48" height="48" viewBox="0 0 48 48" fill="none" className={className}>
    <path d="M24 8V32" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    <path d="M16 20L24 12L32 20" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M8 36H40" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
)

const getFileIcon = (type) => {
  if (!type) return <UploadIcon className="text-[#B8B0A8]" />
  if (type.startsWith('image/')) return <ImageIcon />
  if (type.startsWith('video/')) return <VideoIcon />
  if (type.startsWith('audio/')) return <AudioIcon />
  return <UploadIcon className="text-[#B8B0A8]" />
}

const getVerdictStyle = (v) => {
  const val = (v || '').toString().toUpperCase()
  if (val === 'REAL' || val === 'AUTHENTIC') return { text: 'text-[#5A7A5A]', bg: 'bg-[rgba(90,122,90,0.08)]', border: 'border-[rgba(90,122,90,0.2)]', label: 'Authentic' }
  if (val === 'UNCERTAIN' || val === 'SUSPICIOUS') return { text: 'text-[#B89A6A]', bg: 'bg-[rgba(184,154,106,0.08)]', border: 'border-[rgba(184,154,106,0.2)]', label: 'Suspicious' }
  return { text: 'text-[#9A5A5A]', bg: 'bg-[rgba(154,90,90,0.08)]', border: 'border-[rgba(154,90,90,0.2)]', label: 'Manipulated' }
}

export default function Verify() {
  const [isDragging, setIsDragging] = useState(false)
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [recentFiles, setRecentFiles] = useState([])
  const fileInputRef = useRef(null)
  const navigate = useNavigate()

  useEffect(() => {
    const saved = localStorage.getItem('truthlens_recent')
    if (saved) {
      try { setRecentFiles(JSON.parse(saved).slice(0, 4)) } catch {}
    }
  }, [])

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
      const res = { ...detectRes.data, uploadId, filename: selectedFile.name, date: new Date().toISOString() }
      setResult(res)
      // Save to recent
      const saved = JSON.parse(localStorage.getItem('truthlens_recent') || '[]')
      const updated = [res, ...saved].slice(0, 10)
      localStorage.setItem('truthlens_recent', JSON.stringify(updated))
      setRecentFiles(updated.slice(0, 4))
    } catch (err) {
      setUploading(false); setAnalyzing(false); setError(err.message)
    }
  }

  const reset = () => { setFile(null); setResult(null); setError(null); setProgress(0) }

  return (
    <div className="min-h-screen bg-[#F7F5F0] pt-[72px]">
      <div className="tl-grain" />

      {/* Hero area - focused, immersive */}
      <section className="relative min-h-[calc(100vh-72px)] flex flex-col items-center justify-center py-16 md:py-24">
        <div className="absolute inset-0 bg-gradient-to-b from-[#F7F5F0] via-[#FAFAF8] to-[#F0EDE6]" />

        <div className="relative z-10 max-w-[900px] mx-auto px-6 w-full">
          {/* Header */}
          {!result && !uploading && !analyzing && (
            <div className="text-center mb-12 animate-fade-in-up">
              <p className="text-[11px] tracking-[0.2em] uppercase text-[#A67B5B] mb-4">Verify Media</p>
              <h1 className="font-serif text-[clamp(2.5rem,5vw,4rem)] leading-[1.1] text-[#1A1A1A] mb-6">
                Drop it here.<br />
                <span className="italic text-[#A67B5B]">We'll do the rest.</span>
              </h1>
              <p className="text-[16px] text-[#8A8580] max-w-[400px] mx-auto">
                Images, videos, and audio files. Our models analyze every layer to determine authenticity.
              </p>
            </div>
          )}

          {/* Upload / Result Zone */}
          <div className="animate-fade-in-up animate-delay-1">
            {!result ? (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !uploading && !analyzing && fileInputRef.current?.click()}
                className={`relative bg-[#FAFAF8] rounded-[16px] min-h-[420px] md:min-h-[480px] flex flex-col items-center justify-center cursor-pointer transition-all duration-500 ${
                  isDragging ? 'border-2 border-dashed border-[#A67B5B] bg-[rgba(166,123,91,0.03)]' : 'border border-dashed border-[rgba(138,133,128,0.25)]'
                } ${uploading || analyzing ? 'cursor-default' : ''}`}
              >
                <input ref={fileInputRef} type="file" accept="image/*,video/*,audio/*" onChange={handleFileSelect} className="hidden" />

                {/* Corner accents */}
                <div className="absolute top-8 left-8 w-4 h-4 border-l border-t border-[rgba(166,123,91,0.2)]" />
                <div className="absolute top-8 right-8 w-4 h-4 border-r border-t border-[rgba(166,123,91,0.2)]" />
                <div className="absolute bottom-8 left-8 w-4 h-4 border-l border-b border-[rgba(166,123,91,0.2)]" />
                <div className="absolute bottom-8 right-8 w-4 h-4 border-r border-b border-[rgba(166,123,91,0.2)]" />

                {uploading && (
                  <div className="flex flex-col items-center gap-6 w-full max-w-[360px] px-6">
                    <div className="relative w-20 h-20">
                      <svg className="w-20 h-20 -rotate-90" viewBox="0 0 80 80">
                        <circle cx="40" cy="40" r="36" fill="none" stroke="rgba(166,123,91,0.1)" strokeWidth="2" />
                        <circle cx="40" cy="40" r="36" fill="none" stroke="#A67B5B" strokeWidth="2" strokeDasharray={`${progress * 2.26} 226`} strokeLinecap="round" className="transition-all duration-300" />
                      </svg>
                      <span className="absolute inset-0 flex items-center justify-center font-mono text-[14px] text-[#1A1A1A]">{progress}%</span>
                    </div>
                    <div className="text-center space-y-1">
                      <p className="font-serif text-[18px] text-[#1A1A1A]">Uploading</p>
                      <p className="text-[13px] text-[#8A8580] truncate max-w-[280px]">{file?.name}</p>
                    </div>
                  </div>
                )}

                {analyzing && (
                  <div className="flex flex-col items-center gap-8">
                    <div className="relative w-24 h-24">
                      <div className="absolute inset-0 rounded-full border border-[rgba(166,123,91,0.15)]" />
                      <div className="absolute inset-2 rounded-full border border-[rgba(166,123,91,0.25)] tl-scan-ring" />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <Logo size={32} className="text-[#A67B5B]" />
                      </div>
                    </div>
                    <div className="text-center space-y-2">
                      <p className="font-serif text-[24px] text-[#1A1A1A]">Analyzing authenticity</p>
                      <p className="text-[14px] text-[#8A8580] italic">Examining metadata, artifacts, and signatures</p>
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
                      <p className="text-[14px] text-[#9A5A5A] max-w-[320px]">{error}</p>
                    </div>
                    <button onClick={(e) => { e.stopPropagation(); reset() }} className="text-[13px] font-medium tracking-[0.06em] text-[#A67B5B] hover:text-[#7A5A3D] transition-colors duration-300 link-underline">
                      Try again
                    </button>
                  </div>
                )}

                {!uploading && !analyzing && !error && (
                  <div className="flex flex-col items-center gap-8 text-center px-8">
                    <div className="w-24 h-24 rounded-full bg-[rgba(166,123,91,0.05)] flex items-center justify-center transition-transform duration-500 hover:scale-105">
                      <UploadIcon className="text-[#B8B0A8]" />
                    </div>
                    <div className="space-y-3">
                      <p className="font-serif text-[28px] md:text-[32px] text-[#1A1A1A]">Drop your file here</p>
                      <p className="text-[15px] text-[#8A8580] leading-relaxed max-w-[320px]">or click anywhere in this area to browse</p>
                    </div>
                    <div className="flex items-center gap-3 flex-wrap justify-center">
                      {['JPG', 'PNG', 'MP4', 'MOV', 'WAV', 'MP3'].map((fmt) => (
                        <span key={fmt} className="text-[10px] font-mono tracking-[0.05em] uppercase text-[#B8B0A8] px-3 py-1.5 border border-[rgba(138,133,128,0.15)] rounded-[4px]">{fmt}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-[#FAFAF8] rounded-[16px] p-10 md:p-14 shadow-[0_4px_24px_rgba(26,26,26,0.06)]">
                <div className="flex items-start justify-between mb-10 flex-wrap gap-4">
                  <div>
                    <p className="text-[11px] tracking-[0.15em] uppercase text-[#8A8580] mb-3">Analysis Complete</p>
                    <h3 className="font-serif text-[28px] md:text-[32px] text-[#1A1A1A]">{file?.name}</h3>
                  </div>
                  {(() => {
                    const s = getVerdictStyle(result.verdict || result.label || 'analyzed')
                    return (
                      <div className={`px-5 py-2 rounded-full border ${s.bg} ${s.border} ${s.text}`}>
                        <span className="text-[12px] font-medium tracking-[0.08em] uppercase">{s.label}</span>
                      </div>
                    )
                  })()}
                </div>

                <div className="mb-10">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-[13px] text-[#8A8580]">Confidence</span>
                    <span className="font-mono text-[14px] text-[#1A1A1A]">{((result.confidence_score ?? 0) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-[3px] bg-[rgba(138,133,128,0.15)] rounded-full overflow-hidden">
                    <div className="h-full bg-[#A67B5B] rounded-full transition-all duration-1000" style={{ width: `${((result.confidence_score ?? 0) * 100).toFixed(1)}%` }} />
                  </div>
                </div>

                {/* Details grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-10">
                  <div>
                    <p className="text-[10px] tracking-[0.15em] uppercase text-[#8A8580] mb-1">File Type</p>
                    <p className="font-mono text-[13px] text-[#1A1A1A]">{file?.type?.split('/')[1]?.toUpperCase() || 'UNKNOWN'}</p>
                  </div>
                  <div>
                    <p className="text-[10px] tracking-[0.15em] uppercase text-[#8A8580] mb-1">Size</p>
                    <p className="font-mono text-[13px] text-[#1A1A1A]">{file ? (file.size / 1024 / 1024).toFixed(2) + ' MB' : '--'}</p>
                  </div>
                  <div>
                    <p className="text-[10px] tracking-[0.15em] uppercase text-[#8A8580] mb-1">Upload ID</p>
                    <p className="font-mono text-[13px] text-[#1A1A1A] truncate max-w-[120px]">{result.uploadId?.slice(0, 12)}...</p>
                  </div>
                  <div>
                    <p className="text-[10px] tracking-[0.15em] uppercase text-[#8A8580] mb-1">Date</p>
                    <p className="font-mono text-[13px] text-[#1A1A1A]">{new Date().toLocaleDateString()}</p>
                  </div>
                </div>

                <div className="flex items-center gap-4 flex-wrap">
                  <button onClick={() => navigate(`/report/${result.uploadId}`)} className="flex items-center gap-2 bg-[#1A1A1A] text-[#F7F5F0] px-6 py-3 rounded-[4px] text-[12px] font-medium tracking-[0.08em] uppercase btn-lift">
                    View Full Report <ArrowRight />
                  </button>
                  <a href={`/api/v1/report/${result.uploadId}`} download className="flex items-center gap-2 text-[12px] font-medium tracking-[0.06em] text-[#8A8580] hover:text-[#1A1A1A] transition-colors duration-300 link-underline px-4 py-3">
                    <DownloadIcon /> Download PDF
                  </a>
                  <button onClick={reset} className="text-[12px] font-medium tracking-[0.06em] text-[#8A8580] hover:text-[#1A1A1A] transition-colors duration-300 ml-auto">
                    Analyze another
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Recent scans */}
      {recentFiles.length > 0 && !result && (
        <section className="py-16 md:py-24 bg-[#F0EDE6]">
          <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
            <div className="flex items-baseline justify-between mb-10">
              <p className="text-[11px] tracking-[0.2em] uppercase text-[#A67B5B]">Recent</p>
              <button onClick={() => navigate('/history')} className="text-[12px] font-medium tracking-[0.06em] text-[#8A8580] hover:text-[#1A1A1A] transition-colors duration-300 link-underline">
                View all history
              </button>
            </div>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {recentFiles.map((item, i) => {
                const s = getVerdictStyle(item.verdict || item.label || 'analyzed')
                return (
                  <div key={i} className="group bg-[#FAFAF8] rounded-[8px] p-6 card-hover cursor-pointer" onClick={() => navigate(`/report/${item.uploadId}`)}>
                    <div className="flex items-center gap-3 mb-4">
                      <div className="text-[#8A8580]">{getFileIcon(item.fileType)}</div>
                      <span className={`text-[10px] font-medium tracking-[0.08em] uppercase px-2 py-1 rounded-full border ${s.bg} ${s.border} ${s.text}`}>
                        {s.label}
                      </span>
                    </div>
                    <p className="font-serif text-[16px] text-[#1A1A1A] mb-1 truncate">{item.filename}</p>
                    <p className="font-mono text-[11px] text-[#8A8580]">{new Date(item.date).toLocaleDateString()}</p>
                  </div>
                )
              })}
            </div>
          </div>
        </section>
      )}
    </div>
  )
}
