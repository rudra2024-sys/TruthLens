import { useState, useCallback, useEffect, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { UploadCloud, ImageIcon, Video, Music, Download, RotateCcw, FileWarning } from 'lucide-react'
import { uploadMedia, runDetection, reportUrl } from '../api/client'
import { Eyebrow, Gauge, ScoreRow, VerdictMark, Card } from '../components/ui'
import HeuristicNotice from '../components/StubBadge'

const ACCEPT = {
  'image/jpeg': [], 'image/png': [], 'image/webp': [],
  'video/mp4': [], 'video/quicktime': [], 'video/webm': [],
  'audio/wav': [], 'audio/mpeg': [], 'audio/flac': [],
}

export default function Scan() {
  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState(null)
  const [upload, setUpload] = useState(null)
  const [error, setError] = useState('')
  const [preview, setPreview] = useState(null)
  const objectUrl = useRef(null)

  useEffect(() => () => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
  }, [])

  const processFile = useCallback(async (file) => {
    setResult(null); setError(''); setUpload(null)
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    objectUrl.current = URL.createObjectURL(file)
    setPreview({ url: objectUrl.current, type: file.type })

    try {
      setStatus('uploading'); setProgress(0)
      const { data: up } = await uploadMedia(file, setProgress)
      setUpload(up)
      setStatus('detecting')
      const { data: res } = await runDetection(up.upload_id)
      setResult(res)
      setStatus('done')
    } catch (e) {
      setError(e.message)
      setStatus('error')
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop: files => files[0] && processFile(files[0]),
    accept: ACCEPT, maxFiles: 1, maxSize: 100 * 1024 * 1024,
    disabled: status === 'uploading' || status === 'detecting',
  })

  const reset = () => { setStatus('idle'); setResult(null); setUpload(null); setError(''); setPreview(null) }
  const busy = status === 'uploading' || status === 'detecting'
  const MediaIcon = preview?.type.startsWith('video') ? Video : preview?.type.startsWith('audio') ? Music : ImageIcon

  return (
    <div>
      <Eyebrow>/ 01 · New detection</Eyebrow>
      <h1 className="font-display font-extrabold text-3xl md:text-5xl tracking-tight text-snow mb-2">
        Drop the evidence.
      </h1>
      <p className="text-soft text-sm mb-10 max-w-lg">
        Upload once. We analyze the file bytes, persist the verdict, and give you a downloadable report.
      </p>

      <div className="grid lg:grid-cols-2 gap-6 lg:gap-8">
        <div>
          <Card className={`relative overflow-hidden min-h-[360px] ${isDragActive ? 'border-mint shadow-glow' : ''}`}>
            <div
              {...getRootProps()}
              className={`p-10 text-center cursor-pointer min-h-[360px] flex flex-col items-center justify-center ${busy ? 'cursor-wait' : ''}`}
            >
              <input {...getInputProps()} />
              {busy && (
                <div className="absolute inset-0 pointer-events-none overflow-hidden">
                  <div className="absolute left-0 right-0 h-28 bg-gradient-to-b from-transparent via-mint/20 to-transparent animate-sweep" />
                </div>
              )}

              {preview?.type.startsWith('image') && (
                <img src={preview.url} alt="preview" className="max-h-36 rounded-xl mb-5 object-contain" />
              )}
              {preview && !preview.type.startsWith('image') && <MediaIcon className="w-12 h-12 text-mint mb-5" />}
              {!preview && <UploadCloud className="w-12 h-12 text-soft mb-5" />}

              {status === 'idle' && (
                <>
                  <p className="font-display font-bold text-xl text-snow">
                    {isDragActive ? 'Release to upload' : 'Drag & drop a file'}
                  </p>
                  <p className="text-sm text-soft mt-1">or click to browse</p>
                  <p className="text-[11px] text-soft mt-5">JPG · PNG · WebP · MP4 · MOV · WAV · MP3 · FLAC — max 100MB</p>
                </>
              )}

              {status === 'uploading' && (
                <div className="w-full max-w-[240px] relative z-10">
                  <p className="text-sm font-semibold text-snow mb-2">Uploading — {progress}%</p>
                  <div className="w-full bg-stroke rounded-full h-1.5">
                    <div className="bg-mint h-1.5 rounded-full transition-all" style={{ width: `${progress}%` }} />
                  </div>
                </div>
              )}

              {status === 'detecting' && (
                <div className="relative z-10 text-center">
                  <div className="w-9 h-9 border-2 border-mint border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                  <p className="text-sm font-semibold text-snow">Analyzing file content…</p>
                  <p className="text-xs text-soft mt-1">Waiting on the detection API</p>
                </div>
              )}

              {status === 'done' && <p className="text-sm font-semibold text-mint relative z-10">Scan complete</p>}
              {status === 'error' && (
                <div className="relative z-10">
                  <FileWarning className="w-7 h-7 mx-auto mb-2" style={{ color: '#FF4D6A' }} />
                  <p className="text-sm font-semibold max-w-xs" style={{ color: '#FF4D6A' }}>{error}</p>
                </div>
              )}
            </div>
          </Card>

          {fileRejections.length > 0 && (
            <p className="text-xs mt-2" style={{ color: '#FF4D6A' }}>{fileRejections[0].errors[0].message}</p>
          )}
          {(status === 'done' || status === 'error') && (
            <button type="button" onClick={reset} className="mt-4 flex items-center gap-1.5 text-sm text-soft hover:text-snow transition">
              <RotateCcw className="w-3.5 h-3.5" /> Scan another file
            </button>
          )}
        </div>

        <div>
          <AnimatePresence mode="wait">
            {!result && status !== 'error' && (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <Card className="min-h-[360px] flex items-center justify-center px-8">
                  <p className="text-sm text-soft text-center">Results appear here after a scan finishes.</p>
                </Card>
              </motion.div>
            )}

            {status === 'error' && !result && (
              <motion.div key="err" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <Card className="min-h-[360px] flex flex-col justify-center px-8 border-danger/40" style={{ background: 'rgba(255,77,106,0.06)' }}>
                  <p className="font-display font-bold text-lg mb-2" style={{ color: '#FF4D6A' }}>Scan failed</p>
                  <p className="text-sm text-soft max-w-sm">{error}</p>
                </Card>
              </motion.div>
            )}

            {result && upload && (
              <motion.div key="result" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                <Card className="p-6">
                  <p className="text-xs text-soft mb-5 truncate">
                    {upload.file_name} · {upload.media_type.toUpperCase()} · {upload.file_size_kb.toFixed(0)} KB
                  </p>

                  <div className="flex items-center gap-6 mb-6">
                    <Gauge score={result.confidence_score} verdict={result.verdict} />
                    <VerdictMark verdict={result.verdict} />
                  </div>

                  <div className="mb-5"><HeuristicNotice /></div>

                  <div className="flex flex-wrap gap-4 text-xs text-soft mb-6">
                    {result.model_used && <span>{result.model_used}</span>}
                    {result.processing_time_ms != null && <span>{result.processing_time_ms.toFixed(1)} ms</span>}
                    {result.detected_at && <span>{new Date(result.detected_at).toLocaleString()}</span>}
                  </div>

                  {(result.image_analysis || result.video_analysis || result.audio_analysis) && (
                    <div className="bg-elev2 rounded-xl px-4 py-4 mb-5">
                      <p className="text-[10px] font-bold uppercase tracking-widest text-soft mb-3">Breakdown</p>
                      {result.image_analysis && <>
                        <ScoreRow label="Noise residual" score={result.image_analysis.efficientnet_score} />
                        <ScoreRow label="FFT frequency" score={result.image_analysis.fft_score} />
                      </>}
                      {result.video_analysis && <>
                        <ScoreRow label="Structure entropy" score={result.video_analysis.xception_score} />
                        {result.video_analysis.face_voice_sync != null &&
                          <ScoreRow label="Byte consistency" score={result.video_analysis.face_voice_sync} />}
                      </>}
                      {result.audio_analysis && <>
                        <ScoreRow label="Spectral entropy" score={result.audio_analysis.wav2vec_score} />
                        <ScoreRow label="Amplitude / packing" score={result.audio_analysis.lcnn_score} />
                      </>}
                    </div>
                  )}

                  <a
                    href={reportUrl(upload.upload_id)}
                    download
                    className="flex items-center justify-center gap-2 w-full bg-mint text-canvas text-sm font-bold py-3.5 rounded-xl hover:brightness-110 transition"
                  >
                    <Download className="w-4 h-4" /> Download forensic PDF
                  </a>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
