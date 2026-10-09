import React, { useEffect, useRef, useState } from 'react'
import { Camera, X } from 'lucide-react'

/**
 * Self-contained live-camera capture: requests the camera, shows a preview, and hands the
 * parent a single captured frame as a File (JPEG). Requires a secure context (https:// or
 * localhost) — getUserMedia simply isn't available otherwise.
 *
 * Must stop every MediaStream track both on unmount AND immediately after a successful
 * capture — otherwise the browser's camera recording indicator stays on indefinitely, which
 * is the one easy-to-miss correctness bug with this kind of component.
 */
export default function WebcamCapture({ onCapture, onCancel, className = '' }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [error, setError] = useState(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function start() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError('This browser does not support camera access.')
        return
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        setReady(true)
      } catch (err) {
        if (!cancelled) {
          setError(
            err?.name === 'NotAllowedError'
              ? 'Camera access was denied. Allow camera access and try again.'
              : 'Could not access the camera.'
          )
        }
      }
    }
    start()

    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [])

  const stopStream = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
  }

  const capture = () => {
    const video = videoRef.current
    if (!video || !video.videoWidth) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    canvas.toBlob(
      (blob) => {
        stopStream()
        if (blob) {
          onCapture(new File([blob], 'live-capture.jpg', { type: 'image/jpeg' }))
        }
      },
      'image/jpeg',
      0.92
    )
  }

  const handleCancel = () => {
    stopStream()
    onCancel?.()
  }

  return (
    <div className={className}>
      {error ? (
        <div className="flex flex-col items-center gap-4 text-center py-10">
          <p className="text-[14px] text-verdictDanger max-w-[320px]">{error}</p>
          <button
            type="button"
            onClick={handleCancel}
            className="text-[12px] font-medium tracking-[0.06em] text-bone-dim hover:text-bone link-underline"
          >
            Close
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-5">
          <div className="relative w-full max-w-[360px] aspect-[4/3] bg-ground border border-line rounded-[4px] overflow-hidden">
            <video ref={videoRef} muted playsInline className="w-full h-full object-cover scale-x-[-1]" />
            {!ready && (
              <div className="absolute inset-0 flex items-center justify-center text-bone-faint text-[12px]">
                Requesting camera…
              </div>
            )}
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              disabled={!ready}
              onClick={capture}
              className="flex items-center gap-2 bg-brass text-ground px-5 py-2.5 rounded-[3px] text-[12px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-40"
            >
              <Camera size={14} strokeWidth={1.75} /> Capture
            </button>
            <button
              type="button"
              onClick={handleCancel}
              className="flex items-center gap-2 text-[12px] font-medium tracking-[0.06em] text-bone-dim hover:text-bone transition-colors duration-300 px-3 py-2.5"
            >
              <X size={14} strokeWidth={1.75} /> Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
