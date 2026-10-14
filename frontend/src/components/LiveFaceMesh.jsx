import React, { useEffect, useRef } from 'react'
import { FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision'

/**
 * Real-time, every-frame face mesh (added 2026-10-12, on top of the existing FaceOverlay).
 *
 * This is a separate layer from FaceOverlay: FaceOverlay draws the server's actual per-check
 * MTCNN box/landmarks (the real, backend-verified identity signal, refreshed every 5-20s per
 * monitoring.py's cadence) -- that stays exactly as-is and remains the authoritative overlay.
 * This component is purely a live, client-side visual: Google's MediaPipe FaceLandmarker
 * (468 mesh points + iris + 52 expression "blendshape" scores) runs on-device via WebGL/WASM,
 * once per animation frame, directly on the <video> element -- a real on-device detector, not a
 * fabricated animation, but its output never reaches the backend and never feeds a verdict. It
 * exists so the camera preview visibly reacts to head turns and expressions in real time instead
 * of only updating on the backend's own check cadence.
 *
 * Model + wasm runtime load from Google's/jsDelivr's public CDNs on first use (same pattern as
 * the CLIP backbone downloading from the HF Hub on first use elsewhere in this project) -- no
 * local checkpoint to manage, but it does mean this layer silently does nothing without network
 * access to those CDNs (caught below and surfaced as a console warning, never a crash).
 */

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

// A handful of expression blendshapes that read as "alive" on camera -- drives the mesh's glow
// intensity so it visibly reacts to a smile, raised brows, or an open jaw, not just head pose.
const EXPRESSION_SHAPES = ['jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'browInnerUp', 'eyeBlinkLeft', 'eyeBlinkRight']

// Real-time attention monitoring (added 2026-10-12): the backend's own periodic check only
// samples every 5-20s, which means a student could hold up a phone or look away for 8-15s and
// never get sampled at all -- a real, demonstrated gap, not a hypothetical one. This runs every
// frame instead, using landmarks already being computed for the mesh above, so a lapse gets
// caught inside the window where it used to slip through entirely.
//
// A condition must hold CONTINUOUSLY for SUSTAIN_MS before it fires (long enough to ignore a
// blink or an ordinary head turn, short enough that real phone use can't hide inside it), and
// each type has its own COOLDOWN_MS afterwards so a sustained problem logs once, not every frame.
const SUSTAIN_MS = 2000
const COOLDOWN_MS = 8000
// Face-width-extreme landmark indices (standard MediaPipe FaceMesh topology) used for a simple
// 2D horizontal-turn proxy -- deliberately not a full 3D head-pose solve (which would need the
// facial transformation matrix decomposed correctly, real matrix-convention risk for a signal
// that only needs to be "good enough to trigger a real server check", not itself authoritative).
const NOSE_TIP = 1
const LEFT_FACE_EDGE = 234
const RIGHT_FACE_EDGE = 454
const LOOKING_AWAY_RATIO_THRESHOLD = 0.2 // |ratio - 0.5| beyond this reads as a significant turn

export default function LiveFaceMesh({ videoRef, active, color = '#C89361', onSustainedIssue }) {
  const canvasRef = useRef(null)
  const landmarkerRef = useRef(null)
  const rafRef = useRef(null)
  const stoppedRef = useRef(false)
  const conditionSinceRef = useRef({ no_face: null, looking_away: null, multiple_faces: null })
  const lastFiredRef = useRef({ no_face: 0, looking_away: 0, multiple_faces: 0 })
  const onSustainedIssueRef = useRef(onSustainedIssue)
  onSustainedIssueRef.current = onSustainedIssue

  useEffect(() => {
    if (!active) return undefined
    stoppedRef.current = false

    let cancelled = false

    const setup = async () => {
      try {
        const files = await FilesetResolver.forVisionTasks(WASM_BASE)
        const landmarker = await FaceLandmarker.createFromOptions(files, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numFaces: 2, // only need to know "more than one", not track many precisely
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: false,
        })
        if (cancelled) {
          landmarker.close()
          return
        }
        landmarkerRef.current = landmarker
        loop()
      } catch (err) {
        // No network to the model CDN, no WebGL, etc. -- the rest of the session UI (the real,
        // backend-verified FaceOverlay) is unaffected either way, so this stays silent beyond a
        // console note rather than surfacing an error to the student.
        console.warn('[LiveFaceMesh] unavailable:', err?.message || err)
      }
    }

    const loop = () => {
      if (stoppedRef.current) return
      const video = videoRef.current
      const canvas = canvasRef.current
      const landmarker = landmarkerRef.current
      if (video && canvas && landmarker && video.readyState >= 2 && video.videoWidth) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth
          canvas.height = video.videoHeight
        }
        const result = landmarker.detectForVideo(video, performance.now())
        drawResult(canvas, result, color)
        trackAttention(result, conditionSinceRef, lastFiredRef, onSustainedIssueRef)
      }
      rafRef.current = requestAnimationFrame(loop)
    }

    setup()

    return () => {
      cancelled = true
      stoppedRef.current = true
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      if (landmarkerRef.current) {
        landmarkerRef.current.close()
        landmarkerRef.current = null
      }
      const ctx = canvasRef.current?.getContext('2d')
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active])

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
}

function trackAttention(result, conditionSinceRef, lastFiredRef, onSustainedIssueRef) {
  const faces = result?.faceLandmarks || []
  const now = performance.now()

  const current = {
    no_face: faces.length === 0,
    multiple_faces: faces.length > 1,
    looking_away: faces.length === 1 && isLookingAway(faces[0]),
  }

  for (const type of Object.keys(current)) {
    const since = conditionSinceRef.current
    if (!current[type]) {
      since[type] = null
      continue
    }
    if (since[type] === null) {
      since[type] = now
      continue
    }
    const held = now - since[type]
    const sinceLastFired = now - lastFiredRef.current[type]
    if (held >= SUSTAIN_MS && sinceLastFired >= COOLDOWN_MS) {
      lastFiredRef.current[type] = now
      onSustainedIssueRef.current?.(type)
    }
  }
}

function isLookingAway(landmarks) {
  const nose = landmarks[NOSE_TIP]
  const left = landmarks[LEFT_FACE_EDGE]
  const right = landmarks[RIGHT_FACE_EDGE]
  if (!nose || !left || !right) return false
  const width = right.x - left.x
  if (Math.abs(width) < 1e-6) return false
  const ratio = (nose.x - left.x) / width
  return Math.abs(ratio - 0.5) > LOOKING_AWAY_RATIO_THRESHOLD
}

function expressionIntensity(blendshapes) {
  if (!blendshapes?.length) return 0
  const categories = blendshapes[0].categories
  if (!categories) return 0
  let sum = 0
  let n = 0
  for (const cat of categories) {
    if (EXPRESSION_SHAPES.includes(cat.categoryName)) {
      sum += cat.score
      n += 1
    }
  }
  return n ? sum / n : 0
}

function drawResult(canvas, result, color) {
  const ctx = canvas.getContext('2d')
  ctx.save()
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const faces = result?.faceLandmarks
  if (!faces || faces.length === 0) {
    ctx.restore()
    return
  }

  const intensity = expressionIntensity(result.faceBlendshapes)
  const drawingUtils = new DrawingUtils(ctx)

  for (const landmarks of faces) {
    // Fine tesselation -- the dense "net", thin and faint so it reads as a scan mesh.
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
      color: `${color}55`,
      lineWidth: 1,
    })
    // Contours (eyes, brows, lips, face oval) -- heavier, and its opacity/width pulses with the
    // live expression intensity above, so a smile or raised brows visibly brighten the mesh.
    const contourWidth = 1.5 + intensity * 2
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_CONTOURS, {
      color,
      lineWidth: contourWidth,
    })
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color, lineWidth: 1.5 })
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color, lineWidth: 1.5 })
  }

  ctx.restore()
}
