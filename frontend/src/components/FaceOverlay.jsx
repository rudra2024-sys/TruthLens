import React, { useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

/**
 * Live face-tracking "net" overlay for the monitoring session's camera preview (follow-up
 * item 3: "a net which would cover the face and check for the mismatch between persons").
 *
 * Draws the real per-check MTCNN detection -- the face box and 5-point landmarks already
 * computed server-side for identity matching/head-pose (see backend/app/services/identity/
 * monitoring.py's _compute_signals) -- never a client-side guess or a decorative placeholder.
 * It only updates once per check (every 5-20s, the session's own cadence), not live every
 * frame -- there is no client-side face tracker here, stated plainly rather than implied.
 *
 * `box`/`landmarks` are in the captured frame's own pixel space (imageWidth x imageHeight,
 * i.e. the video element's native resolution). The SVG's viewBox is set to that same size with
 * preserveAspectRatio="xMidYMid slice", which reproduces CSS object-fit:cover's crop exactly --
 * the same reason the <video> itself uses object-cover -- so the net lines up with the face
 * without any manual scale math.
 */

const STATUS_COLOR = {
  match: '#5FA968', // verdictReal
  uncertain: '#D9A94E', // verdictCaution
  mismatch: '#D16565', // verdictDanger
  scanning: '#C89361', // brass -- no verdict yet (session just started, or no face this check)
}

const STATUS_LABEL = {
  match: 'IDENTITY MATCH',
  uncertain: 'IDENTITY UNCERTAIN',
  mismatch: 'IDENTITY MISMATCH',
  scanning: 'SCANNING',
}

function statusFor(identityVerdict) {
  if (identityVerdict === 'MATCH') return 'match'
  if (identityVerdict === 'UNCERTAIN') return 'uncertain'
  if (identityVerdict === 'NO_MATCH') return 'mismatch'
  return 'scanning'
}

// Connects the 5 MTCNN landmarks (left eye, right eye, nose, left mouth corner, right mouth
// corner) into a web-like mesh -- every point linked to its two nearest neighbours plus a
// couple of cross-face diagonals, enough to read as a "net" rather than a bare outline.
const MESH_EDGES = [
  [0, 1], // eye to eye
  [0, 2], [1, 2], // eyes to nose
  [2, 3], [2, 4], // nose to mouth corners
  [3, 4], // mouth corner to mouth corner
  [0, 3], [1, 4], // outer diagonals
]

export default function FaceOverlay({ box, landmarks, imageWidth, imageHeight, identityVerdict }) {
  const status = statusFor(identityVerdict)
  const color = STATUS_COLOR[status]

  const gridLines = useMemo(() => {
    if (!box) return []
    const [x1, y1, x2, y2] = box
    const w = x2 - x1
    const h = y2 - y1
    const cols = 4
    const rows = 4
    const lines = []
    for (let i = 1; i < cols; i++) {
      const x = x1 + (w * i) / cols
      lines.push({ key: `v${i}`, x1: x, y1, x2: x, y2 })
    }
    for (let i = 1; i < rows; i++) {
      const y = y1 + (h * i) / rows
      lines.push({ key: `h${i}`, x1, y1: y, x2, y2: y })
    }
    return lines
  }, [box])

  if (!imageWidth || !imageHeight) return null

  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none"
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      <AnimatePresence>
        {box && (
          <motion.g
            key={status}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.35 }}
          >
            {/* The net -- a scan grid filling the detected face box. */}
            {gridLines.map((l) => (
              <line key={l.key} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} stroke={color} strokeWidth={1} opacity={0.35} />
            ))}

            {/* Bounding box with forensic corner-bracket ticks (same visual language as
                ScanFrame.jsx elsewhere in the app). */}
            <rect x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]} fill="none" stroke={color} strokeWidth={1.5} opacity={0.5} />
            {(() => {
              const [x1, y1, x2, y2] = box
              const arm = Math.max(10, Math.min(x2 - x1, y2 - y1) * 0.12)
              const corners = [
                [[x1, y1 + arm], [x1, y1], [x1 + arm, y1]],
                [[x2 - arm, y1], [x2, y1], [x2, y1 + arm]],
                [[x1, y2 - arm], [x1, y2], [x1 + arm, y2]],
                [[x2 - arm, y2], [x2, y2], [x2, y2 - arm]],
              ]
              return corners.map((pts, i) => (
                <polyline
                  key={i}
                  points={pts.map((p) => p.join(',')).join(' ')}
                  fill="none"
                  stroke={color}
                  strokeWidth={2.5}
                />
              ))
            })()}

            {/* Landmark mesh -- the actual detected eye/nose/mouth points, webbed together. */}
            {landmarks && MESH_EDGES.map(([a, b], i) => (
              <line
                key={i}
                x1={landmarks[a][0]} y1={landmarks[a][1]}
                x2={landmarks[b][0]} y2={landmarks[b][1]}
                stroke={color}
                strokeWidth={1}
                opacity={0.8}
              />
            ))}
            {landmarks && landmarks.map(([x, y], i) => (
              <circle key={i} cx={x} cy={y} r={Math.max(2, (box[2] - box[0]) * 0.012)} fill={color} />
            ))}

            {/* Pulsing scan sweep -- reads as "live", honestly scoped: it animates every check
                cycle (new key => new mount), not continuously, matching how often this data is
                actually refreshed. */}
            <motion.rect
              x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]}
              fill="none" stroke={color} strokeWidth={1}
              initial={{ opacity: 0.9, scale: 1 }}
              animate={{ opacity: 0, scale: 1.08 }}
              transition={{ duration: 1.1, ease: 'easeOut' }}
              style={{ transformOrigin: `${(box[0] + box[2]) / 2}px ${(box[1] + box[3]) / 2}px` }}
            />

            {/* Status label -- counter-mirrored around its own anchor point so it reads
                normally once the parent wrapper's CSS mirror (shared with the <video>) flips
                the whole overlay for display; everything else here (box/lines/dots) is
                direction-agnostic so the single outer flip is enough for those. */}
            <g transform={`translate(${box[0] * 2}, 0) scale(-1, 1)`}>
              <text
                x={box[0]}
                y={Math.max(14, box[1] - 10)}
                fill={color}
                fontSize={Math.max(12, (box[2] - box[0]) * 0.055)}
                fontWeight={600}
                style={{ letterSpacing: '0.06em' }}
              >
                {STATUS_LABEL[status]}
              </text>
            </g>
          </motion.g>
        )}
      </AnimatePresence>
    </svg>
  )
}
