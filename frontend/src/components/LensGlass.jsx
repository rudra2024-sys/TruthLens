import React from 'react'

/**
 * The physical loupe — rebuilt as one precisely-measured SVG for the rim,
 * handle, and collar (previous version approximated these with percentage-
 * based divs, which is why the handle read as thin/disconnected at large
 * scale: percentages of a non-square box distort proportion as size
 * changes, and hand-tuned corner offsets drift). SVG coordinates are exact
 * at any render size, so the object holds its proportions from a 60px
 * inline icon up to the full-viewport verdict-phase scale.
 *
 * Geometry (in a 100-unit frame matching the glass circle, rim radius 47,
 * center 50,50): the handle attaches at 40° below horizontal — a natural
 * "held at an angle" grip, not a straight-down drip — as a tapered
 * quadrilateral (wider where it meets the rim, narrower at the grip end),
 * capped with a rounded tip and a rivet collar exactly at the rim
 * boundary. The SVG's own viewBox is oversized (-34 -14 168 156) so the
 * handle's extent past the glass circle is part of THIS component's own
 * declared box, not an overflow relying on an ancestor to allow it.
 *
 * The glass content (`children`) renders in a separate absolutely
 * positioned div so arbitrary React content shows through the rim's open
 * center — the rim itself is a stroked circle (fill: none), which leaves
 * its middle transparent with no masking needed.
 */
export default function LensGlass({ size = 160, className = '', children }) {
  const cx = 50, cy = 50, r = 47
  const angle = (40 * Math.PI) / 180
  const dir = [Math.cos(angle), Math.sin(angle)]
  const perp = [-dir[1], dir[0]]
  const attach = [cx + r * dir[0], cy + r * dir[1]]
  const length = 60
  const tip = [attach[0] + length * dir[0], attach[1] + length * dir[1]]
  const wNear = 4.4, wFar = 3.1
  const p = (pt, v, w) => [pt[0] + v[0] * w, pt[1] + v[1] * w]
  const a1 = p(attach, perp, wNear)
  const a2 = p(attach, perp, -wNear)
  const b1 = p(tip, perp, wFar)
  const b2 = p(tip, perp, -wFar)
  const handlePath = `M ${a1[0].toFixed(2)},${a1[1].toFixed(2)} L ${a2[0].toFixed(2)},${a2[1].toFixed(2)} L ${b2[0].toFixed(2)},${b2[1].toFixed(2)} L ${b1[0].toFixed(2)},${b1[1].toFixed(2)} Z`

  return (
    <div className={`relative ${className}`} style={{ width: size, height: size, overflow: 'visible' }}>
      {/* soft cast shadow onto the surface beneath — faint, as if resting
          just above it under one work light, not a heavy studio shadow */}
      <div className="absolute inset-[2%] rounded-full" style={{ boxShadow: '0 18px 40px rgba(0,0,0,0.55)' }} />

      {/* glass content — genuinely transparent: nothing but `children` sits
          behind it, so whatever the lens hovers over shows through for
          real. Seam + highlight draw on top, same as before. */}
      <div className="absolute rounded-full overflow-hidden" style={{ inset: '7.5%' }}>
        {children}
        <div
          className="absolute inset-0 rounded-full"
          style={{ boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.5), inset 0 0 0 2.5px rgba(255,255,255,0.12)' }}
        />
        <div
          className="absolute rounded-full lens-highlight"
          style={{
            width: '28%', height: '14%', top: '10%', left: '18%',
            background: 'linear-gradient(135deg, rgba(255,255,255,0.35), rgba(255,255,255,0))',
            transform: 'rotate(-16deg)', filter: 'blur(0.5px)',
          }}
        />
      </div>

      {/* rim + collar + handle — one SVG, exact geometry at any scale */}
      <svg
        className="absolute pointer-events-none"
        style={{ left: '-34%', top: '-14%', width: '168%', height: '156%', overflow: 'visible' }}
        viewBox="-34 -14 168 156"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="tl-rim-metal" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#F4E6D2" />
            <stop offset="30%" stopColor="#C89361" />
            <stop offset="55%" stopColor="#8F6238" />
            <stop offset="80%" stopColor="#DDB37F" />
            <stop offset="100%" stopColor="#F4E6D2" />
          </linearGradient>
          <linearGradient id="tl-handle-metal" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#5C4430" />
            <stop offset="45%" stopColor="#8F6238" />
            <stop offset="100%" stopColor="#3A2A1C" />
          </linearGradient>
          <radialGradient id="tl-rivet-metal" cx="35%" cy="30%" r="70%">
            <stop offset="0%" stopColor="#F4E6D2" />
            <stop offset="60%" stopColor="#C89361" />
            <stop offset="100%" stopColor="#5C4430" />
          </radialGradient>
        </defs>

        {/* handle */}
        <path d={handlePath} fill="url(#tl-handle-metal)" />
        <circle cx={tip[0]} cy={tip[1]} r={wFar} fill="url(#tl-handle-metal)" />

        {/* collar / rivet — exactly at the rim boundary where the handle meets it */}
        <circle cx={attach[0]} cy={attach[1]} r="5.6" fill="url(#tl-rivet-metal)" stroke="rgba(0,0,0,0.35)" strokeWidth="0.5" />

        {/* rim — a stroked circle, so the middle stays genuinely open for
            the glass content div beneath to show through */}
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="url(#tl-rim-metal)" strokeWidth="5.5" />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(0,0,0,0.4)" strokeWidth="0.6" />
      </svg>
    </div>
  )
}
