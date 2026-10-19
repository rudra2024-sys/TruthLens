import { ShieldCheck, ShieldAlert, ShieldEllipsis } from 'lucide-react'

/**
 * Single source of truth for verdict presentation. The backend only ever
 * returns 'REAL' | 'FAKE' | 'UNCERTAIN' (case-sensitive, uppercase) — this
 * normalizes defensively but never invents a fourth state.
 */
export function getVerdictInfo(rawVerdict) {
  const val = (rawVerdict || '').toString().toUpperCase()

  if (val === 'REAL' || val === 'AUTHENTIC') {
    return {
      key: 'real',
      label: 'Authentic',
      caption: 'No manipulation detected',
      Icon: ShieldCheck,
      text: '#5FA968',
      textClass: 'text-verdictReal',
      bg: 'bg-[rgba(95,169,104,0.1)]',
      border: 'border-[rgba(95,169,104,0.35)]',
      accent: '#5FA968',
      rgb: '95,169,104',
    }
  }

  if (val === 'UNCERTAIN' || val === 'SUSPICIOUS') {
    return {
      key: 'uncertain',
      label: 'Uncertain',
      caption: 'Inconclusive signal',
      Icon: ShieldEllipsis,
      text: '#D9A94E',
      textClass: 'text-verdictCaution',
      bg: 'bg-[rgba(217,169,78,0.1)]',
      border: 'border-[rgba(217,169,78,0.35)]',
      accent: '#D9A94E',
      rgb: '217,169,78',
    }
  }

  // FAKE, or anything unrecognized — the conservative default is to flag it,
  // never to silently present unknown data as trustworthy.
  return {
    key: 'manipulated',
    label: 'Manipulated',
    caption: 'Manipulation detected',
    Icon: ShieldAlert,
    text: '#D16565',
    textClass: 'text-verdictDanger',
    bg: 'bg-[rgba(209,101,101,0.1)]',
    border: 'border-[rgba(209,101,101,0.35)]',
    accent: '#D16565',
    rgb: '209,101,101',
  }
}

/**
 * The verdict actually shown on screen, accounting for a provenance conflict. The backend's
 * `conflict_note` (see provenance/service.py::_assess) is ONLY ever set when the detection models
 * said REAL/UNCERTAIN but the file itself carries strong direct AI evidence (an embedded AI
 * declaration or a matched generator watermark) — so its presence alone is the exact trigger,
 * nothing here re-derives that condition. Without this, the headline could read "No manipulation
 * detected" directly above a note saying the file carries an AI watermark, which is the
 * contradiction this fixes. The underlying stored verdict/confidence score are never touched —
 * this only changes what's rendered.
 */
export function getDisplayVerdictInfo(rawVerdict, provenance) {
  if (provenance?.conflict_note) {
    return {
      ...getVerdictInfo('FAKE'),
      caption: 'Manipulation detected — conflicts with provenance evidence below',
    }
  }
  return getVerdictInfo(rawVerdict)
}

/** The verdict string to pass to VerdictSeal/VerdictBadge so every piece of the UI agrees with
 * getDisplayVerdictInfo's override instead of recomputing from the raw (pre-override) verdict. */
export function getDisplayVerdict(rawVerdict, provenance) {
  return provenance?.conflict_note ? 'FAKE' : rawVerdict
}
