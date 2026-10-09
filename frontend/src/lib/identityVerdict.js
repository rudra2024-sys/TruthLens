import { ShieldCheck, ShieldAlert, ShieldEllipsis } from 'lucide-react'

/**
 * Single source of truth for identity-match verdict presentation, analogous to
 * lib/verdict.js's getVerdictInfo but for the MATCH / NO_MATCH / UNCERTAIN vocabulary
 * returned by the identity-match API — kept separate rather than overloading the
 * deepfake-verdict helper, since the two concepts aren't the same thing.
 */
export function getIdentityMatchInfo(rawVerdict) {
  const val = (rawVerdict || '').toString().toUpperCase()

  if (val === 'MATCH') {
    return {
      key: 'match',
      label: 'Match',
      caption: 'Face matches the enrolled reference',
      Icon: ShieldCheck,
      text: '#5FA968',
      textClass: 'text-verdictReal',
      bg: 'bg-[rgba(95,169,104,0.1)]',
      border: 'border-[rgba(95,169,104,0.35)]',
      accent: '#5FA968',
    }
  }

  if (val === 'UNCERTAIN') {
    return {
      key: 'uncertain',
      label: 'Uncertain',
      caption: 'Similarity too close to call',
      Icon: ShieldEllipsis,
      text: '#D9A94E',
      textClass: 'text-verdictCaution',
      bg: 'bg-[rgba(217,169,78,0.1)]',
      border: 'border-[rgba(217,169,78,0.35)]',
      accent: '#D9A94E',
    }
  }

  // NO_MATCH, or anything unrecognized — the conservative default is to flag it.
  return {
    key: 'no_match',
    label: 'No Match',
    caption: 'Face does not match the enrolled reference',
    Icon: ShieldAlert,
    text: '#D16565',
    textClass: 'text-verdictDanger',
    bg: 'bg-[rgba(209,101,101,0.1)]',
    border: 'border-[rgba(209,101,101,0.35)]',
    accent: '#D16565',
  }
}
