import React, { useEffect, useState } from 'react'
import { getFeedback, saveFeedback, withdrawFeedback } from '../api/client'

const TRUTH_OPTIONS = [
  { value: 'real', label: 'It is actually real' },
  { value: 'ai', label: 'It is actually AI-generated' },
  { value: 'unsure', label: 'Not sure' },
]

const btn =
  'text-[11px] font-medium tracking-[0.08em] uppercase px-4 py-2.5 rounded-[3px] border transition-colors duration-300 ' +
  'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass'

/**
 * "Was this result correct?" — lets a user tell TruthLens when a verdict was right or wrong. The feedback is stored per
 * result, can be changed or withdrawn at any time, and the uploaded file is only ever kept for testing/improving the
 * detectors if the user explicitly ticks the (default-off) consent box.
 */
export default function FeedbackPanel({ uploadId, className = '' }) {
  const [saved, setSaved] = useState(undefined)      // undefined = loading, null = none given, object = stored feedback
  const [editing, setEditing] = useState(false)      // the "it was wrong" details form
  const [truth, setTruth] = useState('unsure')
  const [comment, setComment] = useState('')
  const [allowReuse, setAllowReuse] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setSaved(undefined)
    getFeedback(uploadId)
      .then((fb) => { if (!cancelled) setSaved(fb) })
      .catch(() => { if (!cancelled) setSaved(null) })        // supplementary: never block the result screen
    return () => { cancelled = true }
  }, [uploadId])

  const submit = async (body) => {
    setBusy(true); setError(null)
    try {
      setSaved(await saveFeedback(uploadId, body))
      setEditing(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const withdraw = async () => {
    setBusy(true); setError(null)
    try {
      await withdrawFeedback(uploadId)
      setSaved(null); setEditing(false); setComment(''); setTruth('unsure'); setAllowReuse(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  if (saved === undefined) return null

  const heading = <p className="tl-hud-label !text-bone">Was this result correct?</p>

  // ---- already answered
  if (saved && !editing) {
    return (
      <div className={className} role="group" aria-label="Your feedback">
        {heading}
        <p className="text-[13px] text-bone-dim mt-2" aria-live="polite">
          {saved.agrees
            ? 'Thanks — you marked this result as correct.'
            : `Thanks — you marked this result as wrong${saved.true_label === 'real' ? ' (it is real)' : saved.true_label === 'ai' ? ' (it is AI-generated)' : ''}.`}
        </p>
        <label className="flex items-start gap-2 mt-3 text-[12px] text-bone-dim cursor-pointer max-w-[560px]">
          <input
            type="checkbox"
            className="mt-[3px] accent-[#C89361]"
            checked={!!saved.allow_reuse}
            disabled={busy}
            onChange={(e) => submit({ agrees: saved.agrees, true_label: saved.true_label, comment: saved.comment, allow_reuse: e.target.checked })}
          />
          <span>Allow TruthLens to keep this file to test and improve its detectors. Off by default; untick to stop.</span>
        </label>
        <div className="flex items-center gap-3 mt-3">
          <button type="button" onClick={() => { setTruth(saved.true_label || 'unsure'); setComment(saved.comment || ''); setAllowReuse(!!saved.allow_reuse); setEditing(true) }}
                  className={`${btn} border-line-strong text-bone-dim hover:text-bone`}>Change</button>
          <button type="button" onClick={withdraw} disabled={busy}
                  className="text-[11px] tracking-[0.06em] text-bone-faint hover:text-bone link-underline">Withdraw feedback</button>
        </div>
        {error && <p className="text-[12px] text-verdictDanger mt-2">{error}</p>}
      </div>
    )
  }

  // ---- "it was wrong" details
  if (editing) {
    return (
      <div className={className} role="group" aria-label="Tell us what the file really is">
        {heading}
        <fieldset className="mt-3">
          <legend className="text-[13px] text-bone-dim mb-2">What is this file, really?</legend>
          <div className="flex flex-wrap gap-2">
            {TRUTH_OPTIONS.map((o) => (
              <button key={o.value} type="button" aria-pressed={truth === o.value} onClick={() => setTruth(o.value)}
                      className={`${btn} ${truth === o.value ? 'border-brass text-brass' : 'border-line-strong text-bone-dim hover:text-bone'}`}>
                {o.label}
              </button>
            ))}
          </div>
        </fieldset>
        <label className="block mt-4 max-w-[560px]">
          <span className="text-[12px] text-bone-dim">Anything we should know? (optional)</span>
          <textarea value={comment} onChange={(e) => setComment(e.target.value)} maxLength={500} rows={2}
                    className="mt-1 w-full bg-ground border border-line rounded-[3px] px-3 py-2 text-[13px] text-bone focus-visible:outline focus-visible:outline-2 focus-visible:outline-brass" />
          <span className="tl-figure text-[10px] text-bone-faint">{comment.length}/500</span>
        </label>
        <label className="flex items-start gap-2 mt-3 text-[12px] text-bone-dim cursor-pointer max-w-[560px]">
          <input type="checkbox" className="mt-[3px] accent-[#C89361]" checked={allowReuse} onChange={(e) => setAllowReuse(e.target.checked)} />
          <span>Allow TruthLens to keep this file (and this note) to test and improve its detectors. Optional; you can withdraw at any time.</span>
        </label>
        <div className="flex items-center gap-3 mt-4">
          <button type="button" disabled={busy} onClick={() => submit({ agrees: false, true_label: truth, comment, allow_reuse: allowReuse })}
                  className={`${btn} bg-brass border-brass text-ground`}>Send feedback</button>
          <button type="button" onClick={() => setEditing(false)} className={`${btn} border-transparent text-bone-dim hover:text-bone`}>Cancel</button>
        </div>
        {error && <p className="text-[12px] text-verdictDanger mt-2">{error}</p>}
      </div>
    )
  }

  // ---- not answered yet
  return (
    <div className={className} role="group" aria-label="Was this result correct?">
      {heading}
      <div className="flex items-center gap-3 mt-3">
        <button type="button" disabled={busy} onClick={() => submit({ agrees: true, allow_reuse: false })}
                className={`${btn} border-line-strong text-bone hover:border-brass`}>Yes, correct</button>
        <button type="button" disabled={busy} onClick={() => setEditing(true)}
                className={`${btn} border-line-strong text-bone hover:border-brass`}>No, it is wrong</button>
      </div>
      {error && <p className="text-[12px] text-verdictDanger mt-2">{error}</p>}
    </div>
  )
}
