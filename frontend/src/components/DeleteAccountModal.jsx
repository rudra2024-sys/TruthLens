import React, { useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { deleteAccount, logout } from '../api/client'

/**
 * Account deletion confirmation (added 2026-10-03) — closes the gap docs/ETHICS_AND_LIMITATIONS.md has
 * flagged since it was written ("no delete-my-data button yet"). Irreversible and says so: requires typing
 * DELETE before the button is even clickable, same spirit as GitHub's own repo-deletion confirmation.
 */
export default function DeleteAccountModal({ open, onClose }) {
  const [confirmText, setConfirmText] = useState('')
  const [status, setStatus] = useState('idle') // idle | deleting | error
  const [error, setError] = useState(null)

  if (!open) return null

  const canConfirm = confirmText.trim().toUpperCase() === 'DELETE' && status !== 'deleting'

  const handleDelete = async () => {
    setStatus('deleting')
    setError(null)
    try {
      await deleteAccount()
      logout()
      window.location.href = '/'
    } catch (err) {
      setStatus('error')
      setError(err.message || 'Could not delete your account. Please try again.')
    }
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="delete-account-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onClick={(e) => { if (e.target === e.currentTarget && status !== 'deleting') onClose() }}
    >
      <div className="w-full max-w-[440px] rounded-[6px] border border-line bg-ground p-6">
        <div className="flex gap-3 mb-4">
          <AlertTriangle size={20} strokeWidth={1.75} className="text-verdictDanger shrink-0 mt-[2px]" aria-hidden="true" />
          <div>
            <h2 id="delete-account-title" className="text-[16px] font-medium text-bone">Delete your account</h2>
            <p className="text-[13px] text-bone-dim mt-1">
              This permanently deletes your account and everything in it — every upload, scan result, PDF
              report and piece of feedback. This cannot be undone.
            </p>
          </div>
        </div>

        <label htmlFor="delete-confirm-input" className="tl-hud-label !text-[9px] block mb-2">
          Type DELETE to confirm
        </label>
        <input
          id="delete-confirm-input"
          type="text"
          value={confirmText}
          onChange={(e) => setConfirmText(e.target.value)}
          disabled={status === 'deleting'}
          className="w-full bg-ground border border-line rounded-[3px] px-3 py-2 text-[13px] text-bone mb-4
                     focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass"
          placeholder="DELETE"
          autoComplete="off"
        />

        {error && <p className="text-[13px] text-verdictDanger mb-4">{error}</p>}

        <div className="flex gap-3 justify-end">
          <button
            type="button"
            onClick={onClose}
            disabled={status === 'deleting'}
            className="px-4 py-2 text-[13px] text-bone-dim hover:text-bone transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleDelete}
            disabled={!canConfirm}
            className="px-4 py-2 text-[13px] rounded-[3px] bg-verdictDanger text-white disabled:opacity-40
                       disabled:cursor-not-allowed hover:bg-verdictDanger/90 transition-colors"
          >
            {status === 'deleting' ? 'Deleting…' : 'Permanently delete my account'}
          </button>
        </div>
      </div>
    </div>
  )
}
