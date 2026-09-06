import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence, useAnimationControls } from 'framer-motion'
import useReducedMotion from './hooks/useReducedMotion'
import { User, Mail, Lock, AlertCircle } from 'lucide-react'
import { signup, saveAuth } from './api/client'
import AuthField from './components/AuthField'
import Logo from './Logo'

export default function SignUp() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const controls = useAnimationControls()
  const reduced = useReducedMotion()

  useEffect(() => {
    controls.start({ opacity: 1, y: 0, transition: { duration: reduced ? 0.2 : 0.5, ease: [0.22, 1, 0.36, 1] } })
  }, [controls, reduced])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const response = await signup(name, email, password)
      saveAuth(response.data)
      window.location.href = '/'
    } catch (err) {
      setError(err.message)
      if (!reduced) controls.start({ x: [0, -8, 8, -5, 5, 0], transition: { duration: 0.45 } })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen grid md:grid-cols-2">
      <div className="relative overflow-hidden bg-panel tl-inspection-grid flex flex-col justify-center px-6 md:px-16 pt-24 pb-16 md:py-0 border-b md:border-b-0 md:border-r border-line">
        <div className="absolute inset-0 tl-noise-dark" aria-hidden="true" />
        <div className="relative max-w-[420px]">
          <Logo size={30} className="text-brass mb-8" />
          <p className="tl-hud-label mb-4">New Registration</p>
          <h1 className="font-serif text-display-m text-bone mb-6">
            Open your own<br /><span className="italic text-brass">case record.</span>
          </h1>
          <p className="text-[14px] leading-[1.7] text-bone-dim max-w-[340px]">
            An account keeps every investigation you run — history, reports, and verdicts — tied to you alone.
          </p>
        </div>
      </div>

      <div className="relative flex items-center justify-center px-6 py-16 md:py-0 bg-ground">
        <motion.div
          initial={reduced ? { opacity: 0 } : { opacity: 0, y: 16 }}
          animate={controls}
          className="w-full max-w-[380px]"
        >
          <p className="tl-hud-label mb-3">Investigator Registration</p>
          <h2 className="font-serif text-[28px] text-bone mb-8">Create account</h2>

          <form onSubmit={handleSubmit} className="space-y-5">
            <AuthField
              id="signup-name" label="Name" icon={User}
              value={name} onChange={(e) => setName(e.target.value)}
              placeholder="Your name" autoComplete="name"
            />
            <AuthField
              id="signup-email" label="Email" type="email" icon={Mail}
              value={email} onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com" autoComplete="email"
            />
            <AuthField
              id="signup-password" label="Password" type="password" icon={Lock} minLength={8}
              value={password} onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 8 characters" autoComplete="new-password"
            />

            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                  className="flex items-start gap-2 text-[13px] text-verdictDanger bg-[rgba(209,101,101,0.08)] border border-[rgba(209,101,101,0.3)] rounded-[3px] p-3"
                  role="alert"
                >
                  <AlertCircle size={15} strokeWidth={1.75} className="shrink-0 mt-0.5" />
                  <span>{error}</span>
                </motion.div>
              )}
            </AnimatePresence>

            <button
              type="submit"
              disabled={loading}
              className="w-full px-6 py-3.5 bg-brass text-ground rounded-[3px] text-[13px] font-medium tracking-[0.08em] uppercase btn-lift disabled:opacity-50 disabled:pointer-events-none"
            >
              {loading ? 'Creating account…' : 'Create Account'}
            </button>
          </form>

          <p className="text-[13px] text-bone-dim text-center mt-8">
            Already have an account?{' '}
            <Link to="/login" className="text-brass hover:text-bone transition-colors duration-300 link-underline">
              Log in
            </Link>
          </p>
        </motion.div>
      </div>
    </div>
  )
}
