import { useEffect, useState } from 'react'
import {
  getDetectionResult,
  reportUrl,
  signup,
  login,
  saveAuth,
  getCurrentUser,
} from './api/client'
import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navigation from './Navigation'
import Footer from './Footer'
import Home from './Home'
import Verify from './Verify'
import History from './History'
import Reports from './Reports'
import About from './About'

/* Simple placeholder pages for auth routes */
const Login = () => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const response = await login(email, password)

    saveAuth(response.data)
    window.location.href = '/'

    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#F7F5F0] pt-[72px] flex items-center justify-center px-6">
      <div className="w-full max-w-[420px] bg-white border border-[#E5E1DA] rounded-[8px] p-8">
        <h1 className="font-serif text-[32px] text-[#1A1A1A] mb-2">
          Log In
        </h1>

        <p className="text-[15px] text-[#8A8580] mb-8">
          Welcome back to TruthLens.
        </p>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm text-[#1A1A1A] mb-2">
              Email
            </label>

            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full border border-[#E5E1DA] rounded-[4px] px-4 py-3 outline-none focus:border-[#1A1A1A]"
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label className="block text-sm text-[#1A1A1A] mb-2">
              Password
            </label>

            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full border border-[#E5E1DA] rounded-[4px] px-4 py-3 outline-none focus:border-[#1A1A1A]"
              placeholder="••••••••"
            />
          </div>

          {error && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-[4px] p-3">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full px-6 py-3 bg-[#1A1A1A] text-white rounded-[4px] text-[14px] disabled:opacity-50"
          >
            {loading ? 'Logging in...' : 'Log In'}
          </button>
        </form>
      </div>
    </div>
  )
}

const SignUp = () => {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const response = await signup(
        name,
        email,
        password
      )

    saveAuth(response.data)
    window.location.href = '/'

    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#F7F5F0] pt-[72px] flex items-center justify-center px-6">
      <div className="w-full max-w-[420px] bg-white border border-[#E5E1DA] rounded-[8px] p-8">
        <h1 className="font-serif text-[32px] text-[#1A1A1A] mb-2">
          Sign Up
        </h1>

        <p className="text-[15px] text-[#8A8580] mb-8">
          Create your TruthLens account.
        </p>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm text-[#1A1A1A] mb-2">
              Name
            </label>

            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="w-full border border-[#E5E1DA] rounded-[4px] px-4 py-3 outline-none focus:border-[#1A1A1A]"
              placeholder="Your name"
            />
          </div>

          <div>
            <label className="block text-sm text-[#1A1A1A] mb-2">
              Email
            </label>

            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full border border-[#E5E1DA] rounded-[4px] px-4 py-3 outline-none focus:border-[#1A1A1A]"
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label className="block text-sm text-[#1A1A1A] mb-2">
              Password
            </label>

            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              className="w-full border border-[#E5E1DA] rounded-[4px] px-4 py-3 outline-none focus:border-[#1A1A1A]"
              placeholder="At least 8 characters"
            />
          </div>

          {error && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-[4px] p-3">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full px-6 py-3 bg-[#1A1A1A] text-white rounded-[4px] text-[14px] disabled:opacity-50"
          >
            {loading ? 'Creating account...' : 'Create Account'}
          </button>
        </form>
      </div>
    </div>
  )
}

const ReportDetail = () => {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const uploadId = window.location.pathname.split('/').pop()

  useEffect(() => {
    const loadResult = async () => {
      try {
        const response = await getDetectionResult(uploadId)
        setResult(response.data)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }

    loadResult()
  }, [uploadId])

  if (loading) {
    return (
      <div className="min-h-screen bg-[#F7F5F0] pt-[72px] flex items-center justify-center">
        <p className="text-[#8A8580]">Loading report...</p>
      </div>
    )
  }

  if (error || !result) {
    return (
      <div className="min-h-screen bg-[#F7F5F0] pt-[72px] flex items-center justify-center">
        <div className="text-center">
          <h1 className="font-serif text-[32px] text-[#1A1A1A] mb-4">
            Report unavailable
          </h1>
          <p className="text-[15px] text-[#8A8580]">
            {error || 'No detection result found.'}
          </p>
        </div>
      </div>
    )
  }

 const confidence =
  (result.confidence_score ?? 0) * 100

  const fakeProbability =
    (result.image_analysis?.fake_probability ?? 0) * 100

  const realProbability =
    (result.image_analysis?.real_probability ?? 0) * 100

  return (
    <div className="min-h-screen bg-[#F7F5F0] pt-[72px]">
      <div className="max-w-[900px] mx-auto px-6 py-16">

        <div className="mb-10">
          <p className="text-[12px] tracking-[0.2em] uppercase text-[#8A8580] mb-3">
            TruthLens Analysis Report
          </p>

          <h1 className="font-serif text-[42px] text-[#1A1A1A]">
            Detection Result
          </h1>
        </div>

        <div className="bg-white border border-[#E5E1DA] rounded-[8px] p-8">

          <div className="flex items-center justify-between mb-8">
            <div>
              <p className="text-[12px] uppercase tracking-[0.15em] text-[#8A8580]">
                Verdict
              </p>

              <h2 className="font-serif text-[36px] text-[#1A1A1A] mt-2">
                {result.verdict}
              </h2>
            </div>

            <div className="text-right">
              <p className="text-[12px] uppercase tracking-[0.15em] text-[#8A8580]">
                Confidence
              </p>

              <p className="font-mono text-[32px] text-[#1A1A1A] mt-2">
                {confidence.toFixed(1)}%
              </p>
            </div>
          </div>

          <div className="border-t border-[#E5E1DA] pt-6 space-y-5">

            <div className="flex justify-between">
              <span className="text-[#8A8580]">
                Model
              </span>

              <span className="font-medium text-[#1A1A1A]">
                {result.model_used}
              </span>
            </div>

            {result.image_analysis && (
              <>
                <div className="flex justify-between">
                  <span className="text-[#8A8580]">
                    FAKE probability
                  </span>

                  <span className="font-mono text-[#1A1A1A]">
                    {fakeProbability.toFixed(1)}%
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-[#8A8580]">
                    REAL probability
                  </span>

                  <span className="font-mono text-[#1A1A1A]">
                    {realProbability.toFixed(1)}%
                  </span>
                </div>
              </>
            )}

            <div className="flex justify-between">
              <span className="text-[#8A8580]">
                Processing time
              </span>

              <span className="font-mono text-[#1A1A1A]">
                {Math.round(result.processing_time_ms)} ms
              </span>
            </div>

          </div>

          <div className="mt-8 pt-6 border-t border-[#E5E1DA]">

            <a
              href={reportUrl(result.upload_id)}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block px-6 py-3 bg-[#1A1A1A] text-white rounded-[4px] text-[14px]"
            >
              Download PDF Report
            </a>

          </div>

        </div>
      </div>
    </div>
  )
}

const ProtectedRoute = ({ children }) => {
  const [checking, setChecking] = useState(true)
  const [authenticated, setAuthenticated] = useState(false)

  useEffect(() => {
    const token = localStorage.getItem('truthlens_token')

    if (!token) {
      setChecking(false)
      return
    }

    getCurrentUser()
      .then((response) => {
        setAuthenticated(true)

        localStorage.setItem(
          'truthlens_user',
          JSON.stringify(response.data)
        )
      })
      .catch(() => {
        localStorage.removeItem('truthlens_token')
        localStorage.removeItem('truthlens_user')
        setAuthenticated(false)
      })
      .finally(() => {
        setChecking(false)
      })
  }, [])

  if (checking) {
    return (
      <div className="min-h-screen bg-[#F7F5F0] flex items-center justify-center">
        <p className="text-[#8A8580]">
          Checking authentication...
        </p>
      </div>
    )
  }

  if (!authenticated) {
    window.location.replace('/login')
    return null
  }

  return children
}

const GuestRoute = ({ children }) => {
  const token = localStorage.getItem('truthlens_token')

  if (token) {
    window.location.replace('/')
    return null
  }

  return children
}

const Layout = ({ children }) => (
  <>
    <Navigation />
    {children}
    <Footer />
  </>
)

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout><Home /></Layout>} />
        <Route
          path="/verify"
          element={
            <ProtectedRoute>
              <Layout><Verify /></Layout>
            </ProtectedRoute>
          }
        />

        <Route
          path="/history"
          element={
            <ProtectedRoute>
              <Layout><History /></Layout>
            </ProtectedRoute>
          }
        />

        <Route
          path="/reports"
          element={
            <ProtectedRoute>
              <Layout><Reports /></Layout>
            </ProtectedRoute>
          }
        />
        <Route path="/about" element={<Layout><About /></Layout>} />
        <Route
          path="/login"
          element={
            <GuestRoute>
              <Layout><Login /></Layout>
            </GuestRoute>
          }
        />

        <Route
          path="/signup"
          element={
            <GuestRoute>
              <Layout><SignUp /></Layout>
            </GuestRoute>
          }
        />
        <Route path="/report/:id" element={<Layout><ReportDetail /></Layout>} />
      </Routes>
    </BrowserRouter>
  )
}
