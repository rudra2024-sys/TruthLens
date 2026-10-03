import React, { Suspense, lazy } from 'react'
import { BrowserRouter, Routes, Route, useLocation, Navigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import useReducedMotion from './hooks/useReducedMotion'
import { AuthProvider, useAuth } from './context/AuthContext'
import Navigation from './Navigation'
import Footer from './Footer'
import DotField from './components/DotField'
import LoadingState from './components/LoadingState'

// Route-level code-splitting (added 2026-10-03): the production build was one 566KB+ JS bundle (Vite's own
// build warns about chunks over 500KB) because every page - including GSAP/recharts-heavy ones like Verify
// and Reports - loaded on first paint regardless of which route the visitor actually opened. Each page is
// its own chunk now, fetched only when its route is first visited; Suspense's fallback reuses the same
// LoadingState component already used for auth-checking, so a lazy page load looks the same as any other
// brief loading state, not a new one.
const Home = lazy(() => import('./Home'))
const Verify = lazy(() => import('./Verify'))
const History = lazy(() => import('./History'))
const Reports = lazy(() => import('./Reports'))
const About = lazy(() => import('./About'))
const Login = lazy(() => import('./Login'))
const SignUp = lazy(() => import('./SignUp'))
const ReportDetail = lazy(() => import('./ReportDetail'))
const NotFound = lazy(() => import('./NotFound'))

const ProtectedRoute = ({ children }) => {
  const { user, checking } = useAuth()

  if (checking) {
    return <LoadingState label="Verifying access" layout="screen" size="sm" />
  }

  if (!user) {
    return <Navigate to="/login" replace />
  }

  return children
}

const GuestRoute = ({ children }) => {
  const { user, checking } = useAuth()
  if (checking) return null
  if (user) {
    return <Navigate to="/" replace />
  }
  return children
}

/**
 * Global, site-wide ambient background — mounted once here (not per-page)
 * so it sits behind every route. Fixed + a modest z-index keeps it above
 * each page's .tl-inspection-grid (unelevated, z-auto) while staying below
 * real content, which is consistently wrapped in `relative z-10` across
 * pages. pointer-events:none so it never intercepts clicks; DotField
 * tracks the cursor via a window-level listener, so that doesn't affect
 * the bulge/glow interaction. Skipped under prefers-reduced-motion, same
 * convention as the rest of this app's motion.
 */
const GlobalDotField = () => {
  const reduced = useReducedMotion()
  if (reduced) return null
  return (
    <div aria-hidden="true" style={{ position: 'fixed', inset: 0, zIndex: 5, pointerEvents: 'none' }}>
      <DotField
        dotRadius={1.5}
        dotSpacing={14}
        bulgeStrength={67}
        glowRadius={160}
        sparkle={false}
        waveAmplitude={0}
        gradientFrom="rgba(200, 147, 97, 0.6)"
        gradientTo="rgba(200, 147, 97, 0.26)"
        glowColor="rgba(200, 147, 97, 0.45)"
      />
    </div>
  )
}

const Layout = ({ children }) => (
  <>
    <GlobalDotField />
    <Navigation />
    {children}
    <Footer />
  </>
)

/**
 * Tasteful, brief page-transition — a fade with a small rise on arrival.
 *
 * Note: an earlier version of this wrapped <Routes> in <AnimatePresence
 * mode="wait"> to also animate the outgoing page's exit. That combination
 * turned out to deadlock navigation with this React Router version — the
 * URL would change but the new page would never mount (AnimatePresence
 * waiting on an exit that never resolved). Rather than fight a fragile
 * pattern for a marginal polish gain, this keeps the reliable half (a
 * mount-triggered entrance, keyed by path so it re-plays on every
 * navigation) and drops the exit choreography entirely.
 */
const PageTransition = ({ children, pathKey }) => {
  const reduced = useReducedMotion()
  return (
    <motion.div
      key={pathKey}
      initial={reduced ? { opacity: 0 } : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: reduced ? 0.15 : 0.4, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  )
}

/**
 * React Router doesn't reset scroll position on navigation by default --
 * without this, following a link/redirect while scrolled down a page
 * leaves the next page scrolled down too, landing the visitor mid-content
 * instead of at its top.
 */
const ScrollToTop = () => {
  const { pathname } = useLocation()
  React.useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])
  return null
}

function AnimatedRoutes() {
  const location = useLocation()
  const p = location.pathname
  return (
    <>
      <ScrollToTop />
      <Suspense fallback={<LoadingState label="Loading" layout="screen" size="sm" />}>
      <Routes>
      <Route path="/" element={<Layout><PageTransition pathKey={p}><Home /></PageTransition></Layout>} />
      <Route
        path="/verify"
        element={
          <ProtectedRoute>
            <Layout><PageTransition pathKey={p}><Verify /></PageTransition></Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/history"
        element={
          <ProtectedRoute>
            <Layout><PageTransition pathKey={p}><History /></PageTransition></Layout>
          </ProtectedRoute>
        }
      />
      <Route
        path="/reports"
        element={
          <ProtectedRoute>
            <Layout><PageTransition pathKey={p}><Reports /></PageTransition></Layout>
          </ProtectedRoute>
        }
      />
      <Route path="/about" element={<Layout><PageTransition pathKey={p}><About /></PageTransition></Layout>} />
      <Route
        path="/login"
        element={
          <GuestRoute>
            <Layout><PageTransition pathKey={p}><Login /></PageTransition></Layout>
          </GuestRoute>
        }
      />
      <Route
        path="/signup"
        element={
          <GuestRoute>
            <Layout><PageTransition pathKey={p}><SignUp /></PageTransition></Layout>
          </GuestRoute>
        }
      />
      <Route
        path="/report/:id"
        element={
          <ProtectedRoute>
            <Layout><PageTransition pathKey={p}><ReportDetail /></PageTransition></Layout>
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Layout><PageTransition pathKey={p}><NotFound /></PageTransition></Layout>} />
      </Routes>
      </Suspense>
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AnimatedRoutes />
      </AuthProvider>
    </BrowserRouter>
  )
}
