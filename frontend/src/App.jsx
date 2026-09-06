import React from 'react'
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import useReducedMotion from './hooks/useReducedMotion'
import { AuthProvider, useAuth } from './context/AuthContext'
import Navigation from './Navigation'
import Footer from './Footer'
import Home from './Home'
import Verify from './Verify'
import History from './History'
import Reports from './Reports'
import About from './About'
import Login from './Login'
import SignUp from './SignUp'
import ReportDetail from './ReportDetail'
import NotFound from './NotFound'
import LoadingState from './components/LoadingState'

const ProtectedRoute = ({ children }) => {
  const { user, checking } = useAuth()

  if (checking) {
    return <LoadingState label="Verifying access" layout="screen" size="sm" />
  }

  if (!user) {
    window.location.replace('/login')
    return null
  }

  return children
}

const GuestRoute = ({ children }) => {
  const { user, checking } = useAuth()
  if (checking) return null
  if (user) {
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

function AnimatedRoutes() {
  const location = useLocation()
  const p = location.pathname
  return (
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
      <Route path="/report/:id" element={<Layout><PageTransition pathKey={p}><ReportDetail /></PageTransition></Layout>} />
      <Route path="*" element={<Layout><PageTransition pathKey={p}><NotFound /></PageTransition></Layout>} />
    </Routes>
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
