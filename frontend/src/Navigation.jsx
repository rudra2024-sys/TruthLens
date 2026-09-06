import React, { useEffect, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import useReducedMotion from './hooks/useReducedMotion'
import Logo from './Logo'
import { useAuth } from './context/AuthContext'

/**
 * Instrument rail — an index of the app's sections rather than a generic
 * pill nav, but still a single persistent horizontal bar with visible text
 * labels: distinctive without sacrificing the intuitiveness/accessibility
 * a traditional navbar gives you for free.
 */
const Navigation = () => {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const { user, logout } = useAuth()
  const reduced = useReducedMotion()

  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 60)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  const handleLogout = () => {
    logout()
    setMobileOpen(false)
    navigate('/')
  }

  const navLinks = [
    { label: 'Home', path: '/' },
    { label: 'Verify', path: '/verify' },
    { label: 'History', path: '/history' },
    { label: 'Reports', path: '/reports' },
    { label: 'About', path: '/about' },
  ]

  const isActive = (path) => location.pathname === path

  return (
    <>
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-colors duration-500 ${
          scrolled || mobileOpen ? 'nav-panel' : 'bg-transparent'
        }`}
      >
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <div className="flex items-center justify-between h-[72px]">
            <Link to="/" className="flex items-center gap-3 group shrink-0">
              <Logo size={24} className="text-brass transition-transform duration-500 group-hover:scale-105" />
              <span className="tl-figure text-[13px] tracking-[0.14em] uppercase text-bone">TruthLens</span>
            </Link>

            <div className="hidden md:flex items-center">
              {navLinks.map((link, i) => (
                <React.Fragment key={link.path}>
                  {i > 0 && <span className="w-px h-3 bg-line-strong mx-4" aria-hidden="true" />}
                  <Link
                    to={link.path}
                    className={`relative flex items-baseline gap-2 pb-1 text-[12px] font-medium tracking-[0.1em] uppercase transition-colors duration-300 ${
                      isActive(link.path) ? 'text-brass' : 'text-bone-dim hover:text-bone'
                    }`}
                  >
                    <span className="tl-figure text-[9px] opacity-60">{String(i + 1).padStart(2, '0')}</span>
                    {link.label}
                    {isActive(link.path) && (
                      <motion.span
                        layoutId="nav-active-indicator"
                        className="absolute left-0 right-0 -bottom-[1px] h-px bg-brass"
                        transition={reduced ? { duration: 0 } : { type: 'spring', stiffness: 380, damping: 32 }}
                      />
                    )}
                  </Link>
                </React.Fragment>
              ))}
            </div>

            <div className="flex items-center gap-5">
              {user ? (
                <>
                  <span className="hidden sm:flex items-baseline gap-1.5 tl-figure text-[11px] text-bone-dim">
                    <span className="text-bone-faint">OPERATOR</span> {user.name}
                  </span>
                  <button
                    onClick={handleLogout}
                    className="text-[11px] font-medium tracking-[0.1em] uppercase bg-panel-raised border border-line-strong text-bone px-5 py-2.5 rounded-[3px] btn-lift"
                  >
                    Log Out
                  </button>
                </>
              ) : (
                <>
                  <Link to="/login" className="hidden sm:block text-[12px] font-medium tracking-[0.06em] text-bone-dim hover:text-bone transition-colors duration-300">
                    Log in
                  </Link>
                  <Link to="/signup" className="text-[11px] font-medium tracking-[0.1em] uppercase bg-brass text-ground px-5 py-2.5 rounded-[3px] btn-lift">
                    Sign Up
                  </Link>
                </>
              )}

              <button
                onClick={() => setMobileOpen(!mobileOpen)}
                className="md:hidden flex flex-col gap-1.5 p-2"
                aria-label="Toggle menu"
                aria-expanded={mobileOpen}
              >
                <span className={`w-5 h-px bg-bone transition-transform duration-300 ${mobileOpen ? 'rotate-45 translate-y-[3.5px]' : ''}`} />
                <span className={`w-5 h-px bg-bone transition-transform duration-300 ${mobileOpen ? '-rotate-45 -translate-y-[3.5px]' : ''}`} />
              </button>
            </div>
          </div>
        </div>
      </nav>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="fixed inset-0 z-40 bg-ground md:hidden"
          >
            <div className="flex flex-col items-center justify-center h-full gap-8">
              {navLinks.map((link, i) => (
                <motion.div
                  key={link.path}
                  initial={reduced ? { opacity: 0 } : { opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: reduced ? 0 : i * 0.05, duration: 0.35 }}
                  className="flex items-baseline gap-3"
                >
                  <span className="tl-figure text-[12px] text-bone-faint">{String(i + 1).padStart(2, '0')}</span>
                  <Link
                    to={link.path}
                    className={`font-serif text-[32px] transition-colors duration-300 ${isActive(link.path) ? 'text-brass' : 'text-bone'}`}
                  >
                    {link.label}
                  </Link>
                </motion.div>
              ))}

              {user ? (
                <button onClick={handleLogout} className="font-serif text-[28px] text-brass mt-4">
                  Log Out
                </button>
              ) : (
                <div className="flex items-center gap-8 mt-4">
                  <Link to="/login" className="font-serif text-[26px] text-bone">Log In</Link>
                  <Link to="/signup" className="font-serif text-[26px] text-brass">Sign Up</Link>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

export default Navigation
