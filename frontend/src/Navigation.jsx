import React, { useEffect, useState } from 'react'
import {
  Link,
  useLocation,
  useNavigate,
} from 'react-router-dom'
import Logo from './Logo'
import {
  getCurrentUser,
  getStoredUser,
  logout,
} from './api/client'

const Navigation = () => {
  const [scrolled, setScrolled] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [user, setUser] = useState(getStoredUser())

  const location = useLocation()
  const navigate = useNavigate()

  useEffect(() => {
    const onScroll = () =>
      setScrolled(window.scrollY > 60)

    window.addEventListener('scroll', onScroll, {
      passive: true,
    })

    return () =>
      window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    setMobileOpen(false)
  }, [location.pathname])

  /*
   * Restore and validate session.
   */
  useEffect(() => {
    const token =
      localStorage.getItem('truthlens_token')

    if (!token) {
      setUser(null)
      return
    }

    getCurrentUser()
      .then((response) => {
        setUser(response.data)

        localStorage.setItem(
          'truthlens_user',
          JSON.stringify(response.data)
        )
      })
      .catch(() => {
        logout()
        setUser(null)
      })
  }, [location.pathname])

  const handleLogout = () => {
    logout()
    setUser(null)
    setMobileOpen(false)
    navigate('/')
  }

  const navLinks = [
    { label: 'Home', path: '/' },
    { label: 'Verify Media', path: '/verify' },
    { label: 'History', path: '/history' },
    { label: 'Reports', path: '/reports' },
    { label: 'About', path: '/about' },
  ]

  const isActive = (path) =>
    location.pathname === path

  return (
    <>
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
          scrolled || mobileOpen
            ? 'nav-blur border-b border-[rgba(138,133,128,0.1)]'
            : 'bg-transparent'
        }`}
      >
        <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
          <div className="flex items-center justify-between h-[72px]">

            <Link
              to="/"
              className="flex items-center gap-3 group"
            >
              <Logo
                size={28}
                className="text-[#1A1A1A] transition-transform duration-500 group-hover:scale-105"
              />

              <span className="font-serif text-[18px] tracking-[0.02em] text-[#1A1A1A] font-medium">
                TruthLens
              </span>
            </Link>

            <div className="hidden md:flex items-center gap-10">
              {navLinks.map((link) => (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`text-[13px] font-medium tracking-[0.08em] uppercase transition-colors duration-300 ${
                    isActive(link.path)
                      ? 'text-[#A67B5B]'
                      : 'text-[#8A8580] hover:text-[#1A1A1A]'
                  }`}
                >
                  {link.label}
                </Link>
              ))}
            </div>

            <div className="flex items-center gap-6">

              {user ? (
                <>
                  <span className="hidden sm:block text-[13px] font-medium text-[#8A8580]">
                    {user.name}
                  </span>

                  <button
                    onClick={handleLogout}
                    className="text-[13px] font-medium tracking-[0.08em] uppercase bg-[#1A1A1A] text-[#F7F5F0] px-6 py-2.5 rounded-[4px] btn-lift"
                  >
                    Log Out
                  </button>
                </>
              ) : (
                <>
                  <Link
                    to="/login"
                    className="hidden sm:block text-[13px] font-medium tracking-[0.06em] text-[#8A8580] hover:text-[#1A1A1A] transition-colors duration-300"
                  >
                    Log in
                  </Link>

                  <Link
                    to="/signup"
                    className="text-[13px] font-medium tracking-[0.08em] uppercase bg-[#A67B5B] text-[#F7F5F0] px-6 py-2.5 rounded-[4px] btn-lift"
                  >
                    Sign Up
                  </Link>
                </>
              )}

              <button
                onClick={() =>
                  setMobileOpen(!mobileOpen)
                }
                className="md:hidden flex flex-col gap-1.5 p-2"
                aria-label="Toggle menu"
              >
                <span
                  className={`w-5 h-px bg-[#1A1A1A] transition-all duration-300 ${
                    mobileOpen
                      ? 'rotate-45 translate-y-[3.5px]'
                      : ''
                  }`}
                />

                <span
                  className={`w-5 h-px bg-[#1A1A1A] transition-all duration-300 ${
                    mobileOpen
                      ? '-rotate-45 -translate-y-[3.5px]'
                      : ''
                  }`}
                />
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Mobile menu */}
      <div
        className={`fixed inset-0 z-40 bg-[#F7F5F0] transition-all duration-500 md:hidden ${
          mobileOpen
            ? 'opacity-100 visible'
            : 'opacity-0 invisible'
        }`}
      >
        <div className="flex flex-col items-center justify-center h-full gap-8">

          {navLinks.map((link, i) => (
            <Link
              key={link.path}
              to={link.path}
              className={`font-serif text-[32px] transition-all duration-500 ${
                isActive(link.path)
                  ? 'text-[#A67B5B]'
                  : 'text-[#1A1A1A]'
              }`}
              style={{
                transitionDelay: mobileOpen
                  ? `${i * 50}ms`
                  : '0ms',
              }}
            >
              {link.label}
            </Link>
          ))}

          {user ? (
            <button
              onClick={handleLogout}
              className="font-serif text-[28px] text-[#A67B5B]"
            >
              Log Out
            </button>
          ) : (
            <>
              <Link
                to="/login"
                className="font-serif text-[28px] text-[#1A1A1A]"
              >
                Log In
              </Link>

              <Link
                to="/signup"
                className="font-serif text-[28px] text-[#A67B5B]"
              >
                Sign Up
              </Link>
            </>
          )}
        </div>
      </div>
    </>
  )
}

export default Navigation