import React from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from './context/AuthContext'
import StaggeredMenu from './components/StaggeredMenu'
import './components/StaggeredMenuTheme.css'

/**
 * Primary site navigation — the React Bits StaggeredMenu (JS-CSS variant),
 * integrated as-is. See StaggeredMenu.jsx for the two integration
 * adaptations (SPA routing on item click, TruthLens logo asset) and
 * StaggeredMenuTheme.css for the scoped dark-theme repaint. Animation
 * structure/timing/GSAP logic is untouched from the registry source.
 */
const GREETINGS = ['Hello', 'Welcome back', 'Hola']

/**
 * Deterministic per-user pick, not per-render random -- avoids the
 * greeting flipping on every route change/re-render, while still varying
 * across different accounts.
 */
function pickGreeting(seed) {
  let hash = 0
  for (let i = 0; i < seed.length; i++) {
    hash = (hash * 31 + seed.charCodeAt(i)) | 0
  }
  return GREETINGS[Math.abs(hash) % GREETINGS.length]
}

const Navigation = () => {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/')
  }

  const firstName = user?.name?.trim().split(/\s+/)[0]
  const greeting = firstName ? (
    <>
      {pickGreeting(user.user_id || firstName)}, <strong>{firstName}</strong>
    </>
  ) : null

  const items = [
    { label: 'Home', link: '/', ariaLabel: 'Go to home' },
    { label: 'Verify', link: '/verify', ariaLabel: 'Go to verify' },
    { label: 'History', link: '/history', ariaLabel: 'Go to history' },
    { label: 'Reports', link: '/reports', ariaLabel: 'Go to reports' },
    { label: 'About', link: '/about', ariaLabel: 'Go to about' },
    user
      ? { label: 'Log Out', link: '#', ariaLabel: 'Log out', onClick: handleLogout }
      : { label: 'Log In', link: '/login', ariaLabel: 'Log in' },
    ...(!user ? [{ label: 'Sign Up', link: '/signup', ariaLabel: 'Create an account' }] : []),
  ]

  return (
    <StaggeredMenu
      position="right"
      items={items}
      displaySocials={false}
      displayItemNumbering
      colors={['#1F2022', '#0B0C0D']}
      logoUrl="/truthlens-logo.svg"
      logoText="TruthLens"
      greeting={greeting}
      menuButtonColor="#EDEAE3"
      openMenuButtonColor="#EDEAE3"
      accentColor="#C89361"
      changeMenuColorOnOpen={false}
      isFixed
      closeOnClickAway
    />
  )
}

export default Navigation
