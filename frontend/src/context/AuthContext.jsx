import React, { createContext, useContext, useEffect, useState, useCallback } from 'react'
import { getCurrentUser, getStoredUser, logout as apiLogout } from '../api/client'

/**
 * Single shared auth check for the whole app. Before this existed,
 * ProtectedRoute (re-mounted on every protected-route navigation) and
 * Navigation (re-checking on every location change) each independently
 * called GET /auth/me, which both wasted requests and made the nav bar
 * flash away and back on every protected navigation. Now the check runs
 * once, and every consumer just reads the shared result.
 */
const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(getStoredUser())
  const [checking, setChecking] = useState(true)

  // Sets checking=true synchronously before the async /auth/me call (added 2026-10-12). Without
  // this, a post-login/signup navigate() to a role-gated route races GuestRoute: navigate()
  // updates the URL, but React hasn't swapped <Routes> away from the old /login or /signup match
  // yet, so the still-mounted GuestRoute can re-render with the freshly-set `user` and fire its
  // own "already logged in" redirect to "/" before the target route ever mounts. checking=true
  // makes every route guard (GuestRoute's `if (checking) return null`, AdminRoute/ProtectedRoute's
  // loading state) wait out that transition instead of deciding on stale-vs-fresh state.
  const refresh = useCallback(() => {
    const token = localStorage.getItem('truthlens_token')
    if (!token) {
      setUser(null)
      setChecking(false)
      return Promise.resolve()
    }
    setChecking(true)
    return getCurrentUser()
      .then((res) => {
        setUser(res.data)
        localStorage.setItem('truthlens_user', JSON.stringify(res.data))
      })
      .catch(() => {
        apiLogout()
        setUser(null)
      })
      .finally(() => setChecking(false))
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const logout = useCallback(() => {
    apiLogout()
    setUser(null)
  }, [])

  return (
    <AuthContext.Provider value={{ user, checking, refresh, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider')
  return ctx
}
