import React from 'react'
import { Link } from 'react-router-dom'
import Logo from './Logo'
import { useAuth } from './context/AuthContext'

const PRODUCT_LINKS = {
  'Verify Media': '/verify',
  'History': '/history',
  'Reports': '/reports',
}

const Footer = () => {
  const { user, logout } = useAuth()

  return (
    <footer className="bg-panel text-bone-dim py-16 md:py-20 border-t border-line">
      <div className="max-w-[1400px] mx-auto px-6 md:px-12 lg:px-16">
        <div className="grid grid-cols-2 md:grid-cols-12 gap-12 mb-16">
          <div className="col-span-2 md:col-span-4">
            <div className="flex items-center gap-3 mb-6">
              <Logo size={24} color="#EDEAE3" />
              <span className="font-serif text-[16px] text-bone">TruthLens</span>
            </div>
            <p className="text-[14px] leading-[1.7] text-bone-dim max-w-[280px]">
              A forensic media examination instrument. We believe clarity is a right, not a privilege.
            </p>
          </div>

          <div className="md:col-span-2 md:col-start-6">
            <p className="text-[11px] tracking-[0.15em] uppercase text-bone mb-4">Product</p>
            <ul className="space-y-3">
              {Object.entries(PRODUCT_LINKS).map(([label, path]) => (
                <li key={label}>
                  <Link
                    to={path}
                    className="text-[14px] hover:text-bone transition-colors duration-300"
                  >
                    {label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          <div className="md:col-span-2">
            <p className="text-[11px] tracking-[0.15em] uppercase text-bone mb-4">Company</p>
            <ul className="space-y-3">
              <li>
                <Link to="/about" className="text-[14px] hover:text-bone transition-colors duration-300">
                  About
                </Link>
              </li>
            </ul>
          </div>

          <div className="md:col-span-2">
            <p className="text-[11px] tracking-[0.15em] uppercase text-bone mb-4">Account</p>
            <ul className="space-y-3">
              {user ? (
                <>
                  <li className="text-[14px] text-bone">{user.name}</li>
                  <li>
                    <button
                      onClick={() => { logout(); window.location.href = '/' }}
                      className="text-[14px] hover:text-bone transition-colors duration-300"
                    >
                      Log Out
                    </button>
                  </li>
                </>
              ) : (
                <>
                  <li>
                    <Link to="/login" className="text-[14px] hover:text-bone transition-colors duration-300">
                      Log in
                    </Link>
                  </li>
                  <li>
                    <Link to="/signup" className="text-[14px] hover:text-bone transition-colors duration-300">
                      Sign Up
                    </Link>
                  </li>
                </>
              )}
            </ul>
          </div>
        </div>

        <div className="pt-8 border-t border-line flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-[12px] text-bone-faint">
            &copy; {new Date().getFullYear()} TruthLens. All rights reserved.
          </p>
          <p className="text-[12px] text-bone-faint">Designed with precision.</p>
        </div>
      </div>
    </footer>
  )
}

export default Footer
