import React from 'react'

export const Logo = ({ className = '', size = 32, color = 'currentColor' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 48 48"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
  >
    <path
      d="M4 24C4 24 12 8 24 8C36 8 44 24 44 24C44 24 36 40 24 40C12 40 4 24 4 24Z"
      stroke={color}
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M24 16C19.58 16 16 19.58 16 24C16 28.42 19.58 32 24 32C26.5 32 28.7 30.9 30.2 29.2"
      stroke={color}
      strokeWidth="1.5"
      strokeLinecap="round"
    />
    <path
      d="M24 20C22.5 20 21.2 21.1 20.8 22.5"
      stroke={color}
      strokeWidth="1.5"
      strokeLinecap="round"
    />
    <circle cx="28" cy="18" r="1.5" fill={color} opacity="0.4" />
  </svg>
)

export default Logo
