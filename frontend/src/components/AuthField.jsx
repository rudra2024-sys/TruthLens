import React, { useState } from 'react'
import { Eye, EyeOff } from 'lucide-react'

export default function AuthField({ label, type = 'text', value, onChange, placeholder, icon: Icon, minLength, autoComplete, id }) {
  const [show, setShow] = useState(false)
  const isPassword = type === 'password'
  const inputType = isPassword && show ? 'text' : type

  return (
    <div>
      <label htmlFor={id} className="block text-[11px] tracking-[0.15em] uppercase text-bone-faint mb-2 tl-figure">
        {label}
      </label>
      <div className="relative">
        {Icon && (
          <Icon size={16} strokeWidth={1.5} className="absolute left-4 top-1/2 -translate-y-1/2 text-bone-faint pointer-events-none" />
        )}
        <input
          id={id}
          type={inputType}
          value={value}
          onChange={onChange}
          required
          minLength={minLength}
          autoComplete={autoComplete}
          placeholder={placeholder}
          className={`w-full border border-line-strong rounded-[3px] ${Icon ? 'pl-11' : 'pl-4'} ${isPassword ? 'pr-11' : 'pr-4'} py-3 text-[14px] text-bone bg-ground transition-colors duration-300 focus:border-brass focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass placeholder:text-bone-faint`}
        />
        {isPassword && (
          <button
            type="button"
            onClick={() => setShow((s) => !s)}
            aria-label={show ? 'Hide password' : 'Show password'}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-bone-faint hover:text-bone transition-colors duration-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brass rounded-[2px]"
          >
            {show ? <EyeOff size={16} strokeWidth={1.5} /> : <Eye size={16} strokeWidth={1.5} />}
          </button>
        )}
      </div>
    </div>
  )
}
