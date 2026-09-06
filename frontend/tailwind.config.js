export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // TruthLens "Forensic Evidence Bench" system — a near-black
        // instrument-panel ground with warm brass as the one signal accent.
        // Replaces the previous warm-ivory editorial palette. Every pairing
        // here is contrast-checked against ground/panel; see index.css's
        // :root comment for the numbers.
        ground: '#0B0C0D',
        panel: '#17181A',
        'panel-raised': '#1F2022',
        line: 'rgba(237,234,227,0.10)',
        'line-strong': 'rgba(237,234,227,0.20)',
        bone: '#EDEAE3',
        'bone-dim': '#9C9890',
        'bone-faint': '#6B6862',
        brass: '#C89361',
        'brass-deep': '#8F6238',
        verdictReal: '#5FA968',
        verdictCaution: '#D9A94E',
        verdictDanger: '#D16565',
      },
      fontFamily: {
        serif: ['"Playfair Display"', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scanPulse: {
          '0%': { transform: 'scale(0.9)', opacity: '0.5' },
          '70%': { transform: 'scale(1.15)', opacity: '0' },
          '100%': { transform: 'scale(1.15)', opacity: '0' },
        },
      },
      transitionTimingFunction: {
        smooth: 'cubic-bezier(0.22, 1, 0.36, 1)',
        'out-expo': 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
    },
  },
  plugins: [],
}
