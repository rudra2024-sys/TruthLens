export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        canvas: '#07090E',
        elev:    '#10141C',
        elev2:   '#161B26',
        stroke:  '#262D3B',
        soft:    '#8B94A8',
        snow:    '#F3F5F9',
        mint:    '#3DFF9A',
        mintDim: '#1A3D2E',
        danger:  '#FF4D6A',
        caution: '#FFB020',
      },
      fontFamily: {
        display: ['Syne', 'system-ui', 'sans-serif'],
        body: ['"DM Sans"', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        glow: '0 0 60px rgba(61,255,154,0.12)',
        lift: '0 20px 50px rgba(0,0,0,0.45)',
      },
      keyframes: {
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(18px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        sweep: {
          '0%': { transform: 'translateY(-30%)' },
          '100%': { transform: 'translateY(130%)' },
        },
      },
      animation: {
        fadeUp: 'fadeUp 0.6s cubic-bezier(0.22,1,0.36,1) both',
        sweep: 'sweep 1.6s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
