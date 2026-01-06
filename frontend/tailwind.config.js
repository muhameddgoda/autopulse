/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Porsche-inspired color palette
        porsche: {
          red: '#D5001C',
          black: '#000000',
          gray: {
            900: '#1a1a1a',
            800: '#2d2d2d',
            700: '#404040',
            600: '#525252',
            500: '#6b6b6b',
            400: '#8a8a8a',
            300: '#a3a3a3',
            200: '#d4d4d4',
            100: '#f5f5f5',
          },
          gold: '#C9A227',
          silver: '#8E8E8E',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
