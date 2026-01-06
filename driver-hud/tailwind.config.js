/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        porsche: {
          red: '#D5001C',
          black: '#000000',
          gray: {
            900: '#0a0a0a',
            800: '#1a1a1a',
            700: '#2d2d2d',
            600: '#404040',
            500: '#525252',
            400: '#6b6b6b',
            300: '#8a8a8a',
          },
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
