/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
theme: {
    extend: {
      // 1. Define the animation rules
      keyframes: {
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        }
      },
      // 2. Name the class you will use in your HTML
      animation: {
        'fade-up': 'fadeUp 0.5s ease-out forwards',
      }
    },
  },
  plugins: [],
}

