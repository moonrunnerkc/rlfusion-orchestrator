/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      borderRadius: {
        'lg': '0.5rem',
      },
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        primary: "#10b981",
        secondary: "#8b5cf6",
        'bubble-user': '#3b82f6',
        'bubble-rlfo': '#4b5563',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}
