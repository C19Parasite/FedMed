/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        shell: 'var(--theme-shell, #F3F2F8)',
        surface: {
          DEFAULT: 'var(--theme-card, #FFFFFF)',
          soft: 'var(--theme-surface-soft, #F8F7FC)',
          card: 'var(--theme-card, #FFFFFF)',
          dark: 'var(--theme-sidebar-bg, #111116)',
          darker: 'var(--theme-sidebar-darker, #0B0B0E)',
        },
        lavender: {
          50: 'var(--theme-p50, #F8F6FE)',
          100: 'var(--theme-p100, #F1EDFF)',
          200: 'var(--theme-p200, #E4DCFE)',
          300: 'var(--theme-p300, #D9CFFB)',
          400: 'var(--theme-secondary, #B9A7F5)',
          500: 'var(--theme-p500, #9475F7)',
          600: 'var(--theme-primary, #7C5CFA)',
          700: 'var(--theme-p700, #6441E6)',
          800: 'var(--theme-p800, #4D2DC0)',
          900: 'var(--theme-p900, #341A8E)',
          950: 'var(--theme-p950, #1F0D59)',
        },
        theme: {
          primary: 'var(--theme-primary, #7857FF)',
          secondary: 'var(--theme-secondary, #A78BFA)',
          bg: 'var(--theme-bg, #F4F2FA)',
          shell: 'var(--theme-shell, #E9E7F2)',
          card: 'var(--theme-card, #FFFFFF)',
          text: 'var(--theme-text, #191827)',
          muted: 'var(--theme-text-muted, #71717A)',
          border: 'var(--theme-card-border, #EBE7F5)',
          accent: 'var(--theme-accent, #22C55E)',
        },
        // Clinical BraTS Subregion Palette (Strictly preserved across all themes)
        mint: {
          50: '#F0FDF4',
          100: '#DCFCE7',
          200: '#BBF7D0',
          300: '#86EFAC',
          400: '#4ADE80',
          500: '#22C55E',
          600: '#16A34A',
          700: '#15803D',
        },
        amber: {
          50: '#FFFBEB',
          100: '#FEF3C7',
          400: '#FBBF24',
          500: '#F59E0B',
          600: '#D97706',
        },
        coral: {
          50: '#FEF2F2',
          100: '#FEE2E2',
          400: '#F87171',
          500: '#EF4444',
          600: '#DC2626',
        },
        brand: {
          dark: 'var(--theme-sidebar-bg, #111116)',
          grey: 'var(--theme-text-muted, #6B7280)',
          muted: 'var(--theme-text-muted, #9CA3AF)',
          border: 'var(--theme-card-border, #EBE7F5)',
        }
      },
      borderRadius: {
        '2xl': '20px',
        '3xl': '24px',
        '4xl': '32px',
        '5xl': '40px',
      },
      boxShadow: {
        'soft': '0 8px 30px var(--theme-shadow-soft, rgba(120, 100, 180, 0.08))',
        'card': '0 12px 35px var(--theme-shadow-card, rgba(120, 100, 180, 0.09))',
        'lift': '0 16px 40px var(--theme-shadow-lift, rgba(120, 100, 180, 0.14))',
        'glow-lavender': '0 0 25px var(--theme-primary-glow, rgba(120, 87, 255, 0.35))',
        'glow-mint': '0 0 20px rgba(52, 211, 153, 0.35)',
        'sidebar': '0 10px 40px rgba(17, 17, 22, 0.25)',
      },
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'float': 'float 4s ease-in-out infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px) rotate(0deg)' },
          '50%': { transform: 'translateY(-8px) rotate(2deg)' },
        }
      }
    },
  },
  plugins: [],
}
