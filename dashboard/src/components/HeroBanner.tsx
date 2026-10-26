import React from 'react';
import { Plus, ShieldCheck, Sparkles, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';
import { ShortcutBadge } from './CommandMenu';

export const HeroBanner: React.FC = () => {
  const { open2FAModal, rounds } = useDashboardStore();
  const currentRound = rounds.length > 0 ? rounds[rounds.length - 1].round : 12;

  return (
    <div className="relative overflow-hidden hero-gradient rounded-3xl p-6 sm:p-8 shadow-card flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
      {/* Background Decorative Blur Orbs */}
      <div className="absolute -top-12 -left-12 w-48 h-48 bg-lavender-300/40 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute -bottom-16 right-36 w-60 h-60 bg-lavender-400/30 rounded-full blur-3xl pointer-events-none" />

      {/* Text & Action CTA */}
      <div className="relative z-10 max-w-xl">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-white/70 backdrop-blur-md text-lavender-800 border border-lavender-200/80 mb-3 shadow-sm">
          <ShieldCheck size={14} className="text-lavender-600" />
          <span>Cross-Silo Privacy: Patient Data Never Leaves Hospitals</span>
        </div>

        <h2 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold tracking-tight text-zinc-900 leading-tight">
          Federated learning works <br className="hidden sm:inline" />
          <span className="text-lavender-700">better together.</span>
        </h2>

        <p className="text-sm sm:text-base text-zinc-600 font-medium mt-2 max-w-md">
          Three hospitals. Zero shared patient data. One smarter brain tumor model.
        </p>

        <div className="flex flex-wrap items-center gap-3 mt-5">
          <button
            onClick={open2FAModal}
            className="inline-flex items-center gap-2.5 px-6 py-3 rounded-full text-sm font-semibold bg-surface-dark text-white hover:bg-black shadow-card hover:shadow-lift active:scale-95 transition-all group hit-slop"
            title="Start new federated round (requires 2FA verification code)"
          >
            <Plus size={16} strokeWidth={2.5} className="group-hover:rotate-90 transition-transform duration-200" />
            <span className="tabular-nums">Start Round {currentRound + 1}</span>
            <kbd className="hidden sm:inline-flex items-center text-[10px] font-mono bg-white/20 text-white/90 px-1.5 py-0.5 rounded">
              N
            </kbd>
          </button>

          <div className="flex items-center gap-2 text-xs font-semibold text-zinc-500 bg-white/50 backdrop-blur-sm px-3.5 py-2.5 rounded-full border border-lavender-200/60 tabular-nums">
            <span className="w-2 h-2 rounded-full bg-mint-500 animate-pulse" />
            <span>Target: 30 Rounds</span>
          </div>
        </div>
      </div>

      {/* Decorative 3D Purple Flower / Abstract Graphic */}
      <div className="relative z-10 hidden sm:flex items-center justify-center shrink-0 pr-4">
        <motion.div
          animate={{
            y: [-4, 6, -4],
            rotate: [0, 4, 0],
          }}
          transition={{
            duration: 5,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
          className="relative w-36 h-36 lg:w-44 lg:h-44 flex items-center justify-center select-none"
        >
          {/* Petals 3D Layered Flower */}
          <div className="absolute w-16 h-28 rounded-full flower-petal transform -rotate-45 -translate-y-4" />
          <div className="absolute w-16 h-28 rounded-full flower-petal transform rotate-45 -translate-y-4" />
          <div className="absolute w-16 h-28 rounded-full flower-petal transform rotate-0 translate-y-4 opacity-90" />
          <div className="absolute w-28 h-16 rounded-full flower-petal transform -rotate-15 -translate-x-3 opacity-95" />
          <div className="absolute w-28 h-16 rounded-full flower-petal transform rotate-15 translate-x-3 opacity-95" />

          {/* Central Golden Core */}
          <div className="relative z-20 w-12 h-12 rounded-full flower-core flex items-center justify-center text-amber-900 font-black text-sm">
            ✨
          </div>

          {/* Orbiting Sparkle Element */}
          <motion.div
            animate={{
              rotate: 360,
            }}
            transition={{
              duration: 12,
              repeat: Infinity,
              ease: 'linear',
            }}
            className="absolute inset-0 pointer-events-none"
          >
            <div className="w-3.5 h-3.5 rounded-full bg-white shadow-glow-lavender border border-lavender-300 transform -translate-x-2 translate-y-2" />
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};
