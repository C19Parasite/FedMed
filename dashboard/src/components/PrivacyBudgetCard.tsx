import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, ShieldCheck, EyeOff, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';

export const PrivacyBudgetCard: React.FC = () => {
  const navigate = useNavigate();
  const { privacy } = useDashboardStore();

  const consumedPercent = Math.min(100, Math.round((privacy.epsilon / privacy.total_budget) * 100));
  const strokeDashoffset = 251.2 - (251.2 * consumedPercent) / 100;

  return (
    <div className="relative overflow-hidden rounded-3xl p-6 sm:p-7 bg-gradient-to-br from-lavender-200 via-lavender-100 to-white dark:from-zinc-900 dark:via-zinc-800 dark:to-zinc-900 border border-lavender-300/80 dark:border-zinc-700 shadow-card flex flex-col justify-between">
      {/* 3D Swirl / Decorative Background Orb */}
      <div className="absolute -top-10 -right-10 w-44 h-44 rounded-full bg-gradient-to-tr from-lavender-400/40 to-lavender-300/10 blur-2xl pointer-events-none" />

      {/* Top Header */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-lavender-600 text-white shadow-sm">
            <Lock size={12} />
            <span>{privacy.homomorphic_encryption.scheme}</span>
          </div>

          <span className="text-xs font-bold text-lavender-800 dark:text-lavender-300 bg-white/70 dark:bg-zinc-800 px-2.5 py-1 rounded-full border border-lavender-200 dark:border-zinc-700">
            {privacy.homomorphic_encryption.status}
          </span>
        </div>

        <h3 className="text-lg font-bold text-zinc-900 dark:text-white leading-tight">
          Privacy Budget & Encryption
        </h3>
        <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-1 max-w-sm">
          Client-level Differential Privacy with analytical Rényi accounting + TenSEAL CKKS ciphertext additions.
        </p>
      </div>

      {/* Middle Gauge & Numbers */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 items-center my-4 py-2 border-y border-lavender-300/40 dark:border-zinc-800">
        {/* Left: Epsilon and Delta Metrics */}
        <div className="space-y-2">
          <div>
            <span className="text-[11px] font-bold text-zinc-400 uppercase tracking-wider block">
              Cumulative Loss (ε)
            </span>
            <div className="flex items-baseline gap-1.5">
              <span className="text-3xl font-black text-lavender-700 dark:text-lavender-400 font-sans">
                ε = {privacy.epsilon.toFixed(2)}
              </span>
              <span className="text-xs font-bold text-zinc-400">
                / {privacy.total_budget.toFixed(1)}
              </span>
            </div>
          </div>

          <div className="text-xs font-semibold text-zinc-600 dark:text-zinc-400 flex items-center gap-2">
            <span className="px-2 py-0.5 rounded-md bg-white dark:bg-zinc-800 border border-lavender-200 dark:border-zinc-700 font-mono">
              δ = 10⁻⁵
            </span>
            <span className="px-2 py-0.5 rounded-md bg-white dark:bg-zinc-800 border border-lavender-200 dark:border-zinc-700 font-mono">
              σ = {privacy.noise_sigma}
            </span>
          </div>
        </div>

        {/* Right: Circular Radial Progress */}
        <div className="flex items-center justify-center sm:justify-end gap-3">
          <div className="relative w-20 h-20 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90">
              <circle
                cx="40"
                cy="40"
                r="34"
                stroke="currentColor"
                strokeWidth="7"
                fill="transparent"
                className="text-lavender-200 dark:text-zinc-800"
              />
              <circle
                cx="40"
                cy="40"
                r="34"
                stroke="currentColor"
                strokeWidth="7"
                strokeDasharray="213.6"
                strokeDashoffset={213.6 - (213.6 * consumedPercent) / 100}
                strokeLinecap="round"
                fill="transparent"
                className="text-lavender-600 transition-all duration-700"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
              <span className="text-sm font-extrabold text-zinc-900 dark:text-white leading-none">
                {consumedPercent}%
              </span>
              <span className="text-[9px] text-zinc-400 font-bold uppercase mt-0.5">
                Budget
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Button */}
      <div className="flex items-center justify-between pt-1">
        <span className="text-xs text-zinc-500 font-medium">
          Zero raw MRI leakage guaranteed
        </span>

        <button
          onClick={() => navigate('/privacy')}
          className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-bold bg-surface-dark text-white hover:bg-black shadow-sm active:scale-95 transition-all group"
        >
          <span>View Privacy Report</span>
          <ArrowRight size={13} className="group-hover:translate-x-1 transition-transform" />
        </button>
      </div>
    </div>
  );
};
