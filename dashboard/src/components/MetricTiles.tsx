import React from 'react';
import { TrendingUp, TrendingDown, Award, Flame, BarChart3, ArrowUpRight } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';

export const MetricTiles: React.FC = () => {
  const { rounds, selectedRound } = useDashboardStore();

  const activeRound = selectedRound
    ? rounds.find((r) => r.round === selectedRound) || rounds[rounds.length - 1]
    : rounds[rounds.length - 1];

  const currentDice = activeRound ? activeRound.dice : 0.874;
  const baselineDice = activeRound ? activeRound.baseline_dice : 0.845;
  const diceDiff = currentDice - baselineDice;
  const currentLoss = activeRound ? activeRound.global_loss : 0.306;

  // Mini Sparkline points for loss
  const lossPoints = rounds.slice(-8).map((r) => r.global_loss);
  const minLoss = Math.min(...lossPoints, 0.2);
  const maxLoss = Math.max(...lossPoints, 0.7);
  const sparklineCoords = lossPoints
    .map((val, idx) => {
      const x = (idx / (lossPoints.length - 1 || 1)) * 100;
      const y = 35 - ((val - minLoss) / (maxLoss - minLoss || 1)) * 30;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
      {/* Tile 1: Global Dice Score */}
      <motion.div
        whileHover={{ y: -4 }}
        className="card-base p-6 bg-white dark:bg-zinc-900 flex flex-col justify-between"
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-zinc-400">
            Global Dice Score
          </span>
          <div className="w-8 h-8 rounded-xl bg-mint-100 dark:bg-mint-950/60 text-mint-700 dark:text-mint-300 flex items-center justify-center">
            <Award size={16} />
          </div>
        </div>

        <div className="mt-3">
          <div className="flex items-baseline gap-2">
            <span className="text-4xl lg:text-5xl font-extrabold text-zinc-900 dark:text-white font-sans tracking-tight tabular-nums">
              {currentDice.toFixed(3)}
            </span>
            <span className="text-xs font-bold text-zinc-400 font-mono">
              / 1.000
            </span>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-zinc-100 dark:bg-zinc-800 rounded-full h-2.5 mt-3 overflow-hidden">
            <div
              className="bg-gradient-to-r from-mint-400 to-mint-500 h-2.5 rounded-full transition-all duration-700 shadow-glow-mint"
              style={{ width: `${currentDice * 100}%` }}
            />
          </div>
        </div>

        <div className="mt-4 pt-3 border-t border-brand-border/60 dark:border-zinc-800 flex items-center justify-between text-xs font-semibold tabular-nums">
          <span className="flex items-center gap-1 text-mint-600 dark:text-mint-400">
            <TrendingUp size={14} />
            <span>+{((diceDiff) * 100).toFixed(1)}% vs Baseline</span>
          </span>
          <span className="text-zinc-400 font-mono">Base: {baselineDice.toFixed(3)}</span>
        </div>
      </motion.div>

      {/* Tile 2: Global Loss */}
      <motion.div
        whileHover={{ y: -4 }}
        className="card-base p-6 bg-white dark:bg-zinc-900 flex flex-col justify-between"
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-zinc-400">
            Global Loss (DiceCE)
          </span>
          <div className="w-8 h-8 rounded-xl bg-lavender-100 dark:bg-lavender-950/60 text-lavender-700 dark:text-lavender-300 flex items-center justify-center">
            <Flame size={16} />
          </div>
        </div>

        <div className="mt-3">
          <div className="flex items-baseline justify-between">
            <span className="text-4xl lg:text-5xl font-extrabold text-zinc-900 dark:text-white font-sans tracking-tight tabular-nums">
              {currentLoss.toFixed(4)}
            </span>

            {/* SVG Sparkline */}
            <div className="w-24 h-10">
              <svg viewBox="0 0 100 40" className="w-full h-full overflow-visible">
                <polyline
                  fill="none"
                  stroke="var(--theme-primary, #7C5CFA)"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={sparklineCoords}
                />
              </svg>
            </div>
          </div>
        </div>

        <div className="mt-4 pt-3 border-t border-brand-border/60 dark:border-zinc-800 flex items-center justify-between text-xs font-semibold">
          <span className="flex items-center gap-1 text-lavender-600 dark:text-lavender-400">
            <TrendingDown size={14} />
            <span>-52.3% since Round 1</span>
          </span>
          <span className="text-zinc-400 font-mono">R{activeRound?.round || 12}</span>
        </div>
      </motion.div>

      {/* Tile 3: Aggregation Impact (Lavender Vertical Bar Chart) */}
      <motion.div
        whileHover={{ y: -4 }}
        className="card-base p-6 bg-white dark:bg-zinc-900 flex flex-col justify-between"
      >
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-zinc-400">
            Node Impact Weight
          </span>
          <div className="w-8 h-8 rounded-xl bg-lavender-100 dark:bg-lavender-950/60 text-lavender-700 dark:text-lavender-300 flex items-center justify-center">
            <BarChart3 size={16} />
          </div>
        </div>

        {/* 3 Vertical Lavender Bars */}
        <div className="flex items-end justify-around h-16 mt-2 pt-2 px-2">
          <div className="flex flex-col items-center gap-1.5 h-full justify-end">
            <span className="text-[10px] font-bold text-zinc-600 dark:text-zinc-300">42%</span>
            <div
              className="w-7 rounded-t-lg bg-lavender-600 transition-all duration-500 shadow-sm"
              style={{ height: '42%' }}
            />
            <span className="text-[10px] font-bold text-zinc-400 uppercase">H-A</span>
          </div>

          <div className="flex flex-col items-center gap-1.5 h-full justify-end">
            <span className="text-[10px] font-bold text-zinc-600 dark:text-zinc-300">31%</span>
            <div
              className="w-7 rounded-t-lg bg-lavender-400 transition-all duration-500 shadow-sm"
              style={{ height: '31%' }}
            />
            <span className="text-[10px] font-bold text-zinc-400 uppercase">H-B</span>
          </div>

          <div className="flex flex-col items-center gap-1.5 h-full justify-end">
            <span className="text-[10px] font-bold text-zinc-600 dark:text-zinc-300">27%</span>
            <div
              className="w-7 rounded-t-lg bg-lavender-300 transition-all duration-500 shadow-sm"
              style={{ height: '27%' }}
            />
            <span className="text-[10px] font-bold text-zinc-400 uppercase">H-C</span>
          </div>
        </div>

        <div className="mt-4 pt-3 border-t border-brand-border/60 dark:border-zinc-800 flex items-center justify-between text-xs font-medium text-zinc-500 dark:text-zinc-400">
          <span>Weighted FedAvg</span>
          <span className="font-mono text-zinc-400">∑ n_i = 1,240</span>
        </div>
      </motion.div>
    </div>
  );
};
