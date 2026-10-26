import React from 'react';
import { Calendar, ChevronRight, RotateCcw } from 'lucide-react';
import { useDashboardStore } from '../store/useDashboardStore';

export const RoundSelector: React.FC = () => {
  const { rounds, selectedRound, setSelectedRound } = useDashboardStore();

  const currentMonth = 'OCTOBER 2026';
  // Display up to the latest 8 rounds in the strip
  const displayRounds = rounds.slice(-8);

  return (
    <div className="card-base p-5 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800">
      {/* Month Strip Header */}
      <div className="flex items-center justify-between mb-3.5">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-lavender-100 dark:bg-lavender-900/40 text-lavender-700 dark:text-lavender-300 flex items-center justify-center">
            <Calendar size={14} />
          </div>
          <span className="text-xs font-bold uppercase tracking-wider text-zinc-700 dark:text-zinc-200">
            {currentMonth}
          </span>
        </div>

        {selectedRound !== null && (
          <button
            onClick={() => setSelectedRound(null)}
            className="flex items-center gap-1 text-[11px] font-semibold text-lavender-600 hover:text-lavender-700 hover:underline"
          >
            <RotateCcw size={11} />
            <span>Reset to Live</span>
          </button>
        )}
      </div>

      {/* Horizontal Round Pills (Calendar-style) */}
      <div className="flex items-center gap-2 overflow-x-auto no-scrollbar py-1">
        {displayRounds.map((r) => {
          const isSelected = selectedRound === r.round;
          const isLatest = r.round === rounds[rounds.length - 1].round && selectedRound === null;

          return (
            <button
              key={r.round}
              onClick={() => setSelectedRound(isSelected ? null : r.round)}
              className={`flex-1 min-w-[54px] py-2.5 px-2 rounded-2xl flex flex-col items-center justify-center transition-all ${
                isSelected
                  ? 'bg-surface-dark text-white shadow-card scale-105'
                  : isLatest
                  ? 'bg-lavender-100 text-lavender-800 border border-lavender-300 dark:bg-lavender-900/40 dark:text-lavender-200'
                  : 'bg-surface-soft hover:bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 border border-brand-border/60'
              }`}
            >
              <span className={`text-[10px] font-bold uppercase tracking-wider ${isSelected ? 'text-zinc-400' : 'text-zinc-400'}`}>
                Round
              </span>
              <span className="text-sm font-extrabold font-mono mt-0.5">
                {r.round}
              </span>
              <span
                className={`w-1.5 h-1.5 rounded-full mt-1 ${
                  isSelected ? 'bg-mint-400' : isLatest ? 'bg-lavender-600' : 'bg-transparent'
                }`}
              />
            </button>
          );
        })}
      </div>
    </div>
  );
};
