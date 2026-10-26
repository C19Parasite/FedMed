import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import { Activity, ShieldCheck, Layers, Info } from 'lucide-react';
import { useDashboardStore } from '../store/useDashboardStore';

export const TrainingStatistics: React.FC = () => {
  const { rounds, viewMode, setViewMode, nodes } = useDashboardStore();

  const activeNodesCount = nodes.filter((n) => n.status !== 'offline').length;
  const currentRound = rounds.length > 0 ? rounds[rounds.length - 1].round : 12;
  const encryptedUpdatesCount = rounds.length * 3;

  // Chart data formatted
  const chartData = rounds.map((r) => ({
    round: `R${r.round}`,
    roundNum: r.round,
    dice: parseFloat((r.dice * 100).toFixed(1)),
    diceRaw: r.dice,
    loss: parseFloat(r.global_loss.toFixed(4)),
    baseline: parseFloat(((r.baseline_dice ?? 0.845) * 100).toFixed(1)),
  }));

  const baselinePercent = 84.5;

  return (
    <div className="card-base p-6 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800">
      {/* Top Header Row with Toggle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-brand-border/60 dark:border-zinc-800">
        <div>
          <h3 className="text-base font-bold text-zinc-900 dark:text-white">
            Training Statistics
          </h3>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Convergence curves across federated communication rounds
          </p>
        </div>

        {/* View Mode Toggle (Per Round | Per Epoch) */}
        <div className="flex items-center p-1 bg-surface-soft dark:bg-zinc-800 rounded-2xl border border-brand-border/70 dark:border-zinc-700/60 self-start sm:self-auto">
          <button
            onClick={() => setViewMode('round')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              viewMode === 'round'
                ? 'bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white shadow-sm'
                : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'
            }`}
          >
            Per Round
          </button>
          <button
            onClick={() => setViewMode('epoch')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
              viewMode === 'epoch'
                ? 'bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white shadow-sm'
                : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'
            }`}
          >
            Per Epoch
          </button>
        </div>
      </div>

      {/* Big Round Progress Indicator & Two Small Stat Boxes */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 my-5 items-center">
        {/* Big Progress */}
        <div className="flex flex-col">
          <span className="text-xs font-bold uppercase tracking-wider text-zinc-400">
            Progress Completed
          </span>
          <div className="flex items-baseline gap-2 mt-1 tabular-nums">
            <span className="text-3xl lg:text-4xl font-extrabold text-zinc-900 dark:text-white font-sans">
              Round {currentRound}
            </span>
            <span className="text-base font-bold text-zinc-400">
              | 30
            </span>
          </div>
        </div>

        {/* Stat Box 1: Nodes Active */}
        <div className="p-3.5 rounded-2xl bg-surface-soft dark:bg-zinc-800/50 border border-brand-border/60 dark:border-zinc-700/50 flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-mint-100 dark:bg-mint-950/60 text-mint-700 dark:text-mint-300 flex items-center justify-center font-bold">
            <Activity size={17} />
          </div>
          <div>
            <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider block">
              Nodes Active
            </span>
            <span className="text-base font-extrabold text-zinc-900 dark:text-white tabular-nums">
              {activeNodesCount} / {nodes.length} Silos
            </span>
          </div>
        </div>

        {/* Stat Box 2: Encrypted Updates */}
        <div className="p-3.5 rounded-2xl bg-surface-soft dark:bg-zinc-800/50 border border-brand-border/60 dark:border-zinc-700/50 flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-lavender-100 dark:bg-lavender-950/60 text-lavender-700 dark:text-lavender-300 flex items-center justify-center font-bold">
            <ShieldCheck size={17} />
          </div>
          <div>
            <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider block">
              Encrypted Updates
            </span>
            <span className="text-base font-extrabold text-zinc-900 dark:text-white tabular-nums">
              {encryptedUpdatesCount} Ciphertexts
            </span>
          </div>
        </div>
      </div>

      {/* Main Recharts Area & Line Chart */}
      <div className="h-72 w-full mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -15, bottom: 0 }}>
            <defs>
              <linearGradient id="diceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#34D399" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#34D399" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="lossGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--theme-primary, #7C5CFA)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="var(--theme-primary, #7C5CFA)" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="var(--theme-card-border, #EBE7F5)" vertical={false} />
            <XAxis dataKey="round" stroke="#9CA3AF" tick={{ fontSize: 11 }} />
            <YAxis
              yAxisId="left"
              stroke="#9CA3AF"
              domain={[60, 100]}
              unit="%"
              tick={{ fontSize: 11 }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#9CA3AF"
              domain={[0, 0.8]}
              tick={{ fontSize: 11 }}
            />

            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: '12px', fontSize: '12px' }}
              iconType="circle"
            />

            {/* Centralized Baseline Reference Line */}
            <ReferenceLine
              yAxisId="left"
              y={baselinePercent}
              stroke="#F59E0B"
              strokeDasharray="4 4"
              strokeWidth={2}
              label={{
                value: 'Centralized Baseline (84.5%)',
                fill: '#D97706',
                position: 'insideTopLeft',
                fontSize: 11,
                fontWeight: 600,
              }}
            />

            {/* Global Dice Curve */}
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="dice"
              name="Global Val Dice (%)"
              stroke="#10B981"
              strokeWidth={3}
              fillOpacity={1}
              fill="url(#diceGradient)"
            />

            {/* Global Loss Curve */}
            <Area
              yAxisId="right"
              type="monotone"
              dataKey="loss"
              name="Global Val Loss"
              stroke="var(--theme-primary, #7C5CFA)"
              strokeWidth={2.5}
              fillOpacity={1}
              fill="url(#lossGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

interface TooltipProps {
  active?: boolean;
  payload?: any[];
  label?: string;
}

const CustomTooltip: React.FC<TooltipProps> = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;

  return (
    <div className="p-3.5 bg-white dark:bg-zinc-900 rounded-2xl shadow-lift border border-brand-border dark:border-zinc-800 text-xs">
      <div className="font-bold text-zinc-900 dark:text-white pb-1.5 mb-1.5 border-b border-brand-border/60">
        Round {label} Evaluation
      </div>
      <div className="space-y-1">
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-mint-600 font-semibold">
            <span className="w-2 h-2 rounded-full bg-mint-500" />
            Global Dice:
          </span>
          <span className="font-mono font-bold text-zinc-800 dark:text-zinc-100">
            {payload[0]?.value}%
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5 text-lavender-600 font-semibold">
            <span className="w-2 h-2 rounded-full bg-lavender-600" />
            Global Loss:
          </span>
          <span className="font-mono font-bold text-zinc-800 dark:text-zinc-100">
            {payload[1]?.value}
          </span>
        </div>
        <div className="flex items-center justify-between gap-4 text-zinc-400 text-[11px] pt-1">
          <span>Centralized Target:</span>
          <span className="font-mono text-amber-600 font-semibold">84.5%</span>
        </div>
      </div>
    </div>
  );
};
