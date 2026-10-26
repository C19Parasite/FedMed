import React from 'react';
import { Hospital, Settings, Activity, WifiOff, CheckCircle2, ShieldAlert } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';
import { HospitalNode } from '../types';

export const HospitalNodesRow: React.FC = () => {
  const { nodes } = useDashboardStore();

  return (
    <div className="card-base p-6 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-lavender-100 dark:bg-lavender-900/40 text-lavender-700 dark:text-lavender-300 flex items-center justify-center font-bold text-sm">
            <Hospital size={16} />
          </div>
          <div>
            <h3 className="text-base font-bold text-zinc-900 dark:text-white leading-tight">
              Hospital Nodes
            </h3>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Isolated on-premise compute silos • Dynamic quorum (min 2 of 3)
            </p>
          </div>
        </div>

        <button
          className="w-8 h-8 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 flex items-center justify-center transition-colors"
          title="Node Configuration"
          aria-label="Node Settings"
        >
          <Settings size={16} />
        </button>
      </div>

      {/* 3 Node Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {nodes.map((node, idx) => (
          <NodeCard key={node.id} node={node} index={idx} />
        ))}
      </div>
    </div>
  );
};

interface NodeCardProps {
  node: HospitalNode;
  index: number;
}

const NodeCard: React.FC<NodeCardProps> = ({ node, index }) => {
  const isOffline = node.status === 'offline';
  const isTraining = node.status === 'training';

  // Specific initials and color accents per hospital
  const avatars: Record<string, { initials: string; bg: string; text: string }> = {
    hospital_a: { initials: 'HA', bg: 'bg-lavender-100 dark:bg-lavender-900/50', text: 'text-lavender-700 dark:text-lavender-300' },
    hospital_b: { initials: 'HB', bg: 'bg-mint-100 dark:bg-mint-900/50', text: 'text-mint-700 dark:text-mint-300' },
    hospital_c: { initials: 'HC', bg: 'bg-amber-100 dark:bg-amber-900/50', text: 'text-amber-700 dark:text-amber-300' },
  };

  const avatar = avatars[node.id] || { initials: 'H', bg: 'bg-zinc-100', text: 'text-zinc-700' };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.08 }}
      whileHover={{ y: isOffline ? 0 : -3 }}
      className={`rounded-2xl p-4 border transition-all duration-300 flex flex-col justify-between ${
        isOffline
          ? 'bg-zinc-50 dark:bg-zinc-900/40 border-zinc-200 dark:border-zinc-800 text-zinc-400 opacity-75'
          : 'bg-surface-soft dark:bg-zinc-800/40 border-brand-border dark:border-zinc-700/80 hover:border-lavender-300 dark:hover:border-lavender-700 hover:shadow-soft'
      }`}
    >
      {/* Top Row: Avatar + Status */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div
            className={`w-11 h-11 rounded-2xl flex items-center justify-center font-bold text-sm tracking-wider shadow-sm transition-colors ${
              isOffline ? 'bg-zinc-200 dark:bg-zinc-800 text-zinc-500' : `${avatar.bg} ${avatar.text}`
            }`}
          >
            {avatar.initials}
          </div>

          <div>
            <h4
              className={`text-sm font-bold truncate max-w-[140px] sm:max-w-[160px] ${
                isOffline ? 'text-zinc-500 line-through' : 'text-zinc-900 dark:text-white'
              }`}
            >
              {node.name}
            </h4>
            <p className="text-[11px] text-zinc-500 dark:text-zinc-400 font-medium">{node.city}</p>
          </div>
        </div>

        {/* Status indicator dot */}
        <div className="flex items-center gap-1.5 shrink-0">
          {isOffline ? (
            <span className="w-2.5 h-2.5 rounded-full bg-zinc-400" />
          ) : isTraining ? (
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-lavender-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-lavender-600" />
            </span>
          ) : (
            <span className="w-2.5 h-2.5 rounded-full bg-mint-500 shadow-sm" />
          )}
          <span className="text-[11px] font-semibold capitalize text-zinc-600 dark:text-zinc-300">
            {node.status}
          </span>
        </div>
      </div>

      {/* Resilience Banner when Offline */}
      {isOffline ? (
        <div className="mt-3 py-1.5 px-2.5 rounded-xl bg-coral-50 dark:bg-coral-950/40 border border-coral-200 dark:border-coral-800/60 flex items-center gap-1.5">
          <WifiOff size={13} className="text-coral-500 shrink-0" />
          <span className="text-[11px] font-bold text-coral-600 dark:text-coral-400 leading-tight">
            Dropped, round continues (2/3)
          </span>
        </div>
      ) : (
        <div className="mt-4 pt-3 border-t border-brand-border/60 dark:border-zinc-700/40 grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-[10px] uppercase tracking-wider text-zinc-400 font-bold block">
              Private Cohort
            </span>
            <span className="font-bold text-zinc-800 dark:text-zinc-200 font-mono">
              {node.samples} MRIs
            </span>
          </div>

          <div className="text-right">
            <span className="text-[10px] uppercase tracking-wider text-zinc-400 font-bold block">
              Latency / Norm
            </span>
            <span className="font-bold text-zinc-800 dark:text-zinc-200 font-mono">
              {node.latencyMs}ms • ||g|| {node.gradientNorm}
            </span>
          </div>
        </div>
      )}
    </motion.div>
  );
};
