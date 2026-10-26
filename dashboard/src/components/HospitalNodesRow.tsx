import React from 'react';
import { Hospital, Settings, Activity, WifiOff, CheckCircle2, ShieldCheck, Power } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';
import { HospitalNode } from '../types';
import { toast } from 'sonner';

export const HospitalNodesRow: React.FC = () => {
  const { nodes } = useDashboardStore();

  return (
    <div className="card-base p-5 sm:p-6 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-8 h-8 rounded-xl bg-lavender-100 dark:bg-lavender-900/40 text-lavender-700 dark:text-lavender-300 flex items-center justify-center font-bold text-sm shrink-0">
            <Hospital size={16} />
          </div>
          <div className="min-w-0">
            <h3 className="text-base font-bold text-zinc-900 dark:text-white leading-tight truncate">
              Hospital Nodes
            </h3>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 truncate">
              Isolated on-premise compute silos • Dynamic quorum (min 2 of 3)
            </p>
          </div>
        </div>

        <button
          className="w-8 h-8 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 flex items-center justify-center transition-colors hit-slop shrink-0"
          title="Node Configuration"
          aria-label="Node Settings"
        >
          <Settings size={16} />
        </button>
      </div>

      {/* 3 Equal-Width Hospital Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-stretch">
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
  const { toggleNodeStatus } = useDashboardStore();
  const isOffline = node.status === 'offline';
  const isTraining = node.status === 'training';

  const handleToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    const willBeOffline = !isOffline;
    toggleNodeStatus(node.id);

    toast(willBeOffline ? `${node.name} disconnected` : `${node.name} reconnected`, {
      description: willBeOffline
        ? 'Simulated failure: node dropped from round; quorum held at 2/3.'
        : 'Node resumed participation in federated aggregation.',
      duration: 6000,
      action: {
        label: 'Undo',
        onClick: () => {
          toggleNodeStatus(node.id);
          toast.success(`Restored state for ${node.name}`);
        },
      },
    });
  };

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
      transition={{ duration: 0.25, delay: index * 0.05 }}
      className={`rounded-2xl p-4 border transition-all duration-300 flex flex-col justify-between h-full min-w-0 ${
        isOffline
          ? 'bg-zinc-50/90 dark:bg-zinc-900/40 border-zinc-200 dark:border-zinc-800/80 text-zinc-400'
          : 'bg-surface-soft dark:bg-zinc-800/40 border-brand-border dark:border-zinc-700/80 hover:border-lavender-300 dark:hover:border-lavender-700 hover:shadow-soft'
      }`}
    >
      <div>
        {/* Top Header: Avatar + Hospital Name & Location + Anchored Power Toggle */}
        <div className="flex items-start justify-between gap-2.5">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div
              className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-xs tracking-wider shadow-sm transition-colors shrink-0 ${
                isOffline ? 'bg-zinc-200 dark:bg-zinc-800 text-zinc-400' : `${avatar.bg} ${avatar.text}`
              }`}
            >
              {avatar.initials}
            </div>

            <div className="min-w-0 flex-1">
              <h4
                className={`text-sm font-bold truncate leading-snug ${
                  isOffline ? 'text-zinc-500 line-through' : 'text-zinc-900 dark:text-white'
                }`}
                title={node.name}
              >
                {node.name}
              </h4>
              <p className="text-[11px] text-zinc-500 dark:text-zinc-400 font-medium truncate">
                {node.city}
              </p>
            </div>
          </div>

          {/* Power toggle control inside card boundary */}
          <button
            onClick={handleToggle}
            className="w-8 h-8 rounded-lg hover:bg-zinc-200/80 dark:hover:bg-zinc-700 text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 flex items-center justify-center transition-colors hit-slop shrink-0 focus:outline-none focus:ring-2 focus:ring-lavender-400"
            title={isOffline ? `Reconnect ${node.name}` : `Simulate node drop for ${node.name}`}
            aria-label={`Toggle ${node.name} connection status`}
          >
            <Power
              size={14}
              className={isOffline ? 'text-coral-500' : 'text-mint-600 hover:text-coral-500'}
            />
          </button>
        </div>

        {/* Dedicated Status Badge Row */}
        <div className="mt-3 flex items-center gap-2">
          <div
            className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold ${
              isOffline
                ? 'bg-zinc-200/80 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400'
                : isTraining
                ? 'bg-lavender-100 text-lavender-700 dark:bg-lavender-900/50 dark:text-lavender-300'
                : 'bg-mint-100 text-mint-700 dark:bg-mint-900/50 dark:text-mint-300'
            }`}
          >
            {isOffline ? (
              <span className="w-2 h-2 rounded-full bg-zinc-400 shrink-0" />
            ) : isTraining ? (
              <span className="relative flex h-2 w-2 shrink-0">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-lavender-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-lavender-600" />
              </span>
            ) : (
              <span className="w-2 h-2 rounded-full bg-mint-500 shrink-0" />
            )}
            <span className="capitalize">{node.status}</span>
          </div>
        </div>

        {/* Resilience Notification or Security Verification (Constant height ensures perfect card alignment) */}
        <div className="h-8 flex items-center mt-2.5">
          {isOffline ? (
            <div className="w-full py-1 px-2.5 rounded-xl bg-coral-50 dark:bg-coral-950/40 border border-coral-200 dark:border-coral-800/60 flex items-center gap-1.5 text-coral-600 dark:text-coral-400 text-[11px] font-bold">
              <WifiOff size={13} className="shrink-0" />
              <span className="truncate">Dropped, round continues (2/3)</span>
            </div>
          ) : (
            <div className="w-full py-1 px-2.5 rounded-xl bg-white/70 dark:bg-zinc-850 border border-brand-border/60 dark:border-zinc-700/50 flex items-center justify-between text-[11px] text-zinc-500 dark:text-zinc-400 font-medium">
              <span className="flex items-center gap-1 text-[10px] uppercase font-bold tracking-wider text-zinc-400">
                <ShieldCheck size={12} className="text-lavender-600 shrink-0" /> TLS 1.3
              </span>
              <span className="font-mono text-[10px] text-mint-600 dark:text-mint-400 font-bold">
                Quorum Verified
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Footer Metrics Row: Consistently aligned along bottom edge across all cards */}
      <div className="mt-3.5 pt-3 border-t border-brand-border/60 dark:border-zinc-700/40 grid grid-cols-2 gap-2 text-xs">
        <div className="min-w-0">
          <span className="text-[10px] uppercase tracking-wider text-zinc-400 font-bold block truncate">
            Private Cohort
          </span>
          <span
            className={`font-bold font-mono text-xs block truncate ${
              isOffline ? 'text-zinc-400' : 'text-zinc-800 dark:text-zinc-200'
            }`}
          >
            {node.samples} MRIs
          </span>
        </div>

        <div className="text-right min-w-0">
          <span className="text-[10px] uppercase tracking-wider text-zinc-400 font-bold block truncate">
            Latency / Norm
          </span>
          <span
            className={`font-bold font-mono text-xs block truncate ${
              isOffline ? 'text-zinc-400' : 'text-zinc-800 dark:text-zinc-200'
            }`}
          >
            {node.latencyMs}ms • ||g|| {node.gradientNorm}
          </span>
        </div>
      </div>
    </motion.div>
  );
};
