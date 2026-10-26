import React from 'react';
import { Wifi, WifiOff, RefreshCw, Shield, Sparkles, Plus, Server, Search, Loader2, Settings } from 'lucide-react';
import { useDashboardStore } from '../store/useDashboardStore';
import { useIsFetching } from '@tanstack/react-query';
import { isMac } from './CommandMenu';

export const Header: React.FC = () => {
  const {
    connectionStatus,
    isMockMode,
    toggleMockMode,
    open2FAModal,
    openSettingsModal,
    rounds,
    nodes,
  } = useDashboardStore();

  const isFetching = useIsFetching();
  const activeNodesCount = nodes.filter((n) => n.status !== 'offline').length;
  const currentRound = rounds.length > 0 ? rounds[rounds.length - 1].round : 12;

  const openCommandPalette = () => {
    const event = new KeyboardEvent('keydown', {
      key: 'k',
      ctrlKey: !isMac,
      metaKey: isMac,
      bubbles: true,
    });
    document.dispatchEvent(event);
  };

  return (
    <header className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-4 border-b border-brand-border/60">
      {/* Title & Consortium Metadata */}
      <div>
        <div className="flex items-center gap-2.5">
          <h1 className="text-xl md:text-2xl font-bold tracking-tight text-zinc-900 dark:text-white">
            FedMed Consortium
          </h1>
          <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-lavender-100 text-lavender-700 dark:bg-lavender-900/40 dark:text-lavender-300 border border-lavender-200 dark:border-lavender-800">
            Cross-Silo FL
          </span>
          {/* Subtle TanStack Query Stale-While-Revalidate Indicator */}
          {isFetching > 0 && (
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-zinc-100 dark:bg-zinc-800 text-zinc-500 animate-pulse">
              <Loader2 size={10} className="animate-spin" />
              <span>Updating…</span>
            </span>
          )}
        </div>
        <p className="text-xs md:text-sm text-zinc-500 dark:text-zinc-400 mt-0.5">
          Brain Tumor MRI 3D Segmentation (MONAI U-Net • 4 Channels • TC/WT/ET)
        </p>
      </div>

      {/* Status Badges & Controls */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Quick Search / Command Menu Shortcut Pill */}
        <button
          onClick={openCommandPalette}
          className="hidden sm:inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-white dark:bg-zinc-800 border border-brand-border dark:border-zinc-700 text-zinc-600 dark:text-zinc-300 shadow-sm hover:border-lavender-300 transition-colors hit-slop"
          title="Open Command Palette"
        >
          <Search size={13} className="text-zinc-400" />
          <span>Quick Find</span>
          <kbd className="font-mono text-[10px] text-zinc-400 bg-zinc-100 dark:bg-zinc-700 px-1 py-0.2 rounded">
            {isMac ? '⌘K' : 'Ctrl+K'}
          </kbd>
        </button>

        {/* Stream Source Toggle (Live WS vs Mock Generator) */}
        <button
          onClick={toggleMockMode}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-all hit-slop ${
            isMockMode
              ? 'bg-lavender-100 dark:bg-lavender-950/60 text-lavender-700 dark:text-lavender-300 border border-lavender-300/80'
              : 'bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 border border-zinc-200 dark:border-zinc-700'
          }`}
          title="Toggle between Live WebSocket and Auto-simulating Mock Data"
        >
          <Sparkles size={13} className={isMockMode ? 'text-lavender-600 animate-spin-slow' : 'text-zinc-400'} />
          <span>{isMockMode ? 'Mock Simulation (1.5s)' : 'Live WebSocket'}</span>
        </button>

        {/* Connection Status Pill */}
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold ${
            connectionStatus === 'connected' || connectionStatus === 'mock'
              ? 'bg-mint-50 dark:bg-mint-950/40 text-mint-700 dark:text-mint-300 border border-mint-200 dark:border-mint-800'
              : 'bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-800'
          }`}
        >
          <span className="relative flex h-2 w-2">
            <span
              className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                connectionStatus === 'connected' || connectionStatus === 'mock' ? 'bg-mint-400' : 'bg-amber-400'
              }`}
            />
            <span
              className={`relative inline-flex rounded-full h-2 w-2 ${
                connectionStatus === 'connected' || connectionStatus === 'mock' ? 'bg-mint-500' : 'bg-amber-500'
              }`}
            />
          </span>
          <span>
            {connectionStatus === 'mock'
              ? 'Simulated Feed'
              : connectionStatus === 'connected'
              ? 'Aggregator Online'
              : 'Reconnecting to aggregator...'}
          </span>
        </div>

        {/* Quorum Pill */}
        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-white dark:bg-zinc-800 border border-brand-border dark:border-zinc-700 text-zinc-600 dark:text-zinc-300 shadow-sm tabular-nums">
          <Server size={13} className="text-lavender-600" />
          <span>Quorum: <strong>{activeNodesCount}/{nodes.length}</strong> Silos</span>
        </div>

        {/* Appearance & Theme Settings Button */}
        <button
          onClick={openSettingsModal}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-white dark:bg-zinc-800 border border-brand-border dark:border-zinc-700 text-zinc-600 dark:text-zinc-300 shadow-sm hover:border-lavender-400 hover:text-lavender-600 dark:hover:text-lavender-300 transition-all hit-slop cursor-pointer"
          title="Theme & Brand Appearance Settings"
          aria-label="Theme & Brand Appearance Settings"
        >
          <Settings size={13} className="text-lavender-600" />
          <span className="hidden sm:inline">Theme</span>
        </button>

        {/* Start New Round Action Button */}
        <button
          onClick={open2FAModal}
          className="flex items-center gap-1.5 px-4 py-2 rounded-full text-xs font-semibold bg-surface-dark dark:bg-lavender-600 text-white shadow-soft hover:shadow-lift active:scale-95 transition-all hit-slop"
        >
          <Plus size={14} strokeWidth={2.5} />
          <span className="tabular-nums">Start Round {currentRound + 1}</span>
        </button>
      </div>
    </header>
  );
};
