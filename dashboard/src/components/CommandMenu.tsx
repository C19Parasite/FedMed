import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Command } from 'cmdk';
import {
  LayoutDashboard,
  Network,
  History,
  Scan,
  ShieldCheck,
  Plus,
  Moon,
  Sun,
  FileSpreadsheet,
  RotateCcw,
  Search,
} from 'lucide-react';
import { useDashboardStore } from '../store/useDashboardStore';
import { transitionView } from '../lib/viewTransitions';
import { toast } from 'sonner';

export const isMac =
  typeof window !== 'undefined' &&
  /Mac|iPod|iPhone|iPad/.test(window.navigator.userAgent);

export const ShortcutBadge: React.FC<{ keys: string[] }> = ({ keys }) => {
  return (
    <span className="inline-flex items-center gap-1 font-mono text-[10px] font-semibold text-zinc-400 bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 px-1.5 py-0.5 rounded-md">
      {keys.map((k) => (k === 'meta' ? (isMac ? '⌘' : 'Ctrl') : k))}
    </span>
  );
};

export const CommandMenu: React.FC = () => {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const {
    darkMode,
    toggleDarkMode,
    startNewRound,
    rounds,
    nodes,
    setSelectedRound,
  } = useDashboardStore();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if ((e.key === 'k' && (e.metaKey || e.ctrlKey))) {
        e.preventDefault();
        setOpen((prev) => !prev);
      } else if (e.key === 'n' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        startNewRound();
        toast.success('Optimistic Action: New round initiated', {
          description: 'Triggered synchronous FedAvg aggregation over encrypted weights.',
        });
      } else if (e.key === 't' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        toggleDarkMode();
      }
    };

    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, [toggleDarkMode, startNewRound]);

  const runAction = (callback: () => void) => {
    setOpen(false);
    transitionView(callback);
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-24 px-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={() => setOpen(false)}
    >
      <div
        className="w-full max-w-xl bg-white dark:bg-zinc-900 rounded-3xl shadow-2xl border border-brand-border dark:border-zinc-800 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <Command className="p-3">
          {/* Search Input Bar */}
          <div className="flex items-center gap-3 px-3 py-2 border-b border-brand-border dark:border-zinc-800">
            <Search size={18} className="text-zinc-400" />
            <Command.Input
              autoFocus
              placeholder="Type a command or search..."
              className="w-full bg-transparent outline-none text-sm text-zinc-900 dark:text-white placeholder:text-zinc-400 font-medium"
            />
            <ShortcutBadge keys={['ESC']} />
          </div>

          {/* List of Filterable Results */}
          <Command.List className="max-h-80 overflow-y-auto p-2 space-y-1">
            <Command.Empty className="p-4 text-center text-xs text-zinc-400 font-medium">
              No matching commands found.
            </Command.Empty>

            {/* Navigation Group */}
            <Command.Group heading={<span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400 px-2 py-1 block">Pages</span>}>
              <Command.Item
                onSelect={() => runAction(() => navigate('/overview'))}
                className="flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold text-zinc-700 dark:text-zinc-200 hover:bg-lavender-100 dark:hover:bg-zinc-800 cursor-pointer"
              >
                <div className="flex items-center gap-2.5">
                  <LayoutDashboard size={15} className="text-lavender-600" />
                  <span>Overview Dashboard</span>
                </div>
                <ShortcutBadge keys={['G', 'O']} />
              </Command.Item>

              <Command.Item
                onSelect={() => runAction(() => navigate('/nodes'))}
                className="flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold text-zinc-700 dark:text-zinc-200 hover:bg-lavender-100 dark:hover:bg-zinc-800 cursor-pointer"
              >
                <div className="flex items-center gap-2.5">
                  <Network size={15} className="text-mint-600" />
                  <span>Hospital Nodes & Quorum</span>
                </div>
                <ShortcutBadge keys={['G', 'N']} />
              </Command.Item>

              <Command.Item
                onSelect={() => runAction(() => navigate('/rounds'))}
                className="flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold text-zinc-700 dark:text-zinc-200 hover:bg-lavender-100 dark:hover:bg-zinc-800 cursor-pointer"
              >
                <div className="flex items-center gap-2.5">
                  <History size={15} className="text-amber-600" />
                  <span>Training Rounds History</span>
                </div>
                <ShortcutBadge keys={['G', 'R']} />
              </Command.Item>

              <Command.Item
                onSelect={() => runAction(() => navigate('/segmentation'))}
                className="flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold text-zinc-700 dark:text-zinc-200 hover:bg-lavender-100 dark:hover:bg-zinc-800 cursor-pointer"
              >
                <div className="flex items-center gap-2.5">
                  <Scan size={15} className="text-cyan-600" />
                  <span>3D MRI Segmentation Gallery</span>
                </div>
                <ShortcutBadge keys={['G', 'S']} />
              </Command.Item>

              <Command.Item
                onSelect={() => runAction(() => navigate('/privacy'))}
                className="flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold text-zinc-700 dark:text-zinc-200 hover:bg-lavender-100 dark:hover:bg-zinc-800 cursor-pointer"
              >
                <div className="flex items-center gap-2.5">
                  <ShieldCheck size={15} className="text-purple-600" />
                  <span>Privacy & Cryptographic Security</span>
                </div>
                <ShortcutBadge keys={['G', 'P']} />
              </Command.Item>
            </Command.Group>

            {/* Quick Actions Group */}
            <Command.Group heading={<span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400 px-2 py-1 block mt-2">Actions</span>}>
              <Command.Item
                onSelect={() =>
                  runAction(() => {
                    startNewRound();
                    toast.success('New Federated Round Started');
                  })
                }
                className="flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold text-zinc-700 dark:text-zinc-200 hover:bg-lavender-100 dark:hover:bg-zinc-800 cursor-pointer"
              >
                <div className="flex items-center gap-2.5">
                  <Plus size={15} className="text-lavender-600" />
                  <span>Start New Round</span>
                </div>
                <ShortcutBadge keys={['N']} />
              </Command.Item>

              <Command.Item
                onSelect={() => runAction(() => toggleDarkMode())}
                className="flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold text-zinc-700 dark:text-zinc-200 hover:bg-lavender-100 dark:hover:bg-zinc-800 cursor-pointer"
              >
                <div className="flex items-center gap-2.5">
                  {darkMode ? <Sun size={15} className="text-amber-500" /> : <Moon size={15} className="text-zinc-500" />}
                  <span>Toggle Theme</span>
                </div>
                <ShortcutBadge keys={['T']} />
              </Command.Item>

              <Command.Item
                onSelect={() =>
                  runAction(() => {
                    const csvContent = 'data:text/csv;charset=utf-8,round,dice,loss\n' + rounds.map(r => `${r.round},${r.dice},${r.global_loss}`).join('\n');
                    const link = document.createElement('a');
                    link.setAttribute('href', encodeURI(csvContent));
                    link.setAttribute('download', 'fedmed_metrics.csv');
                    document.body.appendChild(link);
                    link.click();
                    link.remove();
                    toast.success('Metrics CSV Exported');
                  })
                }
                className="flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold text-zinc-700 dark:text-zinc-200 hover:bg-lavender-100 dark:hover:bg-zinc-800 cursor-pointer"
              >
                <div className="flex items-center gap-2.5">
                  <FileSpreadsheet size={15} className="text-mint-600" />
                  <span>Export Metrics CSV</span>
                </div>
              </Command.Item>
            </Command.Group>

            {/* Jump to Round Group */}
            <Command.Group heading={<span className="text-[10px] font-bold uppercase tracking-wider text-zinc-400 px-2 py-1 block mt-2">Jump to Round</span>}>
              {rounds.slice(-6).map((r) => (
                <Command.Item
                  key={r.round}
                  onSelect={() => runAction(() => setSelectedRound(r.round))}
                  className="flex items-center justify-between px-3 py-2 rounded-xl text-xs font-semibold text-zinc-700 dark:text-zinc-200 hover:bg-lavender-100 dark:hover:bg-zinc-800 cursor-pointer"
                >
                  <span>Filter to Round #{r.round}</span>
                  <span className="font-mono text-mint-600">{(r.dice * 100).toFixed(1)}% Dice</span>
                </Command.Item>
              ))}
            </Command.Group>
          </Command.List>
        </Command>
      </div>
    </div>
  );
};
