import React, { useEffect, useState } from 'react';
import { X, Command, Keyboard } from 'lucide-react';
import { ShortcutBadge, isMac } from './CommandMenu';

export const KeyboardShortcutsModal: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '?' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      } else if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  if (!isOpen) return null;

  const shortcuts = [
    { section: 'Global Navigation', items: [
      { keys: ['meta', 'K'], label: 'Open Command Palette' },
      { keys: ['?'], label: 'Open Keyboard Shortcuts Cheat-sheet' },
      { keys: ['T'], label: 'Toggle Dark / Light Theme' },
      { keys: ['ESC'], label: 'Close Active Modal / Palette / Sheet' },
    ]},
    { section: 'Federation Actions', items: [
      { keys: ['N'], label: 'Start New Federated Round' },
      { keys: ['2', 'F', 'A'], label: 'Open Node Access Verification (2FA)' },
      { keys: ['R'], label: 'Reset Round Filter to Live Feed' },
    ]},
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200"
      onClick={() => setIsOpen(false)}
    >
      <div
        className="w-full max-w-lg bg-white dark:bg-zinc-900 rounded-3xl shadow-2xl border border-brand-border dark:border-zinc-800 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-6 py-4 border-b border-brand-border dark:border-zinc-800 bg-surface-soft dark:bg-zinc-800/40">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-lavender-100 text-lavender-700 dark:bg-lavender-900/40 dark:text-lavender-300 flex items-center justify-center font-bold">
              <Keyboard size={16} />
            </div>
            <h3 className="text-base font-bold text-zinc-900 dark:text-white">
              Keyboard Shortcuts
            </h3>
          </div>
          <button
            onClick={() => setIsOpen(false)}
            className="w-8 h-8 rounded-full hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-400 hover:text-zinc-700 dark:hover:text-white flex items-center justify-center transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {shortcuts.map((group) => (
            <div key={group.section}>
              <h4 className="text-[11px] font-bold uppercase tracking-wider text-zinc-400 mb-3">
                {group.section}
              </h4>
              <div className="space-y-2.5">
                {group.items.map((item) => (
                  <div
                    key={item.label}
                    className="flex items-center justify-between py-1.5 border-b border-brand-border/40 dark:border-zinc-800/50 text-xs"
                  >
                    <span className="font-medium text-zinc-700 dark:text-zinc-200">{item.label}</span>
                    <ShortcutBadge keys={item.keys} />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
