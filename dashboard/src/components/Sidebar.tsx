import React from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard,
  Network,
  History,
  Scan,
  ShieldCheck,
  Settings,
  Moon,
  Sun,
  Keyboard,
  Hospital,
  WifiOff,
  CheckCircle2,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';
import { transitionView } from '../lib/viewTransitions';
import { ShortcutBadge, isMac } from './CommandMenu';
import { useSafeTriangle } from '../hooks/useSafeTriangle';

export const Sidebar: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { darkMode, toggleDarkMode, nodes } = useDashboardStore();

  // 20. Amazon safe triangle hover intent for nodes flyout menu
  const {
    isOpen: isFlyoutOpen,
    triggerRef,
    submenuRef,
    onTriggerMouseEnter,
    onTriggerMouseLeave,
    onSubmenuMouseEnter,
    onSubmenuMouseLeave,
  } = useSafeTriangle(280);

  const navItems = [
    { path: '/overview', icon: LayoutDashboard, label: 'Overview', shortcut: ['G', 'O'] },
    {
      path: '/nodes',
      icon: Network,
      label: 'Hospital Nodes',
      shortcut: ['G', 'N'],
      hasFlyout: true,
    },
    { path: '/rounds', icon: History, label: 'Training Rounds', shortcut: ['G', 'R'] },
    { path: '/segmentation', icon: Scan, label: 'Segmentation Viewer', shortcut: ['G', 'S'] },
    { path: '/privacy', icon: ShieldCheck, label: 'Privacy & Security', shortcut: ['G', 'P'] },
  ];

  const handleNavClick = (path: string) => {
    transitionView(() => navigate(path));
  };

  return (
    <>
      {/* Desktop Persistent Fixed / Sticky Sidebar (Always in same location while main view scrolls) */}
      <aside
        className="hidden md:flex sticky top-6 h-[calc(100dvh-3rem)] w-18 lg:w-20 bg-surface-dark dark:bg-surface-darker text-white rounded-squircle py-6 px-2 flex-col items-center justify-between shadow-sidebar z-30 shrink-0 self-start select-none"
        aria-label="Desktop Sidebar Navigation"
      >
        {/* Brand Logo Mark */}
        <div className="flex flex-col items-center gap-2">
          <button
            onClick={() => handleNavClick('/overview')}
            className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-lavender-600 to-lavender-400 flex items-center justify-center shadow-glow-lavender text-white hover:scale-105 active:scale-95 transition-transform hit-slop"
            aria-label="FedMed Home"
          >
            <div className="relative flex items-center justify-center">
              <span className="text-xl select-none">🧠</span>
              <span className="absolute -bottom-1 -right-1 w-2.5 h-2.5 bg-mint-400 border-2 border-surface-dark rounded-full" />
            </div>
          </button>
          <span className="text-[9px] font-extrabold tracking-wider text-lavender-300/80 uppercase">
            FedMed
          </span>
        </div>

        {/* Navigation Items */}
        <nav className="flex flex-col items-center gap-3.5 my-auto">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive =
              location.pathname === item.path ||
              (item.path === '/overview' && location.pathname === '/');

            return (
              <div
                key={item.path}
                className="relative group"
                ref={item.hasFlyout ? (triggerRef as any) : undefined}
                onMouseEnter={item.hasFlyout ? onTriggerMouseEnter : undefined}
                onMouseLeave={item.hasFlyout ? onTriggerMouseLeave : undefined}
              >
                <button
                  onClick={() => handleNavClick(item.path)}
                  className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all duration-200 relative hit-slop ${
                    isActive
                      ? 'bg-lavender-600 text-white shadow-glow-lavender'
                      : 'text-zinc-400 hover:text-white hover:bg-zinc-800/70'
                  }`}
                  aria-label={item.label}
                >
                  <Icon size={20} strokeWidth={isActive ? 2.5 : 2} />
                  {isActive && (
                    <motion.div
                      layoutId="sidebarActivePill"
                      className="absolute inset-0 rounded-2xl bg-lavender-600 -z-10"
                      transition={{ type: 'spring', stiffness: 380, damping: 28 }}
                    />
                  )}
                </button>

                {/* Tooltip with Keyboard Shortcut Badge */}
                {!item.hasFlyout && (
                  <div className="absolute left-full ml-3 px-3 py-1.5 bg-zinc-900 text-white text-xs font-semibold rounded-xl shadow-xl opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-150 whitespace-nowrap z-50 flex items-center gap-2">
                    <span>{item.label}</span>
                    <ShortcutBadge keys={item.shortcut} />
                  </div>
                )}
              </div>
            );
          })}
        </nav>

        {/* 20. Amazon Safe-Triangle Flyout Submenu for Hospital Nodes */}
        <AnimatePresence>
          {isFlyoutOpen && (
            <div
              ref={submenuRef as any}
              onMouseEnter={onSubmenuMouseEnter}
              onMouseLeave={onSubmenuMouseLeave}
              className="absolute left-full top-36 ml-4 w-72 bg-white dark:bg-zinc-900 rounded-3xl p-4 shadow-2xl border border-brand-border dark:border-zinc-800 z-50 animate-in fade-in zoom-in-95 duration-150 text-zinc-900 dark:text-white"
            >
              <div className="flex items-center justify-between pb-2 mb-2 border-b border-brand-border/60 dark:border-zinc-800">
                <span className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
                  Live Silo Quorum
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-mint-100 text-mint-700">
                  {nodes.filter((n) => n.status !== 'offline').length}/{nodes.length} Online
                </span>
              </div>

              <div className="space-y-2">
                {nodes.map((node) => {
                  const isOffline = node.status === 'offline';
                  return (
                    <div
                      key={node.id}
                      onClick={() => handleNavClick('/nodes')}
                      className="p-2 rounded-xl hover:bg-surface-soft dark:hover:bg-zinc-800 cursor-pointer flex items-center justify-between text-xs transition-colors"
                    >
                      <div className="flex items-center gap-2 truncate">
                        <div className="w-6 h-6 rounded-lg bg-lavender-100 dark:bg-lavender-900/40 text-lavender-700 dark:text-lavender-300 flex items-center justify-center font-bold text-[10px]">
                          {node.name.slice(0, 2).toUpperCase()}
                        </div>
                        <span className="font-semibold truncate">{node.name}</span>
                      </div>
                      {isOffline ? (
                        <span className="text-[10px] font-bold text-coral-500">Dropped</span>
                      ) : (
                        <span className="text-[10px] font-bold text-mint-600">{node.samples} MRIs</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </AnimatePresence>

        {/* Bottom Actions */}
        <div className="flex flex-col items-center gap-3 pt-4 border-t border-zinc-800/80 w-full">
          {/* Shortcuts Cheat Sheet Trigger */}
          <button
            onClick={() => {
              window.dispatchEvent(new KeyboardEvent('keydown', { key: '?' }));
            }}
            className="w-10 h-10 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800/60 flex items-center justify-center transition-colors hit-slop"
            title="Keyboard Shortcuts (?)"
            aria-label="Keyboard Shortcuts"
          >
            <Keyboard size={18} />
          </button>

          {/* Dark Mode Toggle */}
          <button
            onClick={toggleDarkMode}
            className="w-10 h-10 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800/60 flex items-center justify-center transition-colors hit-slop"
            title={`Toggle Theme (T)`}
            aria-label="Toggle Theme"
          >
            {darkMode ? <Sun size={18} className="text-amber-400" /> : <Moon size={18} />}
          </button>
        </div>
      </aside>

      {/* 14. Mobile Persistent Bottom Tab Bar with Safe-Area Insets */}
      <nav
        className="md:hidden fixed bottom-0 left-0 right-0 z-40 bg-surface-dark dark:bg-surface-darker text-white border-t border-zinc-800/80 px-4 py-2 flex items-center justify-around pb-[env(safe-area-inset-bottom)] shadow-2xl backdrop-blur-md"
        aria-label="Mobile Bottom Navigation"
      >
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive =
            location.pathname === item.path ||
            (item.path === '/overview' && location.pathname === '/');

          return (
            <button
              key={item.path}
              onClick={() => handleNavClick(item.path)}
              className={`p-2.5 rounded-2xl flex flex-col items-center justify-center gap-1 hit-slop transition-all ${
                isActive ? 'text-lavender-400' : 'text-zinc-400 hover:text-white'
              }`}
              aria-label={item.label}
            >
              <Icon size={20} strokeWidth={isActive ? 2.5 : 2} />
              <span className="text-[10px] font-bold">{item.label.split(' ')[0]}</span>
            </button>
          );
        })}
      </nav>
    </>
  );
};
