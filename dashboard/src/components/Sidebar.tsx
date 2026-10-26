import React, { useState, useRef, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard,
  Network,
  History,
  Scan,
  ShieldCheck,
  Moon,
  Sun,
  Keyboard,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';
import { transitionView } from '../lib/viewTransitions';
import { ShortcutBadge } from './CommandMenu';

export const Sidebar: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { darkMode, toggleDarkMode, nodes } = useDashboardStore();

  const [isNodesHovered, setIsNodesHovered] = useState(false);
  const hoverTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Automatically close flyout whenever route changes
  useEffect(() => {
    setIsNodesHovered(false);
  }, [location.pathname]);

  const handleNodesMouseEnter = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
    setIsNodesHovered(true);
  };

  const handleNodesMouseLeave = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    hoverTimeoutRef.current = setTimeout(() => {
      setIsNodesHovered(false);
    }, 180);
  };

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
    setIsNodesHovered(false);
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
            className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-lavender-600 to-lavender-400 flex items-center justify-center shadow-glow-lavender text-white hover:scale-105 active:scale-95 transition-transform hit-slop cursor-pointer"
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
                onMouseEnter={item.hasFlyout ? handleNodesMouseEnter : undefined}
                onMouseLeave={item.hasFlyout ? handleNodesMouseLeave : undefined}
              >
                <button
                  onClick={() => handleNavClick(item.path)}
                  className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all duration-200 relative hit-slop cursor-pointer ${
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

                {/* Standard Tooltip for items without flyout */}
                {!item.hasFlyout && (
                  <div className="absolute left-full ml-3 px-3 py-1.5 bg-zinc-900 text-white text-xs font-semibold rounded-xl shadow-xl opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-150 whitespace-nowrap z-50 flex items-center gap-2">
                    <span>{item.label}</span>
                    <ShortcutBadge keys={item.shortcut} />
                  </div>
                )}

                {/* Live Silo Quorum Flyout: ONLY shows on hover over Hospital Nodes tab */}
                {item.hasFlyout && (
                  <AnimatePresence>
                    {isNodesHovered && (
                      <motion.div
                        initial={{ opacity: 0, x: -8, scale: 0.96 }}
                        animate={{ opacity: 1, x: 0, scale: 1 }}
                        exit={{ opacity: 0, x: -6, scale: 0.96 }}
                        transition={{ duration: 0.15 }}
                        onMouseEnter={handleNodesMouseEnter}
                        onMouseLeave={handleNodesMouseLeave}
                        className="absolute left-full -top-4 ml-3 w-72 bg-white dark:bg-zinc-900 rounded-3xl p-4 shadow-2xl border border-brand-border dark:border-zinc-800 z-50 text-zinc-900 dark:text-white"
                      >
                        {/* Invisible hover bridge to eliminate cursor gap drop */}
                        <div className="absolute -left-3 top-0 w-3 h-full" aria-hidden="true" />

                        <div className="flex items-center justify-between pb-2 mb-2 border-b border-brand-border/60 dark:border-zinc-800">
                          <div className="flex items-center gap-1.5">
                            <span className="text-xs font-bold text-zinc-400 uppercase tracking-wider">
                              Live Silo Quorum
                            </span>
                            <ShortcutBadge keys={item.shortcut} />
                          </div>
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-mint-100 text-mint-700 dark:bg-mint-900/40 dark:text-mint-300">
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
                      </motion.div>
                    )}
                  </AnimatePresence>
                )}
              </div>
            );
          })}
        </nav>

        {/* Bottom Actions */}
        <div className="flex flex-col items-center gap-3 pt-4 border-t border-zinc-800/80 w-full">
          {/* Shortcuts Cheat Sheet Trigger */}
          <button
            onClick={() => {
              window.dispatchEvent(new KeyboardEvent('keydown', { key: '?' }));
            }}
            className="w-10 h-10 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800/60 flex items-center justify-center transition-colors hit-slop cursor-pointer"
            title="Keyboard Shortcuts (?)"
            aria-label="Keyboard Shortcuts"
          >
            <Keyboard size={18} />
          </button>

          {/* Dark Mode Toggle */}
          <button
            onClick={toggleDarkMode}
            className="w-10 h-10 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800/60 flex items-center justify-center transition-colors hit-slop cursor-pointer"
            title={`Toggle Theme (T)`}
            aria-label="Toggle Theme"
          >
            {darkMode ? <Sun size={18} className="text-amber-400" /> : <Moon size={18} />}
          </button>
        </div>
      </aside>

      {/* Mobile Persistent Bottom Tab Bar with Safe-Area Insets */}
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
              className={`p-2.5 rounded-2xl flex flex-col items-center justify-center gap-1 hit-slop transition-all cursor-pointer ${
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
