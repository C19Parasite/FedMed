import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Network,
  History,
  Scan,
  ShieldCheck,
  Settings,
  LogOut,
  Moon,
  Sun,
  Activity,
} from 'lucide-react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';

export const Sidebar: React.FC = () => {
  const location = useLocation();
  const { darkMode, toggleDarkMode } = useDashboardStore();

  const navItems = [
    { path: '/overview', icon: LayoutDashboard, label: 'Overview' },
    { path: '/nodes', icon: Network, label: 'Hospital Nodes' },
    { path: '/rounds', icon: History, label: 'Training Rounds' },
    { path: '/segmentation', icon: Scan, label: 'Segmentation Viewer' },
    { path: '/privacy', icon: ShieldCheck, label: 'Privacy & Security' },
  ];

  return (
    <aside
      className="w-16 md:w-20 bg-surface-dark dark:bg-surface-darker text-white rounded-3xl md:rounded-[36px] py-6 px-2 flex flex-col items-center justify-between shadow-sidebar z-30 transition-all duration-300"
      aria-label="Sidebar Navigation"
    >
      {/* Brand Logo Mark */}
      <div className="flex flex-col items-center gap-3">
        <NavLink
          to="/overview"
          className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-lavender-600 to-lavender-400 flex items-center justify-center shadow-glow-lavender text-white hover:scale-105 transition-transform"
          aria-label="FedMed Home"
        >
          <div className="relative flex items-center justify-center">
            <span className="text-xl select-none">🧠</span>
            <span className="absolute -bottom-1 -right-1 w-2.5 h-2.5 bg-mint-400 border-2 border-surface-dark rounded-full" />
          </div>
        </NavLink>
        <span className="text-[10px] font-bold tracking-wider text-lavender-300/80 uppercase">
          FedMed
        </span>
      </div>

      {/* Main Nav Items */}
      <nav className="flex flex-col items-center gap-4 my-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path || (item.path === '/overview' && location.pathname === '/');

          return (
            <div key={item.path} className="relative group">
              <NavLink
                to={item.path}
                className={({ isActive: navActive }) => {
                  const active = navActive || (item.path === '/overview' && location.pathname === '/');
                  return `w-12 h-12 rounded-2xl flex items-center justify-center transition-all duration-200 relative ${
                    active
                      ? 'bg-lavender-600 text-white shadow-glow-lavender'
                      : 'text-zinc-400 hover:text-white hover:bg-zinc-800/60'
                  }`;
                }}
                aria-label={item.label}
              >
                <Icon size={20} strokeWidth={isActive ? 2.4 : 2} />
                {isActive && (
                  <motion.div
                    layoutId="activeIndicator"
                    className="absolute inset-0 rounded-2xl bg-lavender-600 -z-10"
                    transition={{ type: 'spring', stiffness: 350, damping: 30 }}
                  />
                )}
              </NavLink>

              {/* Tooltip on hover */}
              <div className="absolute left-full ml-3 px-3 py-1.5 bg-zinc-900 text-white text-xs font-medium rounded-lg shadow-xl opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-150 whitespace-nowrap z-50">
                {item.label}
              </div>
            </div>
          );
        })}
      </nav>

      {/* Bottom Actions */}
      <div className="flex flex-col items-center gap-3 pt-4 border-t border-zinc-800/80 w-full">
        {/* Dark mode toggle */}
        <button
          onClick={toggleDarkMode}
          className="w-10 h-10 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800/60 flex items-center justify-center transition-colors"
          aria-label="Toggle Dark Mode"
          title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {darkMode ? <Sun size={18} className="text-amber-400" /> : <Moon size={18} />}
        </button>

        {/* Settings button */}
        <button
          className="w-10 h-10 rounded-xl text-zinc-400 hover:text-white hover:bg-zinc-800/60 flex items-center justify-center transition-colors"
          aria-label="Settings"
          title="Settings"
        >
          <Settings size={18} />
        </button>

        {/* Logout / Disconnect indicator */}
        <button
          className="w-10 h-10 rounded-xl text-zinc-400 hover:text-coral-400 hover:bg-zinc-800/60 flex items-center justify-center transition-colors"
          aria-label="Exit Console"
          title="Disconnect Console"
        >
          <LogOut size={18} />
        </button>
      </div>
    </aside>
  );
};
