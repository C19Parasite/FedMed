import React from 'react';
import { Bell, CheckCircle2, AlertTriangle, ShieldCheck, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';
import { NotificationEvent } from '../types';

export const NotificationFeed: React.FC = () => {
  const { notifications, notificationTab, setNotificationTab } = useDashboardStore();

  const filteredNotifications = notifications.filter(
    (n) => notificationTab === 'overview' || n.type === notificationTab
  );

  return (
    <div className="card-base p-5 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800 flex flex-col justify-between">
      {/* Header with Title and Tabs */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-lavender-100 dark:bg-lavender-900/40 text-lavender-700 dark:text-lavender-300 flex items-center justify-center">
              <Bell size={14} />
            </div>
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">
              Live Notifications
            </h3>
          </div>
          <span className="w-2 h-2 rounded-full bg-mint-500 animate-pulse" />
        </div>

        {/* Category Tabs (Overview | Nodes | Security) */}
        <div className="flex p-1 bg-surface-soft dark:bg-zinc-800 rounded-xl border border-brand-border/60 mb-3">
          {(['overview', 'nodes', 'security'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setNotificationTab(tab)}
              className={`flex-1 py-1 rounded-lg text-xs font-bold capitalize transition-all ${
                notificationTab === tab
                  ? 'bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white shadow-sm'
                  : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Animated Feed Items List */}
      <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
        <AnimatePresence initial={false}>
          {filteredNotifications.map((notif) => (
            <NotificationItem key={notif.id} item={notif} />
          ))}
        </AnimatePresence>

        {filteredNotifications.length === 0 && (
          <div className="py-8 text-center text-xs text-zinc-400 font-medium">
            No active alerts in this category
          </div>
        )}
      </div>
    </div>
  );
};

interface NotificationItemProps {
  item: NotificationEvent;
}

const NotificationItem: React.FC<NotificationItemProps> = ({ item }) => {
  const getIcon = () => {
    switch (item.level) {
      case 'success':
        return <CheckCircle2 size={15} className="text-mint-600" />;
      case 'warn':
        return <AlertTriangle size={15} className="text-coral-500" />;
      case 'info':
      default:
        return <ShieldCheck size={15} className="text-lavender-600" />;
    }
  };

  const getBg = () => {
    switch (item.level) {
      case 'success':
        return 'bg-mint-50 dark:bg-mint-950/30 border-mint-100 dark:border-mint-900/40';
      case 'warn':
        return 'bg-coral-50 dark:bg-coral-950/30 border-coral-200 dark:border-coral-900/40';
      case 'info':
      default:
        return 'bg-surface-soft dark:bg-zinc-800/50 border-brand-border/60 dark:border-zinc-700/50';
    }
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.25 }}
      className={`p-3 rounded-2xl border ${getBg()} transition-all`}
    >
      <div className="flex items-start gap-2.5">
        <div className="shrink-0 mt-0.5">{getIcon()}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <h4 className="text-xs font-bold text-zinc-900 dark:text-white truncate">
              {item.title}
            </h4>
            <span className="text-[10px] text-zinc-400 font-medium shrink-0 font-mono">
              {item.timestamp}
            </span>
          </div>
          <p className="text-[11px] text-zinc-600 dark:text-zinc-300 font-medium leading-relaxed mt-0.5">
            {item.message}
          </p>
        </div>
      </div>
    </motion.div>
  );
};
