import { create } from 'zustand';
import { HospitalNode, NotificationEvent, PrivacyState, RoundMetric, SegmentationCase, WebSocketMessage } from '../types';
import { INITIAL_CASES, INITIAL_NODES, INITIAL_NOTIFICATIONS, INITIAL_PRIVACY, INITIAL_ROUNDS } from '../lib/mockStream';

interface DashboardStore {
  // Data
  nodes: HospitalNode[];
  rounds: RoundMetric[];
  privacy: PrivacyState;
  cases: SegmentationCase[];
  notifications: NotificationEvent[];

  // UI state
  selectedRound: number | null;
  viewMode: 'round' | 'epoch';
  notificationTab: 'overview' | 'nodes' | 'security';
  activeModalCase: SegmentationCase | null;
  isModalOpen: boolean;
  is2FAModalOpen: boolean;
  connectionStatus: 'connected' | 'reconnecting' | 'mock' | 'disconnected';
  isMockMode: boolean;
  darkMode: boolean;
  announcement: string;

  // Actions
  handleWebSocketMessage: (msg: WebSocketMessage) => void;
  setSelectedRound: (round: number | null) => void;
  setViewMode: (mode: 'round' | 'epoch') => void;
  setNotificationTab: (tab: 'overview' | 'nodes' | 'security') => void;
  openModal: (caseItem: SegmentationCase) => void;
  closeModal: () => void;
  open2FAModal: () => void;
  close2FAModal: () => void;
  clearNotifications: () => void;
  restoreNotifications: (items: NotificationEvent[]) => void;
  toggleNodeStatus: (nodeId: string) => void;
  toggleMockMode: () => void;
  setConnectionStatus: (status: 'connected' | 'reconnecting' | 'mock' | 'disconnected') => void;
  toggleDarkMode: () => void;
  startNewRound: () => void;
  setAnnouncement: (text: string) => void;
}

export const useDashboardStore = create<DashboardStore>((set, get) => ({
  nodes: INITIAL_NODES,
  rounds: INITIAL_ROUNDS,
  privacy: INITIAL_PRIVACY,
  cases: INITIAL_CASES,
  notifications: INITIAL_NOTIFICATIONS,

  selectedRound: null,
  viewMode: 'round',
  notificationTab: 'overview',
  activeModalCase: null,
  isModalOpen: false,
  is2FAModalOpen: false,
  connectionStatus: 'mock',
  isMockMode: true,
  darkMode: false,
  announcement: '',

  handleWebSocketMessage: (msg: WebSocketMessage) => {
    if (msg.type === 'metrics') {
      const { round, global_loss, dice, baseline_dice } = msg;
      set((state) => {
        const existingIdx = state.rounds.findIndex((r) => r.round === round);
        const newEntry: RoundMetric = {
          round,
          global_loss,
          dice,
          baseline_dice: baseline_dice ?? 0.845,
          round_duration_sec: 5.4,
          participating_clients: state.nodes.filter((n) => n.status !== 'offline').length,
          total_clients: state.nodes.length,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          hospital_metrics: {
            hospital_a: { loss: global_loss + 0.01, dice: dice - 0.01, samples: 520, contribution: 42 },
            hospital_b: { loss: global_loss - 0.01, dice: dice + 0.01, samples: 380, contribution: 31 },
            hospital_c: { loss: global_loss, dice: dice, samples: 340, contribution: 27 },
          },
          dice_tc: Math.min(0.95, dice - 0.02),
          dice_wt: Math.min(0.97, dice + 0.03),
          dice_et: Math.min(0.93, dice - 0.03),
        };

        const updatedRounds = existingIdx >= 0
          ? state.rounds.map((r, i) => (i === existingIdx ? newEntry : r))
          : [...state.rounds, newEntry];

        return {
          rounds: updatedRounds,
          announcement: `Round ${round} completed: Dice ${(dice * 100).toFixed(1)}%, Loss ${global_loss.toFixed(3)}.`,
        };
      });
    } else if (msg.type === 'node_status') {
      const { node, status, samples } = msg;
      set((state) => ({
        nodes: state.nodes.map((n) => {
          if (n.id.toLowerCase() === node.toLowerCase() || n.id.includes(node.toLowerCase())) {
            return {
              ...n,
              status,
              samples: samples ?? n.samples,
              isDropped: status === 'offline',
              lastUpdate: 'Just now',
            };
          }
          return n;
        }),
        announcement: `Node ${node} status changed to ${status}. Quorum maintained.`,
      }));
    } else if (msg.type === 'privacy') {
      const { epsilon, delta, noise_sigma } = msg;
      set((state) => ({
        privacy: {
          ...state.privacy,
          epsilon,
          delta,
          noise_sigma,
        },
        announcement: `Privacy budget updated: epsilon ${epsilon.toFixed(2)}, delta 1e-5.`,
      }));
    } else if (msg.type === 'event') {
      const newNotif: NotificationEvent = {
        id: `event_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
        timestamp: 'Just now',
        type: msg.category ?? 'overview',
        level: msg.level,
        title: msg.title ?? (msg.level === 'warn' ? 'Resilience Alert' : 'System Event'),
        message: msg.message,
      };

      set((state) => ({
        notifications: [newNotif, ...state.notifications.slice(0, 19)],
        announcement: `Alert: ${newNotif.title} - ${newNotif.message}`,
      }));
    }
  },

  setSelectedRound: (round) => set({ selectedRound: round }),
  setViewMode: (mode) => set({ viewMode: mode }),
  setNotificationTab: (tab) => set({ notificationTab: tab }),
  openModal: (caseItem) => set({ activeModalCase: caseItem, isModalOpen: true }),
  closeModal: () => set({ activeModalCase: null, isModalOpen: false }),
  open2FAModal: () => set({ is2FAModalOpen: true }),
  close2FAModal: () => set({ is2FAModalOpen: false }),

  clearNotifications: () => set({ notifications: [] }),
  restoreNotifications: (items) => set({ notifications: items }),

  toggleNodeStatus: (nodeId) => {
    set((state) => ({
      nodes: state.nodes.map((n) => {
        if (n.id === nodeId) {
          const nextStatus = n.status === 'offline' ? 'online' : 'offline';
          return {
            ...n,
            status: nextStatus,
            isDropped: nextStatus === 'offline',
            lastUpdate: 'Just now',
          };
        }
        return n;
      }),
    }));
  },

  setAnnouncement: (text) => set({ announcement: text }),

  toggleMockMode: () => {
    set((state) => ({
      isMockMode: !state.isMockMode,
      connectionStatus: !state.isMockMode ? 'mock' : 'disconnected',
    }));
  },

  setConnectionStatus: (status) => set({ connectionStatus: status }),

  toggleDarkMode: () => {
    set((state) => {
      const nextDark = !state.darkMode;
      if (nextDark) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
      return { darkMode: nextDark };
    });
  },

  startNewRound: () => {
    const { rounds, nodes } = get();
    const nextRoundNum = rounds.length + 1;
    const lastRound = rounds[rounds.length - 1];
    const newLoss = Math.max(0.18, lastRound.global_loss - 0.015);
    const newDice = Math.min(0.93, lastRound.dice + 0.005);

    get().handleWebSocketMessage({
      type: 'metrics',
      round: nextRoundNum,
      global_loss: parseFloat(newLoss.toFixed(4)),
      dice: parseFloat(newDice.toFixed(4)),
      baseline_dice: 0.845,
    });

    get().handleWebSocketMessage({
      type: 'event',
      level: 'success',
      title: `Round ${nextRoundNum} Initiated`,
      message: `Triggered synchronous FedAvg across ${nodes.length} silos on encrypted updates.`,
      category: 'overview',
    });
  },
}));
