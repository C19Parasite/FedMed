import { HospitalNode, NotificationEvent, PrivacyState, RoundMetric, SegmentationCase, WebSocketMessage } from '../types';

export const INITIAL_NODES: HospitalNode[] = [
  {
    id: 'hospital_a',
    name: 'Hospital A (General Scribe)',
    city: 'Boston, MA',
    status: 'training',
    samples: 520,
    lastUpdate: 'Just now',
    ip: '10.240.12.18',
    gradientNorm: 0.84,
    latencyMs: 28,
    contributionPercent: 42,
    isDropped: false,
  },
  {
    id: 'hospital_b',
    name: 'Hospital B (Univ. Health)',
    city: 'Toronto, ON',
    status: 'online',
    samples: 380,
    lastUpdate: '1s ago',
    ip: '10.240.14.92',
    gradientNorm: 0.76,
    latencyMs: 34,
    contributionPercent: 31,
    isDropped: false,
  },
  {
    id: 'hospital_c',
    name: 'Hospital C (Memorial Cancer)',
    city: 'London, UK',
    status: 'online',
    samples: 340,
    lastUpdate: '2s ago',
    ip: '10.240.19.45',
    gradientNorm: 0.69,
    latencyMs: 42,
    contributionPercent: 27,
    isDropped: false,
  },
];

export const INITIAL_ROUNDS: RoundMetric[] = [
  {
    round: 1,
    global_loss: 0.642,
    dice: 0.712,
    baseline_dice: 0.845,
    round_duration_sec: 5.4,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:20:10',
    hospital_metrics: {
      hospital_a: { loss: 0.65, dice: 0.70, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.63, dice: 0.72, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.64, dice: 0.71, samples: 340, contribution: 27 },
    },
    dice_tc: 0.68,
    dice_wt: 0.74,
    dice_et: 0.65,
  },
  {
    round: 2,
    global_loss: 0.584,
    dice: 0.738,
    baseline_dice: 0.845,
    round_duration_sec: 5.2,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:20:45',
    hospital_metrics: {
      hospital_a: { loss: 0.59, dice: 0.73, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.57, dice: 0.74, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.58, dice: 0.73, samples: 340, contribution: 27 },
    },
    dice_tc: 0.71,
    dice_wt: 0.76,
    dice_et: 0.68,
  },
  {
    round: 3,
    global_loss: 0.528,
    dice: 0.764,
    baseline_dice: 0.845,
    round_duration_sec: 5.5,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:21:20',
    hospital_metrics: {
      hospital_a: { loss: 0.53, dice: 0.76, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.52, dice: 0.77, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.53, dice: 0.76, samples: 340, contribution: 27 },
    },
    dice_tc: 0.74,
    dice_wt: 0.79,
    dice_et: 0.71,
  },
  {
    round: 4,
    global_loss: 0.481,
    dice: 0.785,
    baseline_dice: 0.845,
    round_duration_sec: 5.3,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:22:00',
    hospital_metrics: {
      hospital_a: { loss: 0.49, dice: 0.78, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.47, dice: 0.79, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.48, dice: 0.78, samples: 340, contribution: 27 },
    },
    dice_tc: 0.76,
    dice_wt: 0.81,
    dice_et: 0.73,
  },
  {
    round: 5,
    global_loss: 0.442,
    dice: 0.803,
    baseline_dice: 0.845,
    round_duration_sec: 5.6,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:22:40',
    hospital_metrics: {
      hospital_a: { loss: 0.45, dice: 0.80, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.43, dice: 0.81, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.44, dice: 0.80, samples: 340, contribution: 27 },
    },
    dice_tc: 0.78,
    dice_wt: 0.83,
    dice_et: 0.75,
  },
  {
    round: 6,
    global_loss: 0.412,
    dice: 0.819,
    baseline_dice: 0.845,
    round_duration_sec: 5.1,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:23:15',
    hospital_metrics: {
      hospital_a: { loss: 0.42, dice: 0.81, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.40, dice: 0.83, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.41, dice: 0.82, samples: 340, contribution: 27 },
    },
    dice_tc: 0.80,
    dice_wt: 0.85,
    dice_et: 0.77,
  },
  {
    round: 7,
    global_loss: 0.388,
    dice: 0.832,
    baseline_dice: 0.845,
    round_duration_sec: 5.4,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:23:55',
    hospital_metrics: {
      hospital_a: { loss: 0.39, dice: 0.83, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.38, dice: 0.84, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.39, dice: 0.83, samples: 340, contribution: 27 },
    },
    dice_tc: 0.81,
    dice_wt: 0.86,
    dice_et: 0.79,
  },
  {
    round: 8,
    global_loss: 0.365,
    dice: 0.844,
    baseline_dice: 0.845,
    round_duration_sec: 5.3,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:24:35',
    hospital_metrics: {
      hospital_a: { loss: 0.37, dice: 0.84, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.36, dice: 0.85, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.37, dice: 0.84, samples: 340, contribution: 27 },
    },
    dice_tc: 0.83,
    dice_wt: 0.87,
    dice_et: 0.81,
  },
  {
    round: 9,
    global_loss: 0.345,
    dice: 0.853,
    baseline_dice: 0.845,
    round_duration_sec: 5.5,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:25:10',
    hospital_metrics: {
      hospital_a: { loss: 0.35, dice: 0.85, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.34, dice: 0.86, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.35, dice: 0.85, samples: 340, contribution: 27 },
    },
    dice_tc: 0.84,
    dice_wt: 0.88,
    dice_et: 0.82,
  },
  {
    round: 10,
    global_loss: 0.330,
    dice: 0.861,
    baseline_dice: 0.845,
    round_duration_sec: 5.2,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:25:50',
    hospital_metrics: {
      hospital_a: { loss: 0.33, dice: 0.86, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.32, dice: 0.87, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.33, dice: 0.86, samples: 340, contribution: 27 },
    },
    dice_tc: 0.85,
    dice_wt: 0.89,
    dice_et: 0.83,
  },
  {
    round: 11,
    global_loss: 0.318,
    dice: 0.868,
    baseline_dice: 0.845,
    round_duration_sec: 5.7,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:26:30',
    hospital_metrics: {
      hospital_a: { loss: 0.32, dice: 0.86, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.31, dice: 0.87, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.32, dice: 0.87, samples: 340, contribution: 27 },
    },
    dice_tc: 0.85,
    dice_wt: 0.90,
    dice_et: 0.84,
  },
  {
    round: 12,
    global_loss: 0.306,
    dice: 0.874,
    baseline_dice: 0.845,
    round_duration_sec: 5.4,
    participating_clients: 3,
    total_clients: 3,
    timestamp: '14:27:10',
    hospital_metrics: {
      hospital_a: { loss: 0.31, dice: 0.87, samples: 520, contribution: 42 },
      hospital_b: { loss: 0.30, dice: 0.88, samples: 380, contribution: 31 },
      hospital_c: { loss: 0.31, dice: 0.87, samples: 340, contribution: 27 },
    },
    dice_tc: 0.86,
    dice_wt: 0.91,
    dice_et: 0.85,
  },
];

export const INITIAL_PRIVACY: PrivacyState = {
  epsilon: 3.24,
  delta: 1e-5,
  noise_sigma: 1.1,
  clip_norm: 1.0,
  total_budget: 10.0,
  homomorphic_encryption: {
    enabled: true,
    status: 'Aggregating on ciphertext',
    scheme: 'TenSEAL CKKS',
    ciphertext_size_kb: 647.2,
    poly_modulus_degree: 8192,
  },
};

export const INITIAL_CASES: SegmentationCase[] = [
  {
    id: 'case_0132',
    title: 'BraTS Case 0132',
    patientId: 'PAT-4029-BR',
    sliceZ: 16,
    depthSlices: 32,
    diceWT: 0.91,
    diceTC: 0.86,
    diceET: 0.83,
    diceMean: 0.872,
    baselineMean: 0.841,
    hospitalSource: 'Hospital A (General Scribe)',
    modality: 'T1ce + FLAIR',
  },
  {
    id: 'case_0245',
    title: 'BraTS Case 0245',
    patientId: 'PAT-8172-GL',
    sliceZ: 18,
    depthSlices: 32,
    diceWT: 0.89,
    diceTC: 0.84,
    diceET: 0.81,
    diceMean: 0.851,
    baselineMean: 0.832,
    hospitalSource: 'Hospital B (Univ. Health)',
    modality: 'T2 + FLAIR',
  },
  {
    id: 'case_0389',
    title: 'BraTS Case 0389',
    patientId: 'PAT-1934-AS',
    sliceZ: 14,
    depthSlices: 32,
    diceWT: 0.93,
    diceTC: 0.89,
    diceET: 0.87,
    diceMean: 0.898,
    baselineMean: 0.855,
    hospitalSource: 'Hospital C (Memorial Cancer)',
    modality: 'T1 + T1ce',
  },
  {
    id: 'case_0412',
    title: 'BraTS Case 0412',
    patientId: 'PAT-6612-OD',
    sliceZ: 20,
    depthSlices: 32,
    diceWT: 0.88,
    diceTC: 0.82,
    diceET: 0.79,
    diceMean: 0.835,
    baselineMean: 0.812,
    hospitalSource: 'Hospital A (General Scribe)',
    modality: 'FLAIR Multi-contrast',
  },
  {
    id: 'case_0567',
    title: 'BraTS Case 0567',
    patientId: 'PAT-9081-MG',
    sliceZ: 15,
    depthSlices: 32,
    diceWT: 0.92,
    diceTC: 0.87,
    diceET: 0.85,
    diceMean: 0.884,
    baselineMean: 0.849,
    hospitalSource: 'Hospital B (Univ. Health)',
    modality: 'T1ce Post-Gadolinium',
  },
];

export const INITIAL_NOTIFICATIONS: NotificationEvent[] = [
  {
    id: 'notif_1',
    timestamp: 'Just now',
    type: 'overview',
    level: 'success',
    title: 'Round 12 Aggregated Successfully',
    message: 'Global mean Dice reached 0.874 (+0.029 vs centralized baseline).',
  },
  {
    id: 'notif_2',
    timestamp: '2m ago',
    type: 'nodes',
    level: 'warn',
    title: 'Hospital B Resilience Triggered',
    message: 'Hospital B dropped offline mid-epoch; round continued with 2 nodes & renormalized weights.',
  },
  {
    id: 'notif_3',
    timestamp: '5m ago',
    type: 'security',
    level: 'info',
    title: 'TenSEAL CKKS Ciphertext Sum',
    message: 'Encrypted updates summed homomorphically on central server (647 KB payload).',
  },
  {
    id: 'notif_4',
    timestamp: '8m ago',
    type: 'security',
    level: 'info',
    title: 'mTLS Handshake Verified',
    message: 'Hospital C mutual TLS handshake validated with root CA certs.',
  },
  {
    id: 'notif_5',
    timestamp: '12m ago',
    type: 'security',
    level: 'info',
    title: 'Differential Privacy Noise Applied',
    message: 'L2 norm clipped to 1.0, Gaussian noise added with σ = 1.1 (ε = 3.24).',
  },
];

/**
 * Mock stream generator that periodically produces events to keep the UI feeling live.
 */
export class MockTrainingStream {
  private timer: number | null = null;
  private currentRound = 12;
  private currentEpsilon = 3.24;
  private isHospitalBDropped = false;
  private onMessage: (msg: WebSocketMessage) => void;

  constructor(onMessage: (msg: WebSocketMessage) => void) {
    this.onMessage = onMessage;
  }

  public start() {
    if (this.timer) return;

    this.timer = window.setInterval(() => {
      this.tick();
    }, 1500);
  }

  public stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  private tick() {
    const roll = Math.random();

    // 1. Periodic simulated round advancement (every ~8-12 ticks)
    if (roll < 0.25 && this.currentRound < 30) {
      this.currentRound += 1;
      const baseLoss = Math.max(0.20, 0.306 - (this.currentRound - 12) * 0.012 + (Math.random() * 0.006 - 0.003));
      const baseDice = Math.min(0.92, 0.874 + (this.currentRound - 12) * 0.004 + (Math.random() * 0.003 - 0.001));

      this.currentEpsilon += 0.18;

      this.onMessage({
        type: 'metrics',
        round: this.currentRound,
        global_loss: parseFloat(baseLoss.toFixed(4)),
        dice: parseFloat(baseDice.toFixed(4)),
        baseline_dice: 0.845,
      });

      this.onMessage({
        type: 'privacy',
        epsilon: parseFloat(this.currentEpsilon.toFixed(2)),
        delta: 1e-5,
        noise_sigma: 1.1,
      });

      this.onMessage({
        type: 'event',
        level: 'success',
        title: `Round ${this.currentRound} Complete`,
        message: `Global Dice: ${(baseDice * 100).toFixed(1)}% | Aggregated on ciphertext`,
        category: 'overview',
      });
    }

    // 2. Node status toggle (simulate hospital resilience test)
    else if (roll < 0.45) {
      if (!this.isHospitalBDropped && Math.random() < 0.3) {
        this.isHospitalBDropped = true;
        this.onMessage({
          type: 'node_status',
          node: 'hospital_b',
          status: 'offline',
          samples: 380,
        });
        this.onMessage({
          type: 'event',
          level: 'warn',
          title: 'Hospital B Node Offline',
          message: 'Hospital B dropped offline; server proceeding with 2/3 quorum.',
          category: 'nodes',
        });
      } else if (this.isHospitalBDropped) {
        this.isHospitalBDropped = false;
        this.onMessage({
          type: 'node_status',
          node: 'hospital_b',
          status: 'online',
          samples: 380,
        });
        this.onMessage({
          type: 'event',
          level: 'success',
          title: 'Hospital B Reconnected',
          message: 'Hospital B successfully rejoined federation and synchronized weights.',
          category: 'nodes',
        });
      }
    }

    // 3. Minor gradient / privacy telemetry tick
    else {
      this.onMessage({
        type: 'event',
        level: 'info',
        title: 'Local Step Synchronized',
        message: `Hospital A step gradient norm: ${(0.82 + Math.random() * 0.05).toFixed(3)}`,
        category: 'security',
      });
    }
  }
}
