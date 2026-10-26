export type NodeStatus = 'online' | 'training' | 'offline';

export interface HospitalNode {
  id: string;
  name: string;
  city: string;
  status: NodeStatus;
  samples: number;
  lastUpdate: string;
  ip: string;
  gradientNorm: number;
  latencyMs: number;
  contributionPercent: number;
  isDropped?: boolean;
}

export interface HospitalMetricDetail {
  loss: number;
  dice: number;
  samples: number;
  contribution: number;
}

export interface RoundMetric {
  round: number;
  global_loss: number;
  dice: number;
  baseline_dice: number;
  round_duration_sec: number;
  participating_clients: number;
  total_clients: number;
  timestamp: string;
  hospital_metrics: Record<string, HospitalMetricDetail>;
  dice_tc?: number;
  dice_wt?: number;
  dice_et?: number;
}

export interface PrivacyState {
  epsilon: number;
  delta: number;
  noise_sigma: number;
  clip_norm: number;
  total_budget: number;
  homomorphic_encryption: {
    enabled: boolean;
    status: string;
    scheme: string;
    ciphertext_size_kb: number;
    poly_modulus_degree: number;
  };
}

export interface SegmentationCase {
  id: string;
  title: string;
  patientId: string;
  sliceZ: number;
  depthSlices: number;
  diceWT: number;
  diceTC: number;
  diceET: number;
  diceMean: number;
  baselineMean: number;
  hospitalSource: string;
  modality: string;
  previewUrl?: string;
}

export interface NotificationEvent {
  id: string;
  timestamp: string;
  type: 'overview' | 'nodes' | 'security';
  level: 'info' | 'warn' | 'success';
  title: string;
  message: string;
}

export type WebSocketMessage =
  | { type: 'metrics'; round: number; global_loss: number; dice: number; baseline_dice?: number }
  | { type: 'node_status'; node: string; status: NodeStatus; samples?: number }
  | { type: 'privacy'; epsilon: number; delta: number; noise_sigma: number }
  | { type: 'event'; level: 'info' | 'warn' | 'success'; message: string; title?: string; category?: 'overview' | 'nodes' | 'security' };
