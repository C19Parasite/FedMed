import React from 'react';
import { Activity, ShieldCheck, Wifi, WifiOff } from 'lucide-react';

export default function Header({ status, connected }) {
  const isRunning = status?.status === 'running';

  return (
    <header className="header">
      <div className="brand-section">
        <div className="brand-icon">🧠</div>
        <div className="brand-text">
          <h1>FedMed: Cross-Silo Federated Learning Engine</h1>
          <p>Multi-Silo 3D Brain Tumor MRI Segmentation (MONAI U-Net + Flower + CKKS HE + DP)</p>
        </div>
      </div>

      <div className="header-status">
        <div className={`badge ${connected ? 'badge-green' : 'badge-rose'}`}>
          {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
          <span>{connected ? 'WS LIVE' : 'WS DISCONNECTED'}</span>
        </div>

        <div className={`badge ${isRunning ? 'badge-cyan' : 'badge-purple'}`}>
          <div className="pulse-dot" />
          <span>{isRunning ? `TRAINING (R${status?.current_round || 0}/${status?.total_rounds || 3})` : 'SYSTEM IDLE'}</span>
        </div>

        <div className="badge badge-amber">
          <ShieldCheck size={14} />
          <span>EVAL MODE: BRAIN-TUMOR-TEST</span>
        </div>
      </div>
    </header>
  );
}
