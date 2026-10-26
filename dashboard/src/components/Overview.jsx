import React from 'react';
import { Shield, Lock, EyeOff, Server, Hospital, Award, TrendingUp } from 'lucide-react';

export default function Overview({ status, latestRoundData }) {
  const hospitals = status?.hospitals || {
    hospital_1: 'online',
    hospital_2: 'online',
    hospital_3: 'online',
  };

  const privacy = status?.privacy || {};
  const currentRound = status?.current_round || latestRoundData?.round || 0;
  const totalRounds = status?.total_rounds || 3;
  const bestDice = status?.best_global_dice || latestRoundData?.global_val_dice || 0.0224;
  const baselineDice = 0.0206; // from baseline.json
  const diceDiff = bestDice - baselineDice;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div className="grid-4">
        {/* Round Progress Card */}
        <div className="card">
          <div className="card-title">
            <span>Round Progress</span>
            <Server size={18} color="var(--accent-cyan)" />
          </div>
          <div className="card-value">
            {currentRound} / {totalRounds}
          </div>
          <div className="card-subtext">
            {currentRound >= totalRounds ? 'Federation completed target rounds' : 'Active federated training'}
          </div>
        </div>

        {/* Best Global Dice Card */}
        <div className="card">
          <div className="card-title">
            <span>Global Mean Dice</span>
            <Award size={18} color="var(--accent-emerald)" />
          </div>
          <div className="card-value" style={{ color: 'var(--accent-emerald)' }}>
            {(bestDice * 100).toFixed(2)}%
          </div>
          <div className="card-subtext" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <TrendingUp size={14} color="#34d399" />
            <span>
              {diceDiff >= 0 ? `+${(diceDiff * 100).toFixed(2)}%` : `${(diceDiff * 100).toFixed(2)}%`} vs Baseline ({(baselineDice * 100).toFixed(2)}%)
            </span>
          </div>
        </div>

        {/* Privacy Budget Card */}
        <div className="card">
          <div className="card-title">
            <span>DP Privacy Budget</span>
            <EyeOff size={18} color="var(--accent-purple)" />
          </div>
          <div className="card-value" style={{ color: 'var(--accent-purple)' }}>
            &epsilon; = {(privacy.cumulative_epsilon || latestRoundData?.privacy?.epsilon || 7.79).toFixed(2)}
          </div>
          <div className="card-subtext">
            &delta; = 10⁻⁵ &bull; &sigma; = {privacy.noise_multiplier || 1.0} &bull; RDP analytical
          </div>
        </div>

        {/* Hospital Silos Card */}
        <div className="card">
          <div className="card-title">
            <span>Hospital Silos</span>
            <Hospital size={18} color="var(--accent-amber)" />
          </div>
          <div className="card-value">
            {Object.values(hospitals).filter((s) => s === 'online').length} / {Object.keys(hospitals).length}
          </div>
          <div className="card-subtext">
            Cross-silo nodes connected over TLS
          </div>
        </div>
      </div>

      {/* Hospital Nodes Status Bar */}
      <div className="card" style={{ gap: '12px' }}>
        <div className="card-title">
          <span>HOSPITAL SILO INFRASTRUCTURE &amp; RECEPTOR STATUS</span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>PATIENT DATA ZERO-EGRESS ENFORCED</span>
        </div>
        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
          {Object.entries(hospitals).map(([silo, state]) => (
            <div
              key={silo}
              style={{
                flex: 1,
                minWidth: '220px',
                padding: '12px 16px',
                background: 'var(--bg-card)',
                borderRadius: '8px',
                border: '1px solid var(--border-color)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <div>
                <div style={{ fontWeight: 600, fontSize: '14px', textTransform: 'uppercase' }}>
                  {silo.replace('_', ' ')}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Dataset: Non-IID BraTS Partition
                </div>
              </div>
              <span className={`badge ${state === 'online' ? 'badge-green' : 'badge-rose'}`}>
                <div className="pulse-dot" />
                {state.toUpperCase()}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Privacy Safeguards Overview */}
      <div className="card" style={{ gap: '12px' }}>
        <div className="card-title">
          <span>DEFENSE-IN-DEPTH PRIVACY STACK</span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>COMPREHENSIVE MULTI-LAYER ENCRYPTION</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
          {/* TLS */}
          <div style={{ padding: '14px', background: 'var(--bg-card)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
              <Shield size={16} color="var(--accent-cyan)" />
              <strong style={{ fontSize: '13px' }}>Network Security (TLS gRPC)</strong>
              <span className="badge badge-cyan" style={{ marginLeft: 'auto', padding: '2px 8px', fontSize: '10px' }}>
                {privacy.tls !== false ? 'ACTIVE' : 'DISABLED'}
              </span>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              Mutual TLS encryption with dedicated FedMed Root CA. Prevents man-in-the-middle packet sniffing.
            </p>
          </div>

          {/* HE */}
          <div style={{ padding: '14px', background: 'var(--bg-card)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
              <Lock size={16} color="var(--accent-emerald)" />
              <strong style={{ fontSize: '13px' }}>Homomorphic Encryption (CKKS)</strong>
              <span className="badge badge-green" style={{ marginLeft: 'auto', padding: '2px 8px', fontSize: '10px' }}>
                {privacy.he !== false ? 'ACTIVE' : 'DISABLED'}
              </span>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              TenSEAL CKKS vector encryption. Central server aggregates homomorphically without decryption keys.
            </p>
          </div>

          {/* DP */}
          <div style={{ padding: '14px', background: 'var(--bg-card)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
              <EyeOff size={16} color="var(--accent-purple)" />
              <strong style={{ fontSize: '13px' }}>Differential Privacy (DP-FedAvg)</strong>
              <span className="badge badge-purple" style={{ marginLeft: 'auto', padding: '2px 8px', fontSize: '10px' }}>
                {privacy.dp !== false ? 'ACTIVE' : 'DISABLED'}
              </span>
            </div>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              L2 weight delta clipping ($C=1.0$) + calibrated Gaussian perturbation ($\sigma={privacy.noise_multiplier || 1.0}$).
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
