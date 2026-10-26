import React from 'react';
import { Layers, ShieldCheck, CheckCircle2 } from 'lucide-react';

export default function AblationTable({ ablationData, loading }) {
  if (loading) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
        <p style={{ color: 'var(--text-muted)' }}>Loading ablation benchmarks...</p>
      </div>
    );
  }

  const baselineDice = 0.0206;

  return (
    <div className="card">
      <div className="card-title">
        <span>Empirical Privacy-Utility Ablation Benchmarks</span>
        <Layers size={18} color="var(--accent-cyan)" />
      </div>
      <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
        Empirical comparison across federated learning configurations. Evaluates segmentation accuracy (BraTS Mean Dice),
        privacy guarantee (differential privacy &epsilon;), and runtime/transmission costs.
      </p>

      <div className="table-container" style={{ marginTop: '12px' }}>
        <table>
          <thead>
            <tr>
              <th>Configuration</th>
              <th>Final Global Dice</th>
              <th>&Delta; vs Centralized</th>
              <th>DP Privacy (&epsilon;)</th>
              <th>Time / Round</th>
              <th>Payload / Round</th>
              <th>Security Level</th>
            </tr>
          </thead>
          <tbody>
            {(ablationData || []).map((row, idx) => {
              const dice = parseFloat(row.final_dice || 0);
              const diff = dice - baselineDice;
              const eps = parseFloat(row.epsilon || 0);
              const timeSec = parseFloat(row.time_per_round_sec || 0);
              const bytes = parseInt(row.bytes_per_round || 0, 10);
              const isFull = row.configuration.includes('FedMed Full');

              return (
                <tr
                  key={idx}
                  style={isFull ? { backgroundColor: 'rgba(6, 182, 212, 0.08)', fontWeight: 600 } : {}}
                >
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {isFull && <CheckCircle2 size={16} color="var(--accent-cyan)" />}
                      <span>{row.configuration}</span>
                    </div>
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}>
                    {(dice * 100).toFixed(2)}%
                  </td>
                  <td
                    style={{
                      fontFamily: 'var(--font-mono)',
                      color: diff >= 0 ? '#34d399' : '#fb7185',
                    }}
                  >
                    {diff >= 0 ? `+${(diff * 100).toFixed(2)}%` : `${(diff * 100).toFixed(2)}%`}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {eps > 0 ? (
                      <span className="badge badge-purple">&epsilon; = {eps.toFixed(2)}</span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>&infin; (No DP)</span>
                    )}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{timeSec.toFixed(2)}s</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{(bytes / 1024).toFixed(1)} KB</td>
                  <td>
                    {row.configuration.includes('Full') ? (
                      <span className="badge badge-green">TLS + DP + CKKS HE</span>
                    ) : row.configuration.includes('HE') ? (
                      <span className="badge badge-cyan">CKKS HE</span>
                    ) : row.configuration.includes('DP') ? (
                      <span className="badge badge-purple">Differential Privacy</span>
                    ) : row.configuration.includes('TLS') ? (
                      <span className="badge badge-amber">mTLS gRPC</span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>Plaintext FedAvg</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="card-subtext" style={{ marginTop: '8px' }}>
        * Note: All configurations evaluated on identical global holdout validation patients with fixed random seed (42).
      </div>
    </div>
  );
}
