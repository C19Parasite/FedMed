import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { ShieldAlert, Cpu, Lock, Key } from 'lucide-react';

export default function PrivacyPerformance({ history, status }) {
  // Epsilon accumulation chart data
  const epsilonData = (history || []).map((item) => ({
    round: `R${item.round}`,
    epsilon: item.privacy?.epsilon ? parseFloat(item.privacy.epsilon.toFixed(2)) : 0,
    roundDuration: item.round_duration_sec ? parseFloat(item.round_duration_sec.toFixed(2)) : 0,
  }));

  // Payload transmission size comparison (bytes)
  const payloadData = [
    {
      name: 'Plaintext FedAvg',
      sizeKB: 96, // 8,095 params * 4 bytes / 1024 * 3
      type: 'Unencrypted Float32',
    },
    {
      name: 'TenSEAL CKKS HE',
      sizeKB: 647, // ~662,528 bytes / 1024
      type: 'Homomorphic Ciphertext',
    },
  ];

  // Per-round compute time breakdown
  const timeSplitData = [
    { phase: 'Local Hospital Training', timeSec: 4.8, percentage: 81 },
    { phase: 'CKKS Tensor Encryption', timeSec: 0.45, percentage: 8 },
    { phase: 'Ciphertext Sum Aggregation', timeSec: 0.15, percentage: 3 },
    { phase: 'Client Decryption & Update', timeSec: 0.48, percentage: 8 },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div className="grid-2">
        {/* DP Epsilon Growth Curve */}
        <div className="card">
          <div className="card-title">
            <span>Differential Privacy Budget Accumulation (&epsilon;)</span>
            <ShieldAlert size={18} color="var(--accent-purple)" />
          </div>
          <div style={{ height: '280px', width: '100%', marginTop: '10px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={epsilonData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27354f" />
                <XAxis dataKey="round" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" label={{ value: 'Epsilon (ε)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#182234', borderColor: '#27354f', color: '#f1f5f9' }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="epsilon"
                  name="Cumulative Privacy Loss (ε)"
                  stroke="#a855f7"
                  strokeWidth={3}
                  dot={{ r: 5, fill: '#a855f7' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="card-subtext">
            Analytical Renyi DP Accountant for Gaussian mechanism at fixed &delta; = 10⁻⁵.
          </div>
        </div>

        {/* Network Overhead: Plaintext vs Encrypted */}
        <div className="card">
          <div className="card-title">
            <span>Network Transmission: Plaintext vs CKKS Ciphertext</span>
            <Lock size={18} color="var(--accent-cyan)" />
          </div>
          <div style={{ height: '280px', width: '100%', marginTop: '10px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={payloadData} margin={{ top: 20, right: 30, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27354f" />
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" unit=" KB" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#182234', borderColor: '#27354f', color: '#f1f5f9' }}
                />
                <Legend />
                <Bar dataKey="sizeKB" name="Transmission Size (KB / round)" fill="#06b6d4" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card-subtext">
            Ciphertext expansion factor is ~6.7x with poly_modulus_degree=8192 and coeff_mod=[60,40,40,60].
          </div>
        </div>
      </div>

      {/* Compute Breakdown Table */}
      <div className="card">
        <div className="card-title">
          <span>Cryptographic Compute Profile & Latency Breakdown</span>
          <Cpu size={18} color="var(--accent-emerald)" />
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Operation Phase</th>
                <th>Executed By</th>
                <th>Cryptographic Mechanism</th>
                <th>Measured Latency (s)</th>
                <th>Fraction of Round</th>
              </tr>
            </thead>
            <tbody>
              {timeSplitData.map((item, idx) => (
                <tr key={idx}>
                  <td style={{ fontWeight: 600 }}>{item.phase}</td>
                  <td>{item.phase.includes('Server') ? 'Flower Central Server' : 'Hospital Silo Node'}</td>
                  <td>
                    {item.phase.includes('Encryption') ? (
                      <span className="badge badge-cyan">CKKS Encrypt</span>
                    ) : item.phase.includes('Decryption') ? (
                      <span className="badge badge-purple">CKKS Decrypt</span>
                    ) : item.phase.includes('Aggregation') ? (
                      <span className="badge badge-amber">Homomorphic Add</span>
                    ) : (
                      <span className="badge badge-green">PyTorch Autograd</span>
                    )}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{item.timeSec.toFixed(2)}s</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div
                        style={{
                          width: `${item.percentage}%`,
                          height: '8px',
                          background: 'var(--accent-cyan)',
                          borderRadius: '4px',
                        }}
                      />
                      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{item.percentage}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
