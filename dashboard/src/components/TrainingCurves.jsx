import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';

export default function TrainingCurves({ history }) {
  // Transform history data for recharts
  const chartData = (history || []).map((item) => {
    const hMetrics = item.hospital_metrics || {};
    return {
      round: `R${item.round}`,
      roundNum: item.round,
      globalDice: item.global_val_dice ? parseFloat((item.global_val_dice * 100).toFixed(2)) : null,
      diceTC: item.dice_breakdown?.dice_tc ? parseFloat((item.dice_breakdown.dice_tc * 100).toFixed(2)) : null,
      diceWT: item.dice_breakdown?.dice_wt ? parseFloat((item.dice_breakdown.dice_wt * 100).toFixed(2)) : null,
      diceET: item.dice_breakdown?.dice_et ? parseFloat((item.dice_breakdown.dice_et * 100).toFixed(2)) : null,
      globalLoss: item.global_val_loss ? parseFloat(item.global_val_loss.toFixed(4)) : null,
      h1Loss: hMetrics.hospital_1?.train_loss ? parseFloat(hMetrics.hospital_1.train_loss.toFixed(4)) : null,
      h2Loss: hMetrics.hospital_2?.train_loss ? parseFloat(hMetrics.hospital_2.train_loss.toFixed(4)) : null,
      h3Loss: hMetrics.hospital_3?.train_loss ? parseFloat(hMetrics.hospital_3.train_loss.toFixed(4)) : null,
    };
  });

  const baselineDicePercent = 2.06; // 0.0206 * 100 from results/baseline.json

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Global Validation Dice Chart with Baseline Reference */}
      <div className="card">
        <div className="card-title">
          <span>Global Validation BraTS Dice Score (%) vs Centralized Baseline</span>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Centralized Target: 2.06% &bull; Evaluation on Global Holdout
          </span>
        </div>
        <div style={{ height: '320px', width: '100%', marginTop: '10px' }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27354f" />
              <XAxis dataKey="round" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" domain={['dataMin - 0.5', 'dataMax + 0.5']} unit="%" />
              <Tooltip
                contentStyle={{ backgroundColor: '#182234', borderColor: '#27354f', color: '#f1f5f9' }}
                itemStyle={{ color: '#f1f5f9' }}
              />
              <Legend />
              <ReferenceLine
                y={baselineDicePercent}
                label={{ value: 'Centralized Baseline (2.06%)', fill: '#f59e0b', position: 'top', fontSize: 12 }}
                stroke="#f59e0b"
                strokeDasharray="5 5"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="globalDice"
                name="FedMed Global Mean Dice"
                stroke="#10b981"
                strokeWidth={3}
                dot={{ r: 5, fill: '#10b981' }}
                activeDot={{ r: 8 }}
              />
              <Line
                type="monotone"
                dataKey="diceWT"
                name="Whole Tumor (WT)"
                stroke="#06b6d4"
                strokeWidth={1.5}
                strokeDasharray="2 2"
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="diceTC"
                name="Tumor Core (TC)"
                stroke="#6366f1"
                strokeWidth={1.5}
                strokeDasharray="2 2"
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="diceET"
                name="Enhancing Tumor (ET)"
                stroke="#f43f5e"
                strokeWidth={1.5}
                strokeDasharray="2 2"
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Grid of Loss Curves */}
      <div className="grid-2">
        {/* Global Validation Loss */}
        <div className="card">
          <div className="card-title">
            <span>Global Validation DiceCE Loss</span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Evaluated centrally at server</span>
          </div>
          <div style={{ height: '260px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27354f" />
                <XAxis dataKey="round" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#182234', borderColor: '#27354f', color: '#f1f5f9' }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="globalLoss"
                  name="Global Val Loss"
                  stroke="#38bdf8"
                  strokeWidth={2.5}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Per-Hospital Local Training Loss */}
        <div className="card">
          <div className="card-title">
            <span>Per-Hospital Local Training Loss</span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Computed on private silo partitions</span>
          </div>
          <div style={{ height: '260px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27354f" />
                <XAxis dataKey="round" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#182234', borderColor: '#27354f', color: '#f1f5f9' }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="h1Loss"
                  name="Hospital 1"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="h2Loss"
                  name="Hospital 2"
                  stroke="#ec4899"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="h3Loss"
                  name="Hospital 3"
                  stroke="#a855f7"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
