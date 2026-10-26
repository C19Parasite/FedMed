import React, { useState } from 'react';
import { Eye, ShieldCheck, Sliders, Layers, Info } from 'lucide-react';
import { getSliceUrl } from '../api';

export default function SegmentationViewer({ cases, loading }) {
  const [selectedCaseIdx, setSelectedCaseIdx] = useState(0);
  const [sliceZ, setSliceZ] = useState(15);
  const [activeView, setActiveView] = useState('overlay'); // 'overlay' | 'mri' | 'gt' | 'pred' | 'central'

  if (loading || !cases || cases.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
        <p style={{ color: 'var(--text-muted)' }}>Loading public BraTS demonstration cases...</p>
      </div>
    );
  }

  const currentCase = cases[selectedCaseIdx] || cases[0];
  const maxSlices = currentCase.depth_slices || 32;
  const clampedSlice = Math.min(sliceZ, maxSlices - 1);
  const imageUrl = getSliceUrl(currentCase.id, clampedSlice, activeView);

  const fedDice = currentCase.dice || {};
  const centralDice = currentCase.centralized_dice || {};

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Privacy Notice Banner */}
      <div
        className="card"
        style={{
          backgroundColor: 'rgba(6, 182, 212, 0.08)',
          borderColor: 'rgba(6, 182, 212, 0.3)',
          flexDirection: 'row',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <ShieldCheck size={24} color="var(--accent-cyan)" />
          <div>
            <div style={{ fontWeight: 600, fontSize: '14px', color: 'var(--accent-cyan)' }}>
              PUBLIC BRATS DEMO CASE &bull; ZERO RAW PATIENT DATA LEAVING HOSPITALS
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
              {currentCase.dataset_note || 'Held-out global evaluation case. Evaluated post-training for visualization.'}
            </div>
          </div>
        </div>
        <span className="badge badge-cyan">HIPAA / GDPR ISOLATION PRESERVED</span>
      </div>

      <div className="viewer-container">
        {/* Left Side: Image Display and Controls */}
        <div className="image-panel">
          <img
            src={imageUrl}
            alt={`MRI Slice ${clampedSlice} - ${activeView}`}
            className="mri-display"
            onError={(e) => {
              // fallback if specific slice fails
              e.target.style.opacity = '0.5';
            }}
          />

          {/* View Mode Toggle Buttons */}
          <div className="controls-row">
            <button
              className={`btn-toggle ${activeView === 'overlay' ? 'active' : ''}`}
              onClick={() => setActiveView('overlay')}
            >
              Side-by-Side Overlay
            </button>
            <button
              className={`btn-toggle ${activeView === 'mri' ? 'active' : ''}`}
              onClick={() => setActiveView('mri')}
            >
              Raw MRI (FLAIR/T1ce)
            </button>
            <button
              className={`btn-toggle ${activeView === 'gt' ? 'active' : ''}`}
              onClick={() => setActiveView('gt')}
            >
              Ground Truth Mask
            </button>
            <button
              className={`btn-toggle ${activeView === 'pred' ? 'active' : ''}`}
              onClick={() => setActiveView('pred')}
            >
              Federated Model Pred
            </button>
            <button
              className={`btn-toggle ${activeView === 'central' ? 'active' : ''}`}
              onClick={() => setActiveView('central')}
            >
              Centralized Baseline Pred
            </button>
          </div>

          {/* Slice Depth Slider */}
          <div className="slider-group">
            <Sliders size={18} color="var(--accent-cyan)" />
            <span style={{ fontSize: '13px', color: 'var(--text-secondary)', minWidth: '55px' }}>
              Slice Z:
            </span>
            <input
              type="range"
              min={0}
              max={maxSlices - 1}
              value={clampedSlice}
              onChange={(e) => setSliceZ(parseInt(e.target.value, 10))}
            />
            <span className="slice-counter">
              {clampedSlice + 1} / {maxSlices}
            </span>
          </div>
        </div>

        {/* Right Side: Case Metadata & Metrics Breakdown */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {/* Case Selector */}
          <div className="card">
            <div className="card-title">
              <span>Select Demonstration Case</span>
              <Eye size={18} color="var(--accent-cyan)" />
            </div>
            <select
              style={{
                background: 'var(--bg-card)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                padding: '10px',
                fontSize: '13px',
                width: '100%',
                outline: 'none',
              }}
              value={selectedCaseIdx}
              onChange={(e) => setSelectedCaseIdx(parseInt(e.target.value, 10))}
            >
              {cases.map((c, idx) => (
                <option key={c.id} value={idx}>
                  {c.label} ({c.id})
                </option>
              ))}
            </select>
          </div>

          {/* Segmentation Dice Breakdown Card */}
          <div className="card">
            <div className="card-title">
              <span>Per-Case BraTS Segmentation Dice</span>
              <Layers size={18} color="var(--accent-emerald)" />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Whole Tumor (WT):</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-cyan)' }}>
                    {((fedDice.dice_wt || 0) * 100).toFixed(2)}%
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Centralized: {((centralDice.dice_wt || 0) * 100).toFixed(2)}%
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Tumor Core (TC):</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-amber)' }}>
                    {((fedDice.dice_tc || 0) * 100).toFixed(2)}%
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Centralized: {((centralDice.dice_tc || 0) * 100).toFixed(2)}%
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Enhancing Tumor (ET):</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-rose)' }}>
                    {((fedDice.dice_et || 0) * 100).toFixed(2)}%
                  </span>
                </div>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Centralized: {((centralDice.dice_et || 0) * 100).toFixed(2)}%
                </div>
              </div>

              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px', fontWeight: 700 }}>
                  <span>Mean Case Dice:</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-emerald)' }}>
                    {((fedDice.dice_mean || 0) * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Model Architecture Note */}
          <div className="card">
            <div className="card-title">
              <span>Model & Inference Specs</span>
              <Info size={18} color="var(--text-muted)" />
            </div>
            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div>&bull; <strong>Backbone:</strong> MONAI 3D U-Net</div>
              <div>&bull; <strong>Normalization:</strong> InstanceNorm3d</div>
              <div>&bull; <strong>Input Channels:</strong> 4 (FLAIR, T1, T1ce, T2)</div>
              <div>&bull; <strong>Output Channels:</strong> 3 (TC, WT, ET)</div>
              <div>&bull; <strong>Sliding Window:</strong> ROI 32x32x32</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
