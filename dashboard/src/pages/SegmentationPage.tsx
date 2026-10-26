import React, { useState } from 'react';
import { Scan, Eye, Layers, ShieldCheck, Sliders, CheckCircle2 } from 'lucide-react';
import { useDashboardStore } from '../store/useDashboardStore';
import { SegmentationCase } from '../types';

export const SegmentationPage: React.FC = () => {
  const { cases, openModal } = useDashboardStore();
  const [selectedCase, setSelectedCase] = useState<SegmentationCase>(cases[0]);
  const [sliceZ, setSliceZ] = useState<number>(16);
  const [maskOpacity, setMaskOpacity] = useState<number>(0.7);
  const [activeModality, setActiveModality] = useState<'FLAIR' | 'T1ce' | 'T2'>('FLAIR');

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="card-base p-6 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-extrabold text-zinc-900 dark:text-white">
            3D Brain Tumor MRI Segmentation Gallery
          </h2>
          <p className="text-xs text-zinc-500 mt-1">
            Visual inspection of 3D MONAI U-Net segmentations on held-out public BraTS benchmark cases.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-3 py-1.5 rounded-full text-xs font-semibold bg-lavender-100 text-lavender-700 dark:bg-lavender-950/60 dark:text-lavender-300">
            {cases.length} Public Benchmark Cases
          </span>
        </div>
      </div>

      {/* Case Selector Pills */}
      <div className="flex gap-2.5 overflow-x-auto no-scrollbar pb-2">
        {cases.map((c) => {
          const isSelected = selectedCase.id === c.id;
          return (
            <button
              key={c.id}
              onClick={() => setSelectedCase(c)}
              className={`px-4 py-2 rounded-2xl text-xs font-bold transition-all shrink-0 flex items-center gap-2 ${
                isSelected
                  ? 'bg-surface-dark text-white shadow-soft scale-102'
                  : 'bg-white dark:bg-zinc-800 text-zinc-600 dark:text-zinc-300 border border-brand-border hover:border-lavender-300'
              }`}
            >
              <span>{c.title}</span>
              <span className={`px-1.5 py-0.5 rounded-full text-[10px] font-mono ${isSelected ? 'bg-zinc-800 text-mint-300' : 'bg-surface-soft text-zinc-500'}`}>
                {(c.diceMean * 100).toFixed(0)}%
              </span>
            </button>
          );
        })}
      </div>

      {/* Interactive Main Viewer Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Viewports: Side by Side Slice Renderers (8 cols) */}
        <div className="lg:col-span-8 card-base p-6 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-brand-border/60">
            <div>
              <h3 className="text-base font-bold text-zinc-900 dark:text-white">
                {selectedCase.title} ({selectedCase.patientId})
              </h3>
              <p className="text-xs text-zinc-400 font-medium">
                Origin: {selectedCase.hospitalSource} • Modality: {selectedCase.modality}
              </p>
            </div>

            <button
              onClick={() => openModal(selectedCase)}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-lavender-100 hover:bg-lavender-200 dark:bg-lavender-900/40 dark:hover:bg-lavender-900/60 text-lavender-800 dark:text-lavender-200 transition-colors"
            >
              <Eye size={13} />
              <span>Full Multi-Planar Modal</span>
            </button>
          </div>

          {/* Viewports Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Ground Truth */}
            <div>
              <div className="flex justify-between text-xs font-bold text-zinc-600 dark:text-zinc-300 mb-1.5">
                <span>Ground Truth Mask</span>
                <span className="font-mono text-zinc-400">Axial Z: {sliceZ}</span>
              </div>
              <div className="relative aspect-square bg-[#07090E] rounded-2xl overflow-hidden border border-zinc-800 flex items-center justify-center">
                <div className="absolute inset-4 rounded-full bg-gradient-to-tr from-zinc-800 via-zinc-700 to-zinc-900 opacity-80" />
                <div className="absolute w-40 h-48 rounded-full bg-zinc-600/30 blur-[2px]" />
                <div
                  className="absolute w-22 h-22 rounded-full border-2 border-mint-400 bg-mint-500/35 transform translate-x-3 -translate-y-2 transition-opacity"
                  style={{ opacity: maskOpacity }}
                />
                <div
                  className="absolute w-12 h-12 rounded-full border border-amber-400 bg-amber-500/40 transform translate-x-4 -translate-y-1 transition-opacity"
                  style={{ opacity: maskOpacity }}
                />
              </div>
            </div>

            {/* Federated Prediction */}
            <div>
              <div className="flex justify-between text-xs font-bold text-lavender-700 dark:text-lavender-400 mb-1.5">
                <span>FedMed Prediction</span>
                <span className="font-mono text-mint-600 font-bold">Dice: {(selectedCase.diceMean * 100).toFixed(1)}%</span>
              </div>
              <div className="relative aspect-square bg-[#07090E] rounded-2xl overflow-hidden border border-lavender-400/40 flex items-center justify-center">
                <div className="absolute inset-4 rounded-full bg-gradient-to-tr from-zinc-800 via-zinc-700 to-zinc-900 opacity-80" />
                <div className="absolute w-40 h-48 rounded-full bg-zinc-600/30 blur-[2px]" />
                <div
                  className="absolute w-21 h-21 rounded-full border-2 border-mint-400 bg-mint-500/40 transform translate-x-3 -translate-y-2 transition-opacity shadow-glow-mint"
                  style={{ opacity: maskOpacity }}
                />
                <div
                  className="absolute w-11 h-11 rounded-full border border-amber-400 bg-amber-500/45 transform translate-x-4 -translate-y-1 transition-opacity"
                  style={{ opacity: maskOpacity }}
                />
              </div>
            </div>
          </div>

          {/* Sliders */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-3 border-t border-brand-border/60">
            <div>
              <div className="flex justify-between text-xs font-semibold text-zinc-600 dark:text-zinc-300 mb-1">
                <span>Slice Depth (Z):</span>
                <span className="font-mono text-lavender-600">{sliceZ} / {selectedCase.depthSlices}</span>
              </div>
              <input
                type="range"
                min={1}
                max={selectedCase.depthSlices}
                value={sliceZ}
                onChange={(e) => setSliceZ(parseInt(e.target.value, 10))}
                className="w-full accent-lavender-600 cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs font-semibold text-zinc-600 dark:text-zinc-300 mb-1">
                <span>Overlay Opacity:</span>
                <span className="font-mono text-mint-600">{Math.round(maskOpacity * 100)}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={maskOpacity}
                onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                className="w-full accent-mint-500 cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Right Info Card (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="card-base p-6 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800 space-y-4">
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white pb-3 border-b border-brand-border/60">
              Quantitative Dice Analysis
            </h3>

            <div className="space-y-3 text-xs">
              <div>
                <div className="flex justify-between font-semibold mb-1">
                  <span className="text-zinc-500">Whole Tumor (WT):</span>
                  <span className="font-mono font-bold text-mint-600">{(selectedCase.diceWT * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-zinc-100 dark:bg-zinc-800 rounded-full h-2 overflow-hidden">
                  <div className="bg-mint-500 h-2 rounded-full" style={{ width: `${selectedCase.diceWT * 100}%` }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between font-semibold mb-1">
                  <span className="text-zinc-500">Tumor Core (TC):</span>
                  <span className="font-mono font-bold text-amber-600">{(selectedCase.diceTC * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-zinc-100 dark:bg-zinc-800 rounded-full h-2 overflow-hidden">
                  <div className="bg-amber-500 h-2 rounded-full" style={{ width: `${selectedCase.diceTC * 100}%` }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between font-semibold mb-1">
                  <span className="text-zinc-500">Enhancing Tumor (ET):</span>
                  <span className="font-mono font-bold text-coral-500">{(selectedCase.diceET * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-zinc-100 dark:bg-zinc-800 rounded-full h-2 overflow-hidden">
                  <div className="bg-coral-500 h-2 rounded-full" style={{ width: `${selectedCase.diceET * 100}%` }} />
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-brand-border/60 text-xs space-y-2">
              <div className="flex justify-between font-bold text-zinc-900 dark:text-white">
                <span>Mean Case Dice:</span>
                <span className="font-mono text-mint-600">{(selectedCase.diceMean * 100).toFixed(2)}%</span>
              </div>
              <div className="flex justify-between text-zinc-400 text-[11px]">
                <span>Centralized Baseline:</span>
                <span className="font-mono font-semibold">{(selectedCase.baselineMean * 100).toFixed(2)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
