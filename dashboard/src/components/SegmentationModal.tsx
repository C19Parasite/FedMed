import React, { useState } from 'react';
import { X, Layers, Sliders, ShieldCheck, Eye, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';

export const SegmentationModal: React.FC = () => {
  const { isModalOpen, activeModalCase, closeModal } = useDashboardStore();

  const [plane, setPlane] = useState<'axial' | 'sagittal' | 'coronal'>('axial');
  const [sliceZ, setSliceZ] = useState<number>(16);
  const [maskOpacity, setMaskOpacity] = useState<number>(0.65);
  const [showWT, setShowWT] = useState<boolean>(true);
  const [showTC, setShowTC] = useState<boolean>(true);
  const [showET, setShowET] = useState<boolean>(true);

  if (!isModalOpen || !activeModalCase) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 10 }}
          className="w-full max-w-4xl bg-white dark:bg-zinc-900 rounded-3xl shadow-2xl border border-brand-border dark:border-zinc-800 overflow-hidden flex flex-col max-h-[92vh]"
        >
          {/* Header */}
          <div className="px-6 py-4 border-b border-brand-border dark:border-zinc-800 flex items-center justify-between bg-surface-soft dark:bg-zinc-800/40">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-lavender-100 dark:bg-lavender-900/40 text-lavender-700 dark:text-lavender-300 flex items-center justify-center font-bold">
                <Layers size={18} />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-bold text-zinc-900 dark:text-white">
                    {activeModalCase.title}
                  </h3>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-mint-100 text-mint-700 dark:bg-mint-900/40 dark:text-mint-300">
                    Mean Dice: {(activeModalCase.diceMean * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="text-xs text-zinc-500 font-medium">
                  {activeModalCase.modality} • {activeModalCase.hospitalSource}
                </p>
              </div>
            </div>

            <button
              onClick={closeModal}
              className="w-8 h-8 rounded-full hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-400 hover:text-zinc-700 dark:hover:text-white flex items-center justify-center transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          {/* Privacy Notice Banner */}
          <div className="px-6 py-2 bg-lavender-50 dark:bg-lavender-950/30 border-b border-lavender-200 dark:border-lavender-900/50 flex items-center justify-between text-xs text-lavender-800 dark:text-lavender-300">
            <span className="flex items-center gap-1.5 font-semibold">
              <ShieldCheck size={14} className="text-lavender-600" />
              Public BraTS Demo Cohort: Zero private hospital patient records transmitted
            </span>
            <span className="font-mono text-[11px] opacity-75">ID: {activeModalCase.patientId}</span>
          </div>

          {/* Modal Body */}
          <div className="p-6 overflow-y-auto space-y-6">
            {/* Plane Switcher (Axial, Sagittal, Coronal) */}
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex p-1 bg-surface-soft dark:bg-zinc-800 rounded-2xl border border-brand-border">
                {(['axial', 'sagittal', 'coronal'] as const).map((p) => (
                  <button
                    key={p}
                    onClick={() => setPlane(p)}
                    className={`px-4 py-1.5 rounded-xl text-xs font-bold capitalize transition-all ${
                      plane === p
                        ? 'bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white shadow-sm'
                        : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'
                    }`}
                  >
                    {p} View
                  </button>
                ))}
              </div>

              {/* Subregion Color Legend Toggles */}
              <div className="flex items-center gap-3 text-xs">
                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={showWT}
                    onChange={(e) => setShowWT(e.target.checked)}
                    className="accent-mint-500 rounded"
                  />
                  <span className="font-semibold text-mint-600">Whole Tumor (WT)</span>
                </label>

                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={showTC}
                    onChange={(e) => setShowTC(e.target.checked)}
                    className="accent-amber-500 rounded"
                  />
                  <span className="font-semibold text-amber-600">Tumor Core (TC)</span>
                </label>

                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={showET}
                    onChange={(e) => setShowET(e.target.checked)}
                    className="accent-coral-500 rounded"
                  />
                  <span className="font-semibold text-coral-600">Enhancing (ET)</span>
                </label>
              </div>
            </div>

            {/* Side-by-Side Viewport: Ground Truth vs Federated Prediction */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Left Viewport: Ground Truth */}
              <div className="flex flex-col items-center">
                <div className="w-full flex items-center justify-between mb-2 px-1">
                  <span className="text-xs font-bold text-zinc-700 dark:text-zinc-200 uppercase tracking-wider">
                    Neuroradiologist Ground Truth
                  </span>
                  <span className="text-[11px] font-mono text-zinc-400">Manual Contours</span>
                </div>

                <div className="relative w-full aspect-square bg-[#06080D] rounded-2xl overflow-hidden border border-zinc-800 shadow-inner flex items-center justify-center">
                  {/* Simulated Brain Slice Anatomy */}
                  <div className="absolute inset-4 rounded-full bg-gradient-to-tr from-zinc-800 via-zinc-700 to-zinc-900 opacity-80" />
                  <div className="absolute w-44 h-52 rounded-full bg-zinc-600/30 blur-[2px]" />

                  {/* Ground Truth Contours */}
                  {showWT && (
                    <div
                      className="absolute w-24 h-24 rounded-full border-2 border-mint-400 bg-mint-500/30 transform translate-x-4 -translate-y-2 transition-opacity"
                      style={{ opacity: maskOpacity }}
                    />
                  )}
                  {showTC && (
                    <div
                      className="absolute w-14 h-14 rounded-full border-2 border-amber-400 bg-amber-500/35 transform translate-x-5 -translate-y-1 transition-opacity"
                      style={{ opacity: maskOpacity }}
                    />
                  )}
                  {showET && (
                    <div
                      className="absolute w-8 h-8 rounded-full border border-coral-400 bg-coral-500/50 transform translate-x-6 -translate-y-0.5 transition-opacity"
                      style={{ opacity: maskOpacity }}
                    />
                  )}

                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/60 text-[10px] font-mono text-zinc-300">
                    GT {plane.toUpperCase()} #{sliceZ}
                  </div>
                </div>
              </div>

              {/* Right Viewport: Federated Prediction */}
              <div className="flex flex-col items-center">
                <div className="w-full flex items-center justify-between mb-2 px-1">
                  <span className="text-xs font-bold text-lavender-700 dark:text-lavender-400 uppercase tracking-wider">
                    FedMed 3D U-Net Output
                  </span>
                  <span className="text-[11px] font-mono text-mint-600 font-bold">
                    Dice: {(activeModalCase.diceMean * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="relative w-full aspect-square bg-[#06080D] rounded-2xl overflow-hidden border border-lavender-400/40 shadow-inner flex items-center justify-center">
                  {/* Simulated Brain Slice Anatomy */}
                  <div className="absolute inset-4 rounded-full bg-gradient-to-tr from-zinc-800 via-zinc-700 to-zinc-900 opacity-80" />
                  <div className="absolute w-44 h-52 rounded-full bg-zinc-600/30 blur-[2px]" />

                  {/* Federated Model Output Contours */}
                  {showWT && (
                    <div
                      className="absolute w-23 h-23 rounded-full border-2 border-mint-400 bg-mint-500/35 transform translate-x-4 -translate-y-2 transition-opacity shadow-glow-mint"
                      style={{ opacity: maskOpacity }}
                    />
                  )}
                  {showTC && (
                    <div
                      className="absolute w-13 h-13 rounded-full border-2 border-amber-400 bg-amber-500/40 transform translate-x-5 -translate-y-1 transition-opacity"
                      style={{ opacity: maskOpacity }}
                    />
                  )}
                  {showET && (
                    <div
                      className="absolute w-7 h-7 rounded-full border border-coral-400 bg-coral-500/55 transform translate-x-6 -translate-y-0.5 transition-opacity"
                      style={{ opacity: maskOpacity }}
                    />
                  )}

                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-lavender-900/70 text-[10px] font-mono text-lavender-200">
                    PRED {plane.toUpperCase()} #{sliceZ}
                  </div>
                </div>
              </div>
            </div>

            {/* Sliders Controls Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-4 rounded-2xl bg-surface-soft dark:bg-zinc-800/40 border border-brand-border">
              {/* Slice Depth Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-zinc-600 dark:text-zinc-300">Volumetric Slice (Z):</span>
                  <span className="font-mono text-lavender-600 font-bold">
                    {sliceZ} / {activeModalCase.depthSlices}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={activeModalCase.depthSlices}
                  value={sliceZ}
                  onChange={(e) => setSliceZ(parseInt(e.target.value, 10))}
                  className="w-full accent-lavender-600 cursor-pointer"
                />
              </div>

              {/* Mask Opacity Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-zinc-600 dark:text-zinc-300">Tumor Overlay Opacity:</span>
                  <span className="font-mono text-mint-600 font-bold">
                    {Math.round(maskOpacity * 100)}%
                  </span>
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
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
