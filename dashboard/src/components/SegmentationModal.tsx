import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { X, Layers, Sliders, ShieldCheck, Eye, Sparkles } from 'lucide-react';
import { motion, AnimatePresence, PanInfo } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';
import { BlurImage } from './BlurImage';
import { MRIViewport } from './MRIViewport';

export const SegmentationModal: React.FC = () => {
  const { isModalOpen, activeModalCase, closeModal, cases, openModal } = useDashboardStore();
  const [searchParams, setSearchParams] = useSearchParams();

  const [plane, setPlane] = useState<'axial' | 'sagittal' | 'coronal'>('axial');
  const [sliceZ, setSliceZ] = useState<number>(16);
  const [maskOpacity, setMaskOpacity] = useState<number>(0.65);
  const [showWT, setShowWT] = useState<boolean>(true);
  const [showTC, setShowTC] = useState<boolean>(true);
  const [showET, setShowET] = useState<boolean>(true);

  // Sync from URL parameter (?case=case_0132) into modal state
  const caseFromUrl = searchParams.get('case');
  useEffect(() => {
    if (caseFromUrl) {
      const found = cases.find((c) => c.id === caseFromUrl);
      if (found && (!isModalOpen || activeModalCase?.id !== found.id)) {
        openModal(found);
      }
    }
  }, [caseFromUrl, cases, isModalOpen, activeModalCase, openModal]);

  // Sync from modal state into URL parameter
  useEffect(() => {
    if (isModalOpen && activeModalCase) {
      if (searchParams.get('case') !== activeModalCase.id) {
        setSearchParams((prev) => {
          const next = new URLSearchParams(prev);
          next.set('case', activeModalCase.id);
          return next;
        });
      }
    }
  }, [isModalOpen, activeModalCase, searchParams, setSearchParams]);

  const handleClose = () => {
    closeModal();
    if (searchParams.has('case')) {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.delete('case');
        return next;
      });
    }
  };

  if (!isModalOpen || !activeModalCase) return null;

  // 10. Swipe-to-dismiss bottom sheet handler on mobile
  const handleDragEnd = (_e: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    if (info.offset.y > 140 || info.velocity.y > 500) {
      handleClose();
    }
  };

  return (
    <AnimatePresence>
      <div
        className="fixed inset-0 z-50 flex items-end md:items-center justify-center p-0 md:p-4 bg-black/60 backdrop-blur-sm"
        onClick={handleClose}
      >
        <motion.div
          initial={{ opacity: 0, y: 80, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 100, scale: 0.96 }}
          transition={{ type: 'spring', damping: 28, stiffness: 320 }}
          drag="y"
          dragConstraints={{ top: 0 }}
          dragElastic={{ top: 0, bottom: 0.4 }}
          onDragEnd={handleDragEnd}
          onClick={(e) => e.stopPropagation()}
          className="w-full max-w-4xl bg-white dark:bg-zinc-900 rounded-t-[36px] md:rounded-squircle shadow-2xl border border-brand-border dark:border-zinc-800 overflow-hidden flex flex-col max-h-[92dvh] touch-pan-y"
        >
          {/* Mobile Sheet Drag Handle */}
          <div className="md:hidden flex justify-center pt-3 pb-1 cursor-grab active:cursor-grabbing">
            <div className="w-12 h-1.5 bg-zinc-300 dark:bg-zinc-700 rounded-full" />
          </div>

          {/* Header */}
          <div className="px-6 py-4 border-b border-brand-border dark:border-zinc-800 flex items-center justify-between bg-surface-soft dark:bg-zinc-800/40">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-lavender-100 dark:bg-lavender-900/40 text-lavender-700 dark:text-lavender-300 flex items-center justify-center font-bold">
                <Layers size={18} />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-bold text-zinc-900 dark:text-white">
                    {activeModalCase.title}
                  </h3>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-mint-100 text-mint-700 dark:bg-mint-900/40 dark:text-mint-300 tabular-nums">
                    Mean Dice: {(activeModalCase.diceMean * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="text-xs text-zinc-500 font-medium">
                  {activeModalCase.modality} • {activeModalCase.hospitalSource}
                </p>
              </div>
            </div>

            <button
              onClick={handleClose}
              className="w-9 h-9 rounded-full hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-400 hover:text-zinc-700 dark:hover:text-white flex items-center justify-center transition-colors hit-slop"
              aria-label="Close Modal"
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
                    className={`px-4 py-1.5 rounded-xl text-xs font-bold capitalize transition-all hit-slop ${
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
                <label className="flex items-center gap-1.5 cursor-pointer select-none hit-slop">
                  <input
                    type="checkbox"
                    checked={showWT}
                    onChange={(e) => setShowWT(e.target.checked)}
                    className="accent-mint-500 rounded"
                  />
                  <span className="font-semibold text-mint-600">Whole Tumor (WT)</span>
                </label>

                <label className="flex items-center gap-1.5 cursor-pointer select-none hit-slop">
                  <input
                    type="checkbox"
                    checked={showTC}
                    onChange={(e) => setShowTC(e.target.checked)}
                    className="accent-amber-500 rounded"
                  />
                  <span className="font-semibold text-amber-600">Tumor Core (TC)</span>
                </label>

                <label className="flex items-center gap-1.5 cursor-pointer select-none hit-slop">
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
              <MRIViewport
                caseId={activeModalCase.id}
                sliceIndex={Math.max(0, Math.min(31, sliceZ - 1))}
                maskType="gt"
                title="Neuroradiologist Ground Truth"
                badge={<span className="text-[11px] font-mono text-zinc-400">Manual Contours</span>}
                showWT={showWT}
                showTC={showTC}
                showET={showET}
                maskOpacity={maskOpacity}
                plane={plane.toUpperCase()}
              />

              {/* Right Viewport: Federated Prediction */}
              <MRIViewport
                caseId={activeModalCase.id}
                sliceIndex={Math.max(0, Math.min(31, sliceZ - 1))}
                maskType="pred"
                title="FedMed 3D U-Net Output"
                badge={
                  <span className="text-[11px] font-mono text-mint-600 font-bold tabular-nums">
                    Dice: {(activeModalCase.diceMean * 100).toFixed(1)}%
                  </span>
                }
                showWT={showWT}
                showTC={showTC}
                showET={showET}
                maskOpacity={maskOpacity}
                plane={plane.toUpperCase()}
              />
            </div>

            {/* Sliders Controls Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-4 rounded-2xl bg-surface-soft dark:bg-zinc-800/40 border border-brand-border">
              {/* Slice Depth Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-zinc-600 dark:text-zinc-300">Volumetric Slice (Z):</span>
                  <span className="font-mono text-lavender-600 font-bold tabular-nums">
                    {sliceZ} / {activeModalCase.depthSlices}
                  </span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={activeModalCase.depthSlices}
                  value={sliceZ}
                  onChange={(e) => setSliceZ(parseInt(e.target.value, 10))}
                  className="w-full accent-lavender-600 cursor-pointer hit-slop"
                />
              </div>

              {/* Mask Opacity Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-zinc-600 dark:text-zinc-300">Tumor Overlay Opacity:</span>
                  <span className="font-mono text-mint-600 font-bold tabular-nums">
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
                  className="w-full accent-mint-500 cursor-pointer hit-slop"
                />
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
