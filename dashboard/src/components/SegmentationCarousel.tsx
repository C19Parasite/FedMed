import React from 'react';
import { Scan, Eye, ChevronRight, Award } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDashboardStore } from '../store/useDashboardStore';
import { SegmentationCase } from '../types';

export const SegmentationCarousel: React.FC = () => {
  const { cases, openModal } = useDashboardStore();

  return (
    <div className="card-base p-5 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800">
      {/* Header */}
      <div className="flex items-center justify-between mb-3.5">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-mint-100 dark:bg-mint-900/40 text-mint-700 dark:text-mint-300 flex items-center justify-center">
            <Scan size={14} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white leading-tight">
              Latest Segmentation Results
            </h3>
            <p className="text-[11px] text-zinc-400">
              Held-out BraTS evaluation slices • Click to inspect
            </p>
          </div>
        </div>

        <span className="text-xs font-semibold text-lavender-600 dark:text-lavender-400">
          {cases.length} Cases
        </span>
      </div>

      {/* Horizontal Scrollable Thumbnails */}
      <div className="flex gap-3 overflow-x-auto no-scrollbar py-1">
        {cases.map((caseItem, idx) => (
          <motion.div
            key={caseItem.id}
            whileHover={{ y: -3 }}
            onClick={() => openModal(caseItem)}
            className="min-w-[170px] sm:min-w-[190px] rounded-2xl p-2.5 bg-surface-soft dark:bg-zinc-800/60 border border-brand-border dark:border-zinc-700 hover:border-lavender-400 dark:hover:border-lavender-500 cursor-pointer transition-all duration-200 group flex flex-col justify-between"
          >
            {/* Simulated MRI Slice with Tumor Mask Overlay Thumbnail */}
            <div className="relative w-full aspect-square bg-[#07090E] rounded-xl overflow-hidden border border-zinc-800 flex items-center justify-center">
              {/* BlurHash Micro-placeholder */}
              {caseItem.blurHash && (
                <div
                  className="absolute inset-0 bg-cover bg-center filter blur-sm scale-110 opacity-70"
                  style={{ backgroundImage: `url(${caseItem.blurHash})` }}
                  aria-hidden="true"
                />
              )}
              {/* Brain MRI Simulated Background */}
              <div className="absolute inset-2 rounded-full bg-gradient-to-tr from-zinc-800 via-zinc-700 to-zinc-900 opacity-80" />
              <div className="absolute w-20 h-24 rounded-full bg-zinc-600/40 blur-[2px]" />

              {/* Simulated Tumor Mask Contour Overlay */}
              <div
                className="absolute w-10 h-10 rounded-full border-2 border-mint-400 bg-mint-500/30 blur-[1px] transform translate-x-2 -translate-y-1"
                title="Whole Tumor (WT)"
              />
              <div
                className="absolute w-5 h-5 rounded-full border border-coral-400 bg-coral-500/40 transform translate-x-2.5 -translate-y-0.5"
                title="Tumor Core (TC)"
              />

              {/* View Overlay on Hover */}
              <div className="absolute inset-0 bg-lavender-900/60 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity text-white text-xs font-semibold gap-1 backdrop-blur-[1px]">
                <Eye size={14} />
                <span>Inspect</span>
              </div>

              {/* Slice Tag */}
              <span className="absolute bottom-1.5 left-1.5 px-1.5 py-0.5 rounded-md bg-black/70 text-[9px] font-mono text-zinc-300">
                Z: {caseItem.sliceZ}
              </span>
            </div>

            {/* Title & Metadata */}
            <div className="mt-2.5 px-0.5">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-zinc-900 dark:text-white truncate">
                  {caseItem.title}
                </h4>
                <span className="text-[10px] font-bold text-mint-600 dark:text-mint-400 font-mono">
                  {(caseItem.diceMean * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-[10px] text-zinc-400 truncate mt-0.5 font-medium">
                {caseItem.hospitalSource.split('(')[0].trim()}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};
