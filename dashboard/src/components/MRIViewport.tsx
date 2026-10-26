import React, { useState } from 'react';
import { Loader2, AlertCircle } from 'lucide-react';

interface MRIViewportProps {
  caseId: string;
  sliceIndex: number; // 0 to 31
  maskType: 'gt' | 'pred';
  title: string;
  badge?: React.ReactNode;
  showWT?: boolean;
  showTC?: boolean;
  showET?: boolean;
  maskOpacity?: number;
  modality?: string; // 'flair' | 't1' | 't1ce' | 't2'
  plane?: string;
  className?: string;
}

export const MRIViewport: React.FC<MRIViewportProps> = ({
  caseId,
  sliceIndex,
  maskType,
  title,
  badge,
  showWT = true,
  showTC = true,
  showET = true,
  maskOpacity = 0.7,
  modality = 'flair',
  plane = 'AXIAL',
  className = '',
}) => {
  const [isMriLoaded, setIsMriLoaded] = useState<boolean>(false);
  const [hasMriError, setHasMriError] = useState<boolean>(false);

  // Normalize slice index bounds (0-31)
  const z = Math.max(0, Math.min(31, sliceIndex));
  const mod = modality.toLowerCase();

  // Primary URL path (served from Vite /public/cases or FastAPI /api/cases)
  const mriUrl = `/cases/${caseId}/slice_${z}_mri_${mod}.png`;
  const mriFallbackUrl = `/cases/${caseId}/slice_${z}_mri.png`;

  const allSelected = showWT && showTC && showET;
  const anySelected = showWT || showTC || showET;

  return (
    <div className={`flex flex-col items-center w-full ${className}`}>
      {/* Header Info */}
      <div className="w-full flex items-center justify-between mb-2 px-1 text-xs">
        <span className="font-bold text-zinc-700 dark:text-zinc-200 uppercase tracking-wider">
          {title}
        </span>
        {badge}
      </div>

      {/* Main Viewport Container */}
      <div className="relative w-full aspect-square bg-[#05070B] rounded-2xl md:rounded-3xl overflow-hidden border border-zinc-800 shadow-inner flex items-center justify-center select-none">
        {/* Loading Spinner */}
        {!isMriLoaded && !hasMriError && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/70 backdrop-blur-[2px] text-zinc-400 gap-2">
            <Loader2 size={24} className="animate-spin text-lavender-500" />
            <span className="text-[11px] font-mono">Loading MRI Slice {z + 1}…</span>
          </div>
        )}

        {/* Error Fallback */}
        {hasMriError && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center p-4 bg-zinc-950/90 text-center text-zinc-400 gap-2">
            <AlertCircle size={24} className="text-coral-500" />
            <span className="text-xs font-bold text-zinc-300">MRI Slice Unavailable</span>
            <span className="text-[10px] text-zinc-500 font-mono">
              Volume: {caseId} • Slice Z={z}
            </span>
          </div>
        )}

        {/* 1. Base Grayscale MRI Image (Always at 100% opacity, never dimmed by mask opacity slider) */}
        <img
          key={`mri-${caseId}-${z}-${mod}`}
          src={mriUrl}
          alt={`Real MRI ${mod.toUpperCase()} Slice ${z}`}
          onLoad={() => setIsMriLoaded(true)}
          onError={(e) => {
            // Try fallback without modality suffix
            const target = e.currentTarget;
            if (target.src !== mriFallbackUrl) {
              target.src = mriFallbackUrl;
            } else {
              setHasMriError(true);
            }
          }}
          className="w-full h-full object-contain filter contrast-125 brightness-95"
          style={{ imageRendering: 'pixelated' }}
        />

        {/* 2. Segmentation Mask Overlays (Only these layers respond to the maskOpacity slider) */}
        {isMriLoaded && !hasMriError && anySelected && (
          <div
            className="absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-150"
            style={{ opacity: maskOpacity }}
          >
            {allSelected ? (
              // Composite mask containing WT (Green), TC (Orange), ET (Red)
              <img
                key={`mask-comp-${caseId}-${z}-${maskType}`}
                src={`/cases/${caseId}/slice_${z}_${maskType}.png`}
                alt="Composite Tumor Mask"
                className="w-full h-full object-contain"
                style={{ imageRendering: 'pixelated' }}
              />
            ) : (
              // Individual Subregion Overlays
              <>
                {showWT && (
                  <img
                    key={`mask-wt-${caseId}-${z}-${maskType}`}
                    src={`/cases/${caseId}/slice_${z}_${maskType}_wt.png`}
                    alt="Whole Tumor (WT) Mask"
                    className="absolute inset-0 w-full h-full object-contain"
                    style={{ imageRendering: 'pixelated' }}
                  />
                )}
                {showTC && (
                  <img
                    key={`mask-tc-${caseId}-${z}-${maskType}`}
                    src={`/cases/${caseId}/slice_${z}_${maskType}_tc.png`}
                    alt="Tumor Core (TC) Mask"
                    className="absolute inset-0 w-full h-full object-contain"
                    style={{ imageRendering: 'pixelated' }}
                  />
                )}
                {showET && (
                  <img
                    key={`mask-et-${caseId}-${z}-${maskType}`}
                    src={`/cases/${caseId}/slice_${z}_${maskType}_et.png`}
                    alt="Enhancing Tumor (ET) Mask"
                    className="absolute inset-0 w-full h-full object-contain"
                    style={{ imageRendering: 'pixelated' }}
                  />
                )}
              </>
            )}
          </div>
        )}

        {/* Slice Coordinates & Orientation Badge */}
        <div className="absolute top-2 left-2 z-10 px-2 py-0.5 rounded bg-black/75 backdrop-blur-sm text-[10px] font-mono text-zinc-300 border border-white/10 shadow-sm flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-mint-400" />
          <span>
            {plane} #{z + 1} • {mod.toUpperCase()}
          </span>
        </div>
      </div>
    </div>
  );
};
