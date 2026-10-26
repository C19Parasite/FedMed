import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Palette,
  Check,
  Eye,
  RotateCcw,
  Sparkles,
  ShieldCheck,
  Layers,
  Activity,
  CheckCircle2,
} from 'lucide-react';
import { toast } from 'sonner';
import { useDashboardStore } from '../store/useDashboardStore';
import { THEMES, LOGO_VARIANTS, ThemeId, LogoId, ThemeDefinition } from '../types/theme';
import { BrandLogo } from './BrandLogo';

export const ThemeSettingsModal: React.FC = () => {
  const {
    isSettingsModalOpen,
    closeSettingsModal,
    activeThemeId,
    previewThemeId,
    setPreviewTheme,
    applyTheme,
    cancelThemePreview,
    resetThemeToDefault,
    activeLogoId,
    previewLogoId,
    setPreviewLogo,
    applyLogo,
    cancelLogoPreview,
  } = useDashboardStore();

  const [activeTab, setActiveTab] = useState<'themes' | 'logos'>('themes');

  if (!isSettingsModalOpen) return null;

  const currentViewingThemeId = previewThemeId || activeThemeId;
  const currentViewingLogoId = previewLogoId || activeLogoId;

  const handleApplyTheme = (themeId: ThemeId) => {
    applyTheme(themeId);
    toast.success(`Theme Applied: ${THEMES[themeId].name}`, {
      description: 'Your color preferences have been saved and persisted.',
    });
  };

  const handleResetTheme = () => {
    resetThemeToDefault();
    toast.info('Theme Reset to Clinical Violet (Default)');
  };

  const handleApplyLogo = (logoId: LogoId) => {
    applyLogo(logoId);
    toast.success(`Brand Logo Updated: ${LOGO_VARIANTS[logoId].name}`, {
      description: 'Medical AI vector icon updated across sidebar and dashboard.',
    });
  };

  return (
    <AnimatePresence>
      <div
        className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-black/60 backdrop-blur-sm"
        onClick={closeSettingsModal}
        role="dialog"
        aria-modal="true"
        aria-labelledby="theme-settings-title"
      >
        <motion.div
          key="theme-settings-modal"
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 15 }}
          transition={{ type: 'spring', damping: 26, stiffness: 320 }}
          onClick={(e) => e.stopPropagation()}
          className="w-full max-w-5xl bg-white dark:bg-zinc-900 rounded-[28px] sm:rounded-[36px] shadow-2xl border border-brand-border dark:border-zinc-800 overflow-hidden flex flex-col max-h-[92dvh]"
        >
          {/* Header */}
          <div className="px-6 py-4 border-b border-brand-border dark:border-zinc-800 flex items-center justify-between bg-surface-soft dark:bg-zinc-800/40">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-2xl bg-lavender-100 dark:bg-lavender-900/40 text-lavender-700 dark:text-lavender-300 flex items-center justify-center font-bold">
                <Palette size={20} />
              </div>
              <div>
                <h3 id="theme-settings-title" className="text-base font-bold text-zinc-900 dark:text-white">
                  Appearance & Brand Identity
                </h3>
                <p className="text-xs text-zinc-500 font-medium">
                  Preview and personalize the FedMed Consortium color system and medical AI icon
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* Tabs Switcher */}
              <div className="flex p-1 bg-zinc-100 dark:bg-zinc-800 rounded-2xl border border-brand-border text-xs font-semibold">
                <button
                  onClick={() => setActiveTab('themes')}
                  className={`px-3 py-1.5 rounded-xl transition-all cursor-pointer ${
                    activeTab === 'themes'
                      ? 'bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white shadow-sm'
                      : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'
                  }`}
                >
                  Color Themes (6)
                </button>
                <button
                  onClick={() => setActiveTab('logos')}
                  className={`px-3 py-1.5 rounded-xl transition-all cursor-pointer ${
                    activeTab === 'logos'
                      ? 'bg-white dark:bg-zinc-900 text-zinc-900 dark:text-white shadow-sm'
                      : 'text-zinc-500 hover:text-zinc-900 dark:hover:text-white'
                  }`}
                >
                  Brand Icon (3)
                </button>
              </div>

              <button
                onClick={closeSettingsModal}
                className="w-9 h-9 rounded-full hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-400 hover:text-zinc-700 dark:hover:text-white flex items-center justify-center transition-colors hit-slop cursor-pointer"
                aria-label="Close Settings"
              >
                <X size={18} />
              </button>
            </div>
          </div>

          {/* Modal Body */}
          <div className="p-6 overflow-y-auto space-y-6 flex-1">
            {activeTab === 'themes' ? (
              <>
                {/* Clinical Semantic Safety Callout Banner */}
                <div className="px-4 py-2.5 rounded-2xl bg-lavender-50 dark:bg-lavender-950/30 border border-lavender-200 dark:border-lavender-900/50 flex flex-col sm:flex-row sm:items-center justify-between text-xs text-lavender-900 dark:text-lavender-300 gap-2">
                  <div className="flex items-center gap-2 font-medium">
                    <ShieldCheck size={16} className="text-lavender-600 shrink-0" />
                    <span>
                      <strong>Clinical Integrity Guaranteed:</strong> BraTS tumor overlays (WT Green, TC Orange, ET Red) and grayscale MRI volumes remain diagnostic standard across all themes.
                    </span>
                  </div>
                  {previewThemeId && (
                    <span className="shrink-0 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-lavender-200 dark:bg-lavender-800 text-lavender-900 dark:text-lavender-100 animate-pulse">
                      Live Preview Active
                    </span>
                  )}
                </div>

                {/* 6 Theme Cards Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                  {(Object.keys(THEMES) as ThemeId[]).map((themeKey) => {
                    const t = THEMES[themeKey];
                    const isSelected = currentViewingThemeId === themeKey;
                    const isConfirmedActive = activeThemeId === themeKey;
                    const isLivePreviewing = previewThemeId === themeKey;

                    return (
                      <div
                        key={themeKey}
                        onClick={() => setPreviewTheme(themeKey)}
                        className={`relative rounded-3xl p-4 transition-all duration-200 cursor-pointer flex flex-col justify-between border-2 select-none group ${
                          isSelected
                            ? 'border-lavender-600 dark:border-lavender-400 shadow-xl bg-surface-soft dark:bg-zinc-800/80 scale-[1.01]'
                            : 'border-brand-border dark:border-zinc-800/80 bg-white dark:bg-zinc-800/40 hover:border-lavender-300 dark:hover:border-zinc-700 hover:shadow-md'
                        }`}
                      >
                        {/* Miniature Dashboard Mock Representation */}
                        <div
                          className="w-full h-32 rounded-2xl p-2.5 mb-3.5 flex gap-2 overflow-hidden border shadow-inner transition-transform group-hover:scale-[1.02]"
                          style={{
                            backgroundColor: t.colors.background,
                            borderColor: t.colors.cardBorder,
                          }}
                        >
                          {/* Mini Sidebar */}
                          <div
                            className="w-7 h-full rounded-xl p-1 flex flex-col items-center justify-between shrink-0 shadow-sm"
                            style={{ backgroundColor: t.colors.sidebarBg }}
                          >
                            <div
                              className="w-4 h-4 rounded-md flex items-center justify-center text-[8px]"
                              style={{ backgroundColor: t.colors.primary }}
                            >
                              <div className="w-1.5 h-1.5 rounded-full bg-white" />
                            </div>
                            <div className="flex flex-col gap-1 items-center">
                              <div
                                className="w-4 h-4 rounded-md flex items-center justify-center"
                                style={{ backgroundColor: `${t.colors.primary}33` }}
                              >
                                <div className="w-1.5 h-1.5 rounded-sm" style={{ backgroundColor: t.colors.primary }} />
                              </div>
                              <div className="w-2.5 h-1 rounded-full bg-white/20" />
                              <div className="w-2.5 h-1 rounded-full bg-white/20" />
                            </div>
                            <div className="w-2.5 h-2.5 rounded-full bg-white/20" />
                          </div>

                          {/* Mini Content Area */}
                          <div className="flex-1 flex flex-col justify-between min-w-0">
                            {/* Mini Header */}
                            <div className="flex items-center justify-between pb-1">
                              <div className="flex items-center gap-1">
                                <div
                                  className="w-12 h-2 rounded-full"
                                  style={{ backgroundColor: t.colors.text }}
                                />
                                <div
                                  className="w-7 h-2 rounded-full"
                                  style={{ backgroundColor: `${t.colors.primary}33` }}
                                />
                              </div>
                              <div
                                className="w-8 h-2.5 rounded-full"
                                style={{ backgroundColor: t.colors.primary }}
                              />
                            </div>

                            {/* Mini Hospital Node Cards Row */}
                            <div className="grid grid-cols-2 gap-1.5">
                              <div
                                className="p-1 rounded-lg border shadow-xs"
                                style={{
                                  backgroundColor: t.colors.card,
                                  borderColor: t.colors.cardBorder,
                                }}
                              >
                                <div className="flex justify-between items-center mb-0.5">
                                  <div className="w-7 h-1.5 rounded-full" style={{ backgroundColor: t.colors.text }} />
                                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: t.colors.accent }} />
                                </div>
                                <div className="w-10 h-1 rounded-full" style={{ backgroundColor: t.colors.textMuted }} />
                              </div>

                              <div
                                className="p-1 rounded-lg border shadow-xs"
                                style={{
                                  backgroundColor: t.colors.card,
                                  borderColor: t.colors.cardBorder,
                                }}
                              >
                                <div className="flex justify-between items-center mb-0.5">
                                  <div className="w-8 h-1.5 rounded-full" style={{ backgroundColor: t.colors.text }} />
                                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: t.colors.primary }} />
                                </div>
                                <div className="w-6 h-1 rounded-full" style={{ backgroundColor: t.colors.textMuted }} />
                              </div>
                            </div>

                            {/* Mini Chart Curve in Primary Theme Color */}
                            <div
                              className="w-full h-8 rounded-lg p-1 border flex items-end justify-between px-1.5 shadow-xs"
                              style={{
                                backgroundColor: t.colors.card,
                                borderColor: t.colors.cardBorder,
                              }}
                            >
                              <svg className="w-full h-5 overflow-visible" viewBox="0 0 100 20">
                                <path
                                  d="M0 16 Q25 14, 45 9 T75 6 T100 2"
                                  fill="none"
                                  stroke={t.colors.primary}
                                  strokeWidth="2.5"
                                  strokeLinecap="round"
                                />
                                <circle cx="100" cy="2" r="2.5" fill={t.colors.accent} />
                              </svg>
                            </div>
                          </div>
                        </div>

                        {/* Title, Style & Swatches */}
                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <h4 className="text-sm font-bold text-zinc-900 dark:text-white">
                              {t.name}
                            </h4>
                            <span
                              className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                t.isDark
                                  ? 'bg-zinc-800 text-zinc-300 border border-zinc-700'
                                  : 'bg-zinc-100 text-zinc-700 border border-zinc-200'
                              }`}
                            >
                              {t.isDark ? 'Dark Mode' : 'Light UI'}
                            </span>
                          </div>

                          <p className="text-[11px] text-zinc-500 dark:text-zinc-400 line-clamp-2 mb-3">
                            {t.tagline}
                          </p>

                          {/* 6 Color Swatches */}
                          <div className="space-y-1">
                            <div className="flex items-center justify-between text-[10px] text-zinc-400 font-medium">
                              <span>Palette Swatches</span>
                              <span className="font-mono">{t.colors.primary}</span>
                            </div>
                            <div className="flex h-5 rounded-xl overflow-hidden border border-brand-border/60">
                              <div
                                className="flex-1"
                                style={{ backgroundColor: t.colors.primary }}
                                title={`Primary: ${t.colors.primary}`}
                              />
                              <div
                                className="flex-1"
                                style={{ backgroundColor: t.colors.secondary }}
                                title={`Secondary: ${t.colors.secondary}`}
                              />
                              <div
                                className="flex-1"
                                style={{ backgroundColor: t.colors.background }}
                                title={`Background: ${t.colors.background}`}
                              />
                              <div
                                className="flex-1"
                                style={{ backgroundColor: t.colors.card }}
                                title={`Card: ${t.colors.card}`}
                              />
                              <div
                                className="flex-1"
                                style={{ backgroundColor: t.colors.text }}
                                title={`Text: ${t.colors.text}`}
                              />
                              <div
                                className="flex-1"
                                style={{ backgroundColor: t.colors.accent }}
                                title={`Accent: ${t.colors.accent}`}
                              />
                            </div>
                          </div>
                        </div>

                        {/* Card Action / Status Bar */}
                        <div className="pt-3 mt-3 border-t border-brand-border/60 dark:border-zinc-800 flex items-center justify-between">
                          <div className="flex items-center gap-1.5">
                            {isConfirmedActive ? (
                              <span className="inline-flex items-center gap-1 text-[11px] font-bold text-mint-600 dark:text-mint-400">
                                <Check size={13} strokeWidth={3} />
                                Active
                              </span>
                            ) : isLivePreviewing ? (
                              <span className="inline-flex items-center gap-1 text-[11px] font-bold text-lavender-600 dark:text-lavender-400">
                                <Eye size={13} />
                                Previewing
                              </span>
                            ) : (
                              <span className="text-[11px] text-zinc-400 group-hover:text-zinc-600 dark:group-hover:text-zinc-300 transition-colors">
                                Click to preview
                              </span>
                            )}
                          </div>

                          <div className="flex items-center gap-1.5">
                            {isSelected && !isConfirmedActive && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleApplyTheme(themeKey);
                                }}
                                className="px-3 py-1 rounded-xl text-xs font-bold bg-lavender-600 hover:bg-lavender-700 text-white shadow-sm transition-all"
                              >
                                Apply
                              </button>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            ) : (
              /* Brand Icon Selection Section */
              <div className="space-y-6">
                <div className="px-4 py-3 rounded-2xl bg-surface-soft dark:bg-zinc-800/40 border border-brand-border text-xs text-zinc-600 dark:text-zinc-300">
                  <h4 className="font-bold text-zinc-900 dark:text-white mb-1 flex items-center gap-1.5">
                    <Sparkles size={14} className="text-lavender-600" />
                    Custom Medical AI Identity Vector
                  </h4>
                  <p>
                    Replaced generic emoji with a razor-sharp vector mark. Choose between neural network topology, silicon bio-informatics circuitry, or clinical intelligence cross motifs.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                  {(Object.keys(LOGO_VARIANTS) as LogoId[]).map((logoKey) => {
                    const l = LOGO_VARIANTS[logoKey];
                    const isSelected = currentViewingLogoId === logoKey;
                    const isConfirmed = activeLogoId === logoKey;

                    return (
                      <div
                        key={logoKey}
                        onClick={() => setPreviewLogo(logoKey)}
                        className={`rounded-3xl p-5 border-2 transition-all cursor-pointer flex flex-col justify-between select-none ${
                          isSelected
                            ? 'border-lavender-600 dark:border-lavender-400 shadow-xl bg-surface-soft dark:bg-zinc-800/80 scale-[1.01]'
                            : 'border-brand-border dark:border-zinc-800 bg-white dark:bg-zinc-800/40 hover:border-lavender-300'
                        }`}
                      >
                        {/* Realistic Sidebar Logo Container Mock */}
                        <div className="w-full py-6 flex flex-col items-center justify-center bg-zinc-950 rounded-2xl mb-4 border border-zinc-800/80">
                          <div className="relative w-14 h-14 rounded-2xl bg-gradient-to-tr from-lavender-600 to-lavender-400 flex items-center justify-center shadow-glow-lavender text-white">
                            <BrandLogo variant={logoKey} size={28} />
                            {/* Green availability indicator */}
                            <span className="absolute -bottom-1 -right-1 w-3 h-3 bg-mint-400 border-2 border-surface-dark rounded-full" />
                          </div>
                          <span className="mt-2 text-[10px] font-extrabold uppercase tracking-wider text-lavender-300">
                            FedMed
                          </span>
                        </div>

                        <div>
                          <div className="flex items-center justify-between mb-1">
                            <h4 className="text-base font-bold text-zinc-900 dark:text-white">
                              {l.name}
                            </h4>
                            {isConfirmed && (
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-mint-100 text-mint-700 dark:bg-mint-900/40 dark:text-mint-300">
                                Active Logo
                              </span>
                            )}
                          </div>
                          <p className="text-xs font-semibold text-lavender-600 dark:text-lavender-400 mb-1.5">
                            {l.subtitle}
                          </p>
                          <p className="text-xs text-zinc-500 dark:text-zinc-400 leading-relaxed">
                            {l.description}
                          </p>
                        </div>

                        <div className="pt-4 mt-4 border-t border-brand-border/60 dark:border-zinc-800 flex items-center justify-between">
                          <span className="text-xs text-zinc-400">
                            {isSelected ? 'Selected' : 'Click to select'}
                          </span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleApplyLogo(logoKey);
                            }}
                            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold transition-all ${
                              isConfirmed
                                ? 'bg-zinc-100 dark:bg-zinc-800 text-zinc-500 cursor-default'
                                : 'bg-lavender-600 hover:bg-lavender-700 text-white shadow-sm'
                            }`}
                          >
                            {isConfirmed ? 'Current' : 'Apply Icon'}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Bottom Action Toolbar */}
          <div className="px-6 py-4 border-t border-brand-border dark:border-zinc-800 bg-surface-soft dark:bg-zinc-800/40 flex flex-wrap items-center justify-between gap-3">
            <button
              onClick={handleResetTheme}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-semibold text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors cursor-pointer"
            >
              <RotateCcw size={14} />
              <span>Reset to Default (Theme A)</span>
            </button>

            <div className="flex items-center gap-2.5">
              <button
                onClick={closeSettingsModal}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors cursor-pointer"
              >
                Cancel / Revert
              </button>

              <button
                onClick={() => {
                  handleApplyTheme(currentViewingThemeId);
                  if (activeTab === 'logos') {
                    handleApplyLogo(currentViewingLogoId);
                  }
                  closeSettingsModal();
                }}
                className="flex items-center gap-1.5 px-5 py-2 rounded-xl text-xs font-bold bg-lavender-600 hover:bg-lavender-700 text-white shadow-soft hover:shadow-lift transition-all cursor-pointer"
              >
                <Check size={14} strokeWidth={2.5} />
                <span>Confirm & Apply {activeTab === 'themes' ? THEMES[currentViewingThemeId].name.split('—')[1] : LOGO_VARIANTS[currentViewingLogoId].name}</span>
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
