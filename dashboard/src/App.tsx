import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster, toast } from 'sonner';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { SegmentationModal } from './components/SegmentationModal';
import { CommandMenu } from './components/CommandMenu';
import { KeyboardShortcutsModal } from './components/KeyboardShortcutsModal';
import { TwoFactorAuthModal } from './components/TwoFactorAuthModal';
import { ThemeSettingsModal } from './components/ThemeSettingsModal';
import { OverviewPage } from './pages/OverviewPage';
import { NodesPage } from './pages/NodesPage';
import { RoundsPage } from './pages/RoundsPage';
import { SegmentationPage } from './pages/SegmentationPage';
import { PrivacyPage } from './pages/PrivacyPage';
import { useTrainingStream } from './hooks/useTrainingStream';
import { useDashboardStore } from './store/useDashboardStore';

// Dedicated TanStack Query Client with Stale-While-Revalidate Defaults
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      refetchOnWindowFocus: false,
    },
  },
});

export const App: React.FC = () => {
  // Start the background training stream (auto-detects real WS vs simulated mock feed)
  useTrainingStream();

  const { is2FAModalOpen, close2FAModal, startNewRound, announcement } = useDashboardStore();

  const handle2FASuccess = () => {
    close2FAModal();
    startNewRound();
    toast.success('2FA Verified: New Federated Round Dispatched', {
      description: 'Synchronous FedAvg aggregation running with CKKS homomorphic encryption.',
    });
  };

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-[#E9E7F2] dark:bg-zinc-950 p-2 sm:p-4 md:p-6 lg:p-8 flex justify-center transition-colors duration-300">
        {/* 32px Rounded App Shell */}
        <div className="w-full max-w-[1600px] bg-shell dark:bg-zinc-900 rounded-[28px] sm:rounded-[36px] p-3 sm:p-5 md:p-7 shadow-2xl border border-white/70 dark:border-zinc-800/80 flex flex-col md:flex-row gap-5 relative">
          {/* 1. Left Slim Dark Pill Sidebar (Sticky & constant on desktop; bottom nav on mobile) */}
          <Sidebar />

          {/* 2 & 3. Center & Right Content Container (scrolls naturally with page) */}
          <div className="flex-1 flex flex-col gap-6 min-w-0 pb-20 md:pb-6">
            {/* Top Global Header Bar */}
            <Header />

            {/* Router Views */}
            <main className="flex-1">
              <Routes>
                <Route path="/" element={<Navigate to="/overview" replace />} />
                <Route path="/overview" element={<OverviewPage />} />
                <Route path="/nodes" element={<NodesPage />} />
                <Route path="/rounds" element={<RoundsPage />} />
                <Route path="/segmentation" element={<SegmentationPage />} />
                <Route path="/privacy" element={<PrivacyPage />} />
                <Route path="*" element={<Navigate to="/overview" replace />} />
              </Routes>
            </main>
          </div>

          {/* Global Interactive DICOM / Segmentation Modal */}
          <SegmentationModal />

          {/* Global Cmd/Ctrl+K Command Menu */}
          <CommandMenu />

          {/* Global '?' Keyboard Shortcuts Modal */}
          <KeyboardShortcutsModal />

          {/* Sensitive Action 6-Digit 2FA Modal */}
          <TwoFactorAuthModal
            isOpen={is2FAModalOpen}
            onClose={close2FAModal}
            onSuccess={handle2FASuccess}
            title="Authorize Federated Training Round"
          />

          {/* Interactive Appearance, Color Theme & Brand Logo Modal */}
          <ThemeSettingsModal />

          {/* Live Region Announcements for Screen Readers (Item 16) */}
          <div
            className="sr-only"
            aria-live="polite"
            aria-atomic="true"
          >
            {announcement}
          </div>

          {/* Toast Notification Container with Undo progress bars */}
          <Toaster
            position="top-right"
            richColors
            closeButton
            toastOptions={{
              duration: 5000,
              className: 'rounded-2xl font-sans',
            }}
          />
        </div>
      </div>
    </QueryClientProvider>
  );
};

export default App;
