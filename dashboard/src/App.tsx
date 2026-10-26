import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { SegmentationModal } from './components/SegmentationModal';
import { OverviewPage } from './pages/OverviewPage';
import { NodesPage } from './pages/NodesPage';
import { RoundsPage } from './pages/RoundsPage';
import { SegmentationPage } from './pages/SegmentationPage';
import { PrivacyPage } from './pages/PrivacyPage';
import { useTrainingStream } from './hooks/useTrainingStream';

export const App: React.FC = () => {
  // Start the background training stream (auto-detects real WS vs simulated mock feed)
  useTrainingStream();

  return (
    <div className="min-h-screen bg-[#E9E7F2] dark:bg-zinc-950 p-2 sm:p-4 md:p-6 lg:p-8 flex items-center justify-center transition-colors duration-300">
      {/* 32px Rounded App Shell */}
      <div className="w-full max-w-[1600px] min-h-[92vh] bg-shell dark:bg-zinc-900 rounded-[28px] sm:rounded-[36px] p-3 sm:p-5 md:p-7 shadow-2xl border border-white/70 dark:border-zinc-800/80 flex flex-col md:flex-row gap-5 relative overflow-hidden">
        {/* 1. Left Slim Dark Pill Sidebar */}
        <Sidebar />

        {/* 2 & 3. Center & Right Content Container */}
        <div className="flex-1 flex flex-col gap-6 min-w-0 overflow-y-auto">
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
      </div>
    </div>
  );
};

export default App;
