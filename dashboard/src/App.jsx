import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Overview from './components/Overview';
import TrainingCurves from './components/TrainingCurves';
import PrivacyPerformance from './components/PrivacyPerformance';
import AblationTable from './components/AblationTable';
import SegmentationViewer from './components/SegmentationViewer';
import ResilienceTimeline from './components/ResilienceTimeline';
import {
  fetchStatus,
  fetchHistory,
  fetchAblation,
  fetchCases,
  connectMetricsWebSocket,
} from './api';
import {
  LayoutDashboard,
  LineChart,
  Shield,
  Layers,
  Eye,
  Activity,
  RefreshCw,
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [ablationData, setAblationData] = useState([]);
  const [cases, setCases] = useState([]);
  const [connected, setConnected] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [loading, setLoading] = useState(true);

  // Initial data bootstrap
  const loadInitialData = async () => {
    try {
      setLoading(true);
      const [statusRes, historyRes, ablationRes, casesRes] = await Promise.allSettled([
        fetchStatus(),
        fetchHistory(),
        fetchAblation(),
        fetchCases(),
      ]);

      if (statusRes.status === 'fulfilled') setStatus(statusRes.value);
      if (historyRes.status === 'fulfilled') setHistory(historyRes.value);
      if (ablationRes.status === 'fulfilled') setAblationData(ablationRes.value);
      if (casesRes.status === 'fulfilled') setCases(casesRes.value);
    } catch (e) {
      console.error('Error loading initial dashboard data:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadInitialData();

    // Connect WebSocket
    const cleanupWs = connectMetricsWebSocket(
      (newMetric) => {
        // Append or replace metric in history
        setHistory((prev) => {
          const idx = prev.findIndex((m) => m.round === newMetric.round);
          if (idx >= 0) {
            const updated = [...prev];
            updated[idx] = newMetric;
            return updated;
          }
          return [...prev, newMetric];
        });

        // Update status state
        setStatus((prev) => ({
          ...(prev || {}),
          current_round: newMetric.round,
          best_global_dice: Math.max(
            prev?.best_global_dice || 0,
            newMetric.global_val_dice || 0
          ),
          privacy: {
            ...(prev?.privacy || {}),
            cumulative_epsilon: newMetric.privacy?.epsilon || prev?.privacy?.cumulative_epsilon,
          },
        }));
      },
      (isConnected, isReconnecting) => {
        setConnected(isConnected);
        setReconnecting(isReconnecting);
      }
    );

    return () => {
      cleanupWs();
    };
  }, []);

  const latestRound = history.length > 0 ? history[history.length - 1] : null;

  return (
    <div className="app-container">
      {/* Reconnecting banner if WebSocket disconnected */}
      {reconnecting && (
        <div
          style={{
            background: 'rgba(245, 158, 11, 0.2)',
            color: '#fbbf24',
            border: '1px solid rgba(245, 158, 11, 0.4)',
            borderRadius: '8px',
            padding: '8px 16px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '13px',
          }}
        >
          <RefreshCw size={14} className="pulse-dot" />
          <span>WebSocket disconnected. Attempting automatic reconnection...</span>
        </div>
      )}

      {/* Top Header */}
      <Header status={status} connected={connected} />

      {/* Navigation Tabs */}
      <nav className="tabs-nav">
        <button
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          <LayoutDashboard size={16} /> Overview
        </button>
        <button
          className={`tab-btn ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          <LineChart size={16} /> Training Curves
        </button>
        <button
          className={`tab-btn ${activeTab === 'privacy' ? 'active' : ''}`}
          onClick={() => setActiveTab('privacy')}
        >
          <Shield size={16} /> Privacy & Performance
        </button>
        <button
          className={`tab-btn ${activeTab === 'ablation' ? 'active' : ''}`}
          onClick={() => setActiveTab('ablation')}
        >
          <Layers size={16} /> Ablation Benchmarks
        </button>
        <button
          className={`tab-btn ${activeTab === 'segmentation' ? 'active' : ''}`}
          onClick={() => setActiveTab('segmentation')}
        >
          <Eye size={16} /> 3D Segmentation Viewer
        </button>
        <button
          className={`tab-btn ${activeTab === 'resilience' ? 'active' : ''}`}
          onClick={() => setActiveTab('resilience')}
        >
          <Activity size={16} /> Resilience & Quorum
        </button>
      </nav>

      {/* Main Tab Content */}
      <main>
        {activeTab === 'overview' && (
          <Overview status={status} latestRoundData={latestRound} />
        )}

        {activeTab === 'training' && (
          <TrainingCurves history={history} />
        )}

        {activeTab === 'privacy' && (
          <PrivacyPerformance history={history} status={status} />
        )}

        {activeTab === 'ablation' && (
          <AblationTable ablationData={ablationData} loading={loading} />
        )}

        {activeTab === 'segmentation' && (
          <SegmentationViewer cases={cases} loading={loading} />
        )}

        {activeTab === 'resilience' && (
          <ResilienceTimeline history={history} />
        )}
      </main>
    </div>
  );
}
