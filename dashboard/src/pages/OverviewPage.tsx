import React from 'react';
import { HeroBanner } from '../components/HeroBanner';
import { HospitalNodesRow } from '../components/HospitalNodesRow';
import { MetricTiles } from '../components/MetricTiles';
import { TrainingStatistics } from '../components/TrainingStatistics';
import { PrivacyBudgetCard } from '../components/PrivacyBudgetCard';
import { RoundSelector } from '../components/RoundSelector';
import { SegmentationCarousel } from '../components/SegmentationCarousel';
import { NotificationFeed } from '../components/NotificationFeed';

export const OverviewPage: React.FC = () => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Center Main Column (8 cols on large screens) */}
      <div className="lg:col-span-8 flex flex-col gap-6">
        {/* a) Hero Banner */}
        <HeroBanner />

        {/* b) Hospital Nodes Card Row */}
        <HospitalNodesRow />

        {/* c) Three Metric Tiles */}
        <MetricTiles />

        {/* d) Training Statistics Recharts Area Chart */}
        <TrainingStatistics />

        {/* e) Privacy Budget Promo Card */}
        <PrivacyBudgetCard />
      </div>

      {/* Right Sidebar Column (4 cols on large screens) */}
      <div className="lg:col-span-4 flex flex-col gap-6">
        {/* a) Calendar-style round selector */}
        <RoundSelector />

        {/* b) Latest Segmentation Results carousel */}
        <SegmentationCarousel />

        {/* c) Live Notifications feed */}
        <NotificationFeed />
      </div>
    </div>
  );
};
