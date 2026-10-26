import React from 'react';

export const Skeleton: React.FC<{ className?: string; style?: React.CSSProperties }> = ({
  className = '',
  style,
}) => {
  return (
    <div
      className={`skeleton-shimmer rounded-2xl ${className}`}
      style={style}
      aria-hidden="true"
    />
  );
};

export const HeroSkeleton: React.FC = () => (
  <div className="h-44 w-full rounded-3xl skeleton-shimmer p-6" />
);

export const NodeRowSkeleton: React.FC = () => (
  <div className="card-base p-6 space-y-4">
    <div className="flex justify-between items-center">
      <Skeleton className="h-6 w-36" />
      <Skeleton className="h-6 w-6 rounded-lg" />
    </div>
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <Skeleton className="h-28 w-full rounded-2xl" />
      <Skeleton className="h-28 w-full rounded-2xl" />
      <Skeleton className="h-28 w-full rounded-2xl" />
    </div>
  </div>
);

export const MetricTilesSkeleton: React.FC = () => (
  <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
    <div className="card-base p-6 h-40 flex flex-col justify-between">
      <div className="flex justify-between">
        <Skeleton className="h-4 w-28" />
        <Skeleton className="h-8 w-8 rounded-xl" />
      </div>
      <Skeleton className="h-10 w-32" />
      <Skeleton className="h-4 w-full" />
    </div>
    <div className="card-base p-6 h-40 flex flex-col justify-between">
      <div className="flex justify-between">
        <Skeleton className="h-4 w-28" />
        <Skeleton className="h-8 w-8 rounded-xl" />
      </div>
      <Skeleton className="h-10 w-24" />
      <Skeleton className="h-4 w-full" />
    </div>
    <div className="card-base p-6 h-40 flex flex-col justify-between">
      <div className="flex justify-between">
        <Skeleton className="h-4 w-28" />
        <Skeleton className="h-8 w-8 rounded-xl" />
      </div>
      <Skeleton className="h-10 w-24" />
      <Skeleton className="h-4 w-full" />
    </div>
  </div>
);

export const ChartSkeleton: React.FC = () => (
  <div className="card-base p-6 space-y-4">
    <div className="flex justify-between items-center pb-4 border-b border-brand-border/60">
      <Skeleton className="h-6 w-44" />
      <Skeleton className="h-8 w-36 rounded-xl" />
    </div>
    <div className="grid grid-cols-3 gap-4 my-3">
      <Skeleton className="h-12 w-full rounded-xl" />
      <Skeleton className="h-12 w-full rounded-xl" />
      <Skeleton className="h-12 w-full rounded-xl" />
    </div>
    <Skeleton className="h-64 w-full rounded-2xl" />
  </div>
);

export const TableSkeleton: React.FC<{ rows?: number }> = ({ rows = 4 }) => (
  <div className="card-base p-6 space-y-3">
    <div className="flex justify-between mb-4">
      <Skeleton className="h-6 w-48" />
      <Skeleton className="h-6 w-24 rounded-full" />
    </div>
    {Array.from({ length: rows }).map((_, i) => (
      <Skeleton key={i} className="h-12 w-full rounded-xl" />
    ))}
  </div>
);

export const GallerySkeleton: React.FC = () => (
  <div className="card-base p-5 space-y-3">
    <div className="flex justify-between">
      <Skeleton className="h-5 w-40" />
      <Skeleton className="h-4 w-12" />
    </div>
    <div className="flex gap-3 overflow-hidden">
      <Skeleton className="min-w-[180px] h-48 rounded-2xl" />
      <Skeleton className="min-w-[180px] h-48 rounded-2xl" />
      <Skeleton className="min-w-[180px] h-48 rounded-2xl" />
    </div>
  </div>
);
