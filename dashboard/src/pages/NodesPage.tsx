import React from 'react';
import { Hospital, WifiOff, CheckCircle2, Server, Activity, ShieldCheck, RefreshCw } from 'lucide-react';
import { useDashboardStore } from '../store/useDashboardStore';

export const NodesPage: React.FC = () => {
  const { nodes } = useDashboardStore();

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="card-base p-6 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-extrabold text-zinc-900 dark:text-white">
            Hospital Nodes Topology & Health
          </h2>
          <p className="text-xs text-zinc-500 mt-1">
            Real-time status of on-premise compute nodes, sample allocations, network latency, and local gradient telemetry.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-3 py-1.5 rounded-full text-xs font-semibold bg-mint-100 text-mint-700 dark:bg-mint-950/60 dark:text-mint-300">
            {nodes.filter((n) => n.status !== 'offline').length} of {nodes.length} Nodes Active
          </span>
        </div>
      </div>

      {/* Nodes Detailed Table */}
      <div className="card-base p-6 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800 overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-brand-border/80 dark:border-zinc-800 text-zinc-400 font-bold uppercase tracking-wider">
              <th className="pb-3 px-3">Hospital Silo</th>
              <th className="pb-3 px-3">Location & IP</th>
              <th className="pb-3 px-3">Status</th>
              <th className="pb-3 px-3">Dataset Size</th>
              <th className="pb-3 px-3">Latency</th>
              <th className="pb-3 px-3">Gradient Norm</th>
              <th className="pb-3 px-3">FedAvg Weight</th>
              <th className="pb-3 px-3 text-right">TLS Handshake</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-brand-border/50 dark:divide-zinc-800/60">
            {nodes.map((node) => {
              const isOffline = node.status === 'offline';
              return (
                <tr
                  key={node.id}
                  className={`hover:bg-surface-soft dark:hover:bg-zinc-800/40 transition-colors ${
                    isOffline ? 'opacity-60 bg-zinc-50 dark:bg-zinc-900/30' : ''
                  }`}
                >
                  <td className="py-4 px-3">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-xl bg-lavender-100 dark:bg-lavender-900/40 text-lavender-700 dark:text-lavender-300 flex items-center justify-center font-bold">
                        <Hospital size={16} />
                      </div>
                      <div>
                        <span className="font-bold text-zinc-900 dark:text-white block">
                          {node.name}
                        </span>
                        <span className="text-[11px] text-zinc-400 font-mono">{node.id}</span>
                      </div>
                    </div>
                  </td>

                  <td className="py-4 px-3">
                    <span className="font-semibold text-zinc-700 dark:text-zinc-200 block">{node.city}</span>
                    <span className="font-mono text-zinc-400 text-[11px]">{node.ip}</span>
                  </td>

                  <td className="py-4 px-3">
                    {isOffline ? (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold bg-coral-100 text-coral-700 dark:bg-coral-950/60 dark:text-coral-300">
                        <WifiOff size={11} /> Dropped (2/3 Quorum)
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold bg-mint-100 text-mint-700 dark:bg-mint-950/60 dark:text-mint-300">
                        <CheckCircle2 size={11} /> {node.status}
                      </span>
                    )}
                  </td>

                  <td className="py-4 px-3 font-mono font-bold text-zinc-800 dark:text-zinc-200">
                    {node.samples} MRIs
                  </td>

                  <td className="py-4 px-3 font-mono text-zinc-600 dark:text-zinc-300">
                    {node.latencyMs} ms
                  </td>

                  <td className="py-4 px-3 font-mono text-zinc-600 dark:text-zinc-300">
                    ||g|| = {node.gradientNorm}
                  </td>

                  <td className="py-4 px-3">
                    <div className="flex items-center gap-2">
                      <div className="w-16 bg-zinc-100 dark:bg-zinc-800 rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-lavender-600 h-2 rounded-full"
                          style={{ width: `${node.contributionPercent}%` }}
                        />
                      </div>
                      <span className="font-mono font-bold text-zinc-700 dark:text-zinc-300">
                        {node.contributionPercent}%
                      </span>
                    </div>
                  </td>

                  <td className="py-4 px-3 text-right">
                    <span className="inline-flex items-center gap-1 text-[11px] font-bold text-mint-600">
                      <ShieldCheck size={13} /> Verified (CA)
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
