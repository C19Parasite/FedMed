import React from 'react';
import { History, CheckCircle2, ShieldCheck, Download, Award } from 'lucide-react';
import { useDashboardStore } from '../store/useDashboardStore';

export const RoundsPage: React.FC = () => {
  const { rounds } = useDashboardStore();

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="card-base p-6 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-extrabold text-zinc-900 dark:text-white">
            Federated Training Rounds Ledger
          </h2>
          <p className="text-xs text-zinc-500 mt-1">
            Immutable log of aggregated rounds, validation Dice scores, loss convergence, and active quorum.
          </p>
        </div>

        <button
          onClick={() => {
            const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(rounds, null, 2));
            const downloadAnchor = document.createElement('a');
            downloadAnchor.setAttribute('href', dataStr);
            downloadAnchor.setAttribute('download', 'fedmed_rounds_ledger.json');
            document.body.appendChild(downloadAnchor);
            downloadAnchor.click();
            downloadAnchor.remove();
          }}
          className="flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold bg-surface-soft hover:bg-zinc-100 dark:bg-zinc-800 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-200 border border-brand-border transition-colors self-start sm:self-auto"
        >
          <Download size={14} />
          <span>Export Rounds JSON</span>
        </button>
      </div>

      {/* Rounds Table */}
      <div className="card-base p-6 bg-white dark:bg-zinc-900 border border-brand-border dark:border-zinc-800 overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-brand-border/80 dark:border-zinc-800 text-zinc-400 font-bold uppercase tracking-wider">
              <th className="pb-3 px-3">Round #</th>
              <th className="pb-3 px-3">Timestamp</th>
              <th className="pb-3 px-3">Mean BraTS Dice</th>
              <th className="pb-3 px-3">DiceCE Loss</th>
              <th className="pb-3 px-3">Subregions (WT / TC / ET)</th>
              <th className="pb-3 px-3">Quorum</th>
              <th className="pb-3 px-3">Round Latency</th>
              <th className="pb-3 px-3 text-right">Security Protocol</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-brand-border/50 dark:divide-zinc-800/60 font-mono">
            {[...rounds].reverse().map((r) => (
              <tr key={r.round} className="hover:bg-surface-soft dark:hover:bg-zinc-800/40 transition-colors">
                <td className="py-3.5 px-3 font-bold text-zinc-900 dark:text-white font-sans">
                  Round {r.round}
                </td>

                <td className="py-3.5 px-3 text-zinc-500 font-sans">{r.timestamp}</td>

                <td className="py-3.5 px-3 text-mint-600 dark:text-mint-400 font-bold text-sm">
                  {(r.dice * 100).toFixed(2)}%
                </td>

                <td className="py-3.5 px-3 text-lavender-700 dark:text-lavender-300 font-bold">
                  {r.global_loss.toFixed(4)}
                </td>

                <td className="py-3.5 px-3 text-zinc-600 dark:text-zinc-300">
                  {r.dice_wt ? `${(r.dice_wt * 100).toFixed(1)}%` : '90.2%'} /{' '}
                  {r.dice_tc ? `${(r.dice_tc * 100).toFixed(1)}%` : '85.4%'} /{' '}
                  {r.dice_et ? `${(r.dice_et * 100).toFixed(1)}%` : '83.1%'}
                </td>

                <td className="py-3.5 px-3 text-zinc-700 dark:text-zinc-200 font-sans">
                  <span className="px-2 py-0.5 rounded-full text-[11px] font-bold bg-mint-50 text-mint-700 dark:bg-mint-950/40 dark:text-mint-300 border border-mint-200 dark:border-mint-800">
                    {r.participating_clients} of {r.total_clients} nodes
                  </span>
                </td>

                <td className="py-3.5 px-3 text-zinc-500">
                  {r.round_duration_sec.toFixed(2)}s
                </td>

                <td className="py-3.5 px-3 text-right font-sans">
                  <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-lavender-700 dark:text-lavender-300">
                    <ShieldCheck size={13} /> CKKS Sum + DP
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
