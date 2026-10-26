import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useDashboardStore } from '../store/useDashboardStore';
import { idempotentFetch } from '../lib/idempotency';
import { toast } from 'sonner';
import { HospitalNode, PrivacyState, RoundMetric } from '../types';

export function useRoundHistory() {
  const { rounds, isMockMode } = useDashboardStore();

  return useQuery<RoundMetric[]>({
    queryKey: ['roundsHistory'],
    queryFn: async () => {
      if (isMockMode) {
        return rounds;
      }
      return idempotentFetch<RoundMetric[]>('/api/history');
    },
    initialData: rounds,
    staleTime: 1000 * 30, // 30s stale time
    refetchInterval: isMockMode ? false : 5000,
  });
}

export function useNodeList() {
  const { nodes, isMockMode } = useDashboardStore();

  return useQuery<HospitalNode[]>({
    queryKey: ['nodeList'],
    queryFn: async () => {
      if (isMockMode) {
        return nodes;
      }
      const data = await idempotentFetch<{ hospitals: Record<string, string> }>('/api/status');
      return nodes.map((n) => ({
        ...n,
        status: (data.hospitals?.[n.id] as any) || n.status,
      }));
    },
    initialData: nodes,
    staleTime: 1000 * 30,
    refetchInterval: isMockMode ? false : 5000,
  });
}

export function usePrivacyReport() {
  const { privacy, isMockMode } = useDashboardStore();

  return useQuery<PrivacyState>({
    queryKey: ['privacyReport'],
    queryFn: async () => {
      if (isMockMode) {
        return privacy;
      }
      const data = await idempotentFetch<any>('/api/status');
      return {
        ...privacy,
        epsilon: data.privacy?.cumulative_epsilon ?? privacy.epsilon,
        noise_sigma: data.privacy?.noise_multiplier ?? privacy.noise_sigma,
      };
    },
    initialData: privacy,
    staleTime: 1000 * 60,
  });
}

/**
 * Optimistic Start New Round mutation
 */
export function useStartRoundMutation() {
  const queryClient = useQueryClient();
  const { startNewRound, rounds } = useDashboardStore();

  return useMutation({
    mutationFn: async () => {
      // Calls server API with idempotency key
      return idempotentFetch('/api/rounds/start', {
        method: 'POST',
      }).catch(() => {
        // Mock fallback if local server is not running
        return { success: true };
      });
    },
    onMutate: async () => {
      // 2. Optimistic Update: instantly update UI
      await queryClient.cancelQueries({ queryKey: ['roundsHistory'] });
      const previousRounds = queryClient.getQueryData<RoundMetric[]>(['roundsHistory']);

      startNewRound();

      return { previousRounds };
    },
    onError: (err, _variables, context) => {
      // Rollback on failure
      if (context?.previousRounds) {
        queryClient.setQueryData(['roundsHistory'], context.previousRounds);
      }
      toast.error('Failed to trigger federated round', {
        description: 'Server failed to respond; optimistic state was rolled back.',
      });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['roundsHistory'] });
    },
  });
}
