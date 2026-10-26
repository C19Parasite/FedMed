/**
 * Typed API client for FedMed dashboard.
 * Communicates with the FastAPI backend via REST and WebSocket.
 * Guarantees zero sensitive secrets or patient identifiers are transmitted.
 */

const API_BASE = 'http://127.0.0.1:8000';
const WS_BASE = 'ws://127.0.0.1:8000/ws/metrics';

/**
 * Fetch high-level system status and privacy settings.
 */
export async function fetchStatus() {
  const resp = await fetch(`${API_BASE}/api/status`);
  if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
  return resp.json();
}

/**
 * Fetch full history of federated rounds.
 */
export async function fetchHistory() {
  const resp = await fetch(`${API_BASE}/api/history`);
  if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
  return resp.json();
}

/**
 * Fetch safe configuration metadata.
 */
export async function fetchConfig() {
  const resp = await fetch(`${API_BASE}/api/config`);
  if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
  return resp.json();
}

/**
 * Fetch ablation study results.
 */
export async function fetchAblation() {
  const resp = await fetch(`${API_BASE}/api/ablation`);
  if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
  return resp.json();
}

/**
 * Fetch public BraTS demo cases for visualization.
 */
export async function fetchCases() {
  const resp = await fetch(`${API_BASE}/api/cases`);
  if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
  return resp.json();
}

/**
 * Get image URL for a demo case slice view.
 * @param {string} caseId
 * @param {number} sliceZ
 * @param {'mri'|'gt'|'pred'|'central'|'overlay'} view
 */
export function getSliceUrl(caseId, sliceZ, view = 'overlay') {
  return `${API_BASE}/api/cases/${encodeURIComponent(caseId)}/slice/${sliceZ}?view=${view}`;
}

/**
 * Establish resilient auto-reconnecting WebSocket connection.
 * @param {(data: any) => void} onMessage
 * @param {(connected: boolean, reconnecting: boolean) => void} onStatusChange
 */
export function connectMetricsWebSocket(onMessage, onStatusChange) {
  let ws = null;
  let retryTimer = null;
  let isClosedManually = false;
  let reconnectAttempts = 0;

  function connect() {
    try {
      ws = new WebSocket(WS_BASE);

      ws.onopen = () => {
        reconnectAttempts = 0;
        onStatusChange(true, false);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onerror = () => {
        // triggers close handler
      };

      ws.onclose = () => {
        if (!isClosedManually) {
          onStatusChange(false, true);
          const delay = Math.min(1000 * Math.pow(1.5, reconnectAttempts), 10000);
          reconnectAttempts++;
          retryTimer = setTimeout(connect, delay);
        } else {
          onStatusChange(false, false);
        }
      };
    } catch (e) {
      onStatusChange(false, true);
      retryTimer = setTimeout(connect, 3000);
    }
  }

  connect();

  return () => {
    isClosedManually = true;
    if (retryTimer) clearTimeout(retryTimer);
    if (ws) ws.close();
  };
}
