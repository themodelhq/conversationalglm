import type { ChatResponse, ComputeInfo, DatasetAsset, Message, TrainingConfig, TrainingLog, TrainingRun, TrainingTask } from './types';

function normalizedBase(value: string): string {
  return value.trim().replace(/\/+$/, '');
}

const configuredApi = normalizedBase(import.meta.env.VITE_API_URL ?? '');
const useDirectApi = import.meta.env.VITE_DIRECT_API === 'true';
const BASE = import.meta.env.PROD && !useDirectApi ? '/api' : configuredApi;
let token = localStorage.getItem('glm_token') ?? '';

export function setToken(value: string) { token = value; localStorage.setItem('glm_token', value); }
export function clearToken() { token = ''; localStorage.removeItem('glm_token'); }

function endpoint(path: string): string {
  if (!path.startsWith('/')) throw new Error('API paths must start with /.');
  return `${BASE}${path}`;
}

async function networkError(error: unknown): Promise<never> {
  const detail = error instanceof Error ? error.message : String(error);
  if (BASE === '/api') {
    throw new Error(`Unable to contact the training API through the Netlify proxy. Verify Netlify variable GLM_API_ORIGIN points to the full Render URL. ${detail}`);
  }
  throw new Error(`Unable to contact the API at ${BASE || window.location.origin}. Verify VITE_API_URL and the Render service status. ${detail}`);
}

async function request<T>(path: string, body?: unknown, method = 'POST'): Promise<T> {
  let response: Response;
  try {
    response = await fetch(endpoint(path), {
      method,
      headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (error) {
    return networkError(error);
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(payload.detail ?? response.statusText);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const auth = {
  register: (email: string, password: string) => request<{ access_token: string }>('/v1/auth/register', { email, password }),
  login: (email: string, password: string) => request<{ access_token: string }>('/v1/auth/login', { email, password }),
};
export const chat = (messages: Message[], conversationId?: string) => request<ChatResponse>('/v1/chat/completions', { messages, conversation_id: conversationId, max_new_tokens: 512, temperature: .7, top_p: .9, use_memory: true, use_rag: true });
export async function transcribe(file: File): Promise<{ text: string }> {
  const form = new FormData(); form.append('file', file);
  let response: Response;
  try {
    response = await fetch(endpoint('/v1/audio/transcriptions'), { method: 'POST', headers: token ? { Authorization: `Bearer ${token}` } : {}, body: form });
  } catch (error) { return networkError(error); }
  if (!response.ok) throw new Error((await response.json().catch(() => ({ detail: 'Transcription failed' }))).detail);
  return response.json();
}
export const platform = {
  compute: () => request<ComputeInfo>('/v1/platform/compute', undefined, 'GET'),
  datasets: () => request<DatasetAsset[]>('/v1/platform/datasets', undefined, 'GET'),
  runs: () => request<TrainingRun[]>('/v1/platform/runs', undefined, 'GET'),
  createRun: (config: TrainingConfig) => request<TrainingRun>('/v1/platform/runs', config),
  stopRun: (id: string) => request<TrainingRun>(`/v1/platform/runs/${id}/stop`, {}),
  logs: (id: string) => request<TrainingLog>(`/v1/platform/runs/${id}/logs?limit=350`, undefined, 'GET'),
  async uploadDataset(file: File, task: TrainingTask): Promise<DatasetAsset> {
    const form = new FormData(); form.append('file', file);
    let response: Response;
    try {
      response = await fetch(endpoint(`/v1/platform/datasets?task=${task}`), { method: 'POST', headers: token ? { Authorization: `Bearer ${token}` } : {}, body: form });
    } catch (error) { return networkError(error); }
    if (!response.ok) throw new Error((await response.json().catch(() => ({ detail: 'Dataset upload failed' }))).detail);
    return response.json() as Promise<DatasetAsset>;
  },
};
