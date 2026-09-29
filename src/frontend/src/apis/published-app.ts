import { request } from '../utils/request'

export interface AppApiResponse<T> {
  code: number
  data: T
}

export interface PublishedApp {
  app_id: string
  agent_id: string
  owner_id: string
  name: string
  slug: string
  description: string
  status: 'active' | 'paused'
  api_enabled: boolean
  web_enabled: boolean
  current_version?: number | null
  current_version_id?: string | null
  create_time?: string
  update_time?: string
}

export interface AppApiKey {
  key_id: string
  app_id: string
  owner_id: string
  name: string
  key_prefix: string
  enabled: boolean
  last_used_at?: string | null
  create_time?: string
  api_key?: string
}

export interface AppVersion {
  version_id: string
  app_id: string
  agent_id: string
  version: number
  config_hash: string
  created_by: string
  create_time?: string
}

export interface AuditLog {
  audit_id: string
  actor_id: string
  action: string
  trace_id: string
  status: string
  ip: string
  user_agent: string
  detail: Record<string, unknown>
  create_time?: string
}

export interface PublishedAppDetail extends PublishedApp {
  api_keys: AppApiKey[]
  versions: AppVersion[]
  runtime_snapshot?: Record<string, unknown>
}

export interface PublishAppPayload {
  agent_id: string
  name: string
  slug?: string
  description: string
  api_enabled: boolean
  web_enabled: boolean
}

export interface UpdateAppPayload {
  name?: string
  slug?: string
  description?: string
  api_enabled?: boolean
  web_enabled?: boolean
}

export const listPublishedAppsAPI = () =>
  request<AppApiResponse<PublishedApp[]>>({ url: '/api/v1/apps', method: 'GET' })

export const getPublishedAppAPI = (appId: string) =>
  request<AppApiResponse<PublishedAppDetail>>({ url: `/api/v1/apps/${appId}`, method: 'GET' })

export const publishAppAPI = (data: PublishAppPayload) =>
  request<AppApiResponse<PublishedApp>>({ url: '/api/v1/apps/publish', method: 'POST', data })

export const updatePublishedAppAPI = (appId: string, data: UpdateAppPayload) =>
  request<AppApiResponse<PublishedApp>>({ url: `/api/v1/apps/${appId}`, method: 'PUT', data })

export const pausePublishedAppAPI = (appId: string) =>
  request<AppApiResponse<void>>({ url: `/api/v1/apps/${appId}/pause`, method: 'POST' })

export const resumePublishedAppAPI = (appId: string) =>
  request<AppApiResponse<PublishedApp>>({ url: `/api/v1/apps/${appId}/resume`, method: 'POST' })

export const republishAppAPI = (appId: string) =>
  request<AppApiResponse<AppVersion>>({ url: `/api/v1/apps/${appId}/republish`, method: 'POST' })

export const rollbackAppVersionAPI = (appId: string, versionId: string) =>
  request<AppApiResponse<AppVersion>>({
    url: `/api/v1/apps/${appId}/versions/${versionId}/rollback`,
    method: 'POST'
  })

export const createAppApiKeyAPI = (appId: string, name: string) =>
  request<AppApiResponse<AppApiKey>>({
    url: '/api/v1/apps/keys',
    method: 'POST',
    data: { app_id: appId, name }
  })

export const revokeAppApiKeyAPI = (appId: string, keyId: string) =>
  request<AppApiResponse<void>>({ url: `/api/v1/apps/${appId}/keys/${keyId}`, method: 'DELETE' })

export const rotateAppApiKeyAPI = (appId: string, keyId: string) =>
  request<AppApiResponse<AppApiKey>>({ url: `/api/v1/apps/${appId}/keys/${keyId}/rotate`, method: 'POST' })

export const listAppAuditLogsAPI = (appId: string) =>
  request<AppApiResponse<AuditLog[]>>({ url: `/api/v1/apps/${appId}/audit-logs`, method: 'GET' })
