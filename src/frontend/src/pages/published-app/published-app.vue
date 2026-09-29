<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox, type FormInstance, type FormRules } from 'element-plus'
import {
  CopyDocument,
  Edit,
  Key,
  Plus,
  Refresh,
  Search,
  Upload,
  View
} from '@element-plus/icons-vue'

import { getAgentsAPI } from '../../apis/agent'
import {
  createAppApiKeyAPI,
  getPublishedAppAPI,
  listAppAuditLogsAPI,
  listPublishedAppsAPI,
  pausePublishedAppAPI,
  publishAppAPI,
  republishAppAPI,
  rollbackAppVersionAPI,
  resumePublishedAppAPI,
  revokeAppApiKeyAPI,
  rotateAppApiKeyAPI,
  updatePublishedAppAPI,
  type AppApiKey,
  type AuditLog,
  type PublishedApp,
  type PublishedAppDetail
} from '../../apis/published-app'

interface AgentOption {
  id?: string
  agent_id?: string
  name: string
}

const apps = ref<PublishedApp[]>([])
const agents = ref<AgentOption[]>([])
const loading = ref(false)
const submitting = ref(false)
const detailLoading = ref(false)
const searchKeyword = ref('')
const statusFilter = ref('')

const formVisible = ref(false)
const editingAppId = ref('')
const formRef = ref<FormInstance>()
const appForm = reactive({
  agent_id: '',
  name: '',
  slug: '',
  description: '',
  api_enabled: true,
  web_enabled: true
})

const detailVisible = ref(false)
const detail = ref<PublishedAppDetail | null>(null)
const audits = ref<AuditLog[]>([])
const activeDetailTab = ref('overview')

const keyDialogVisible = ref(false)
const keyName = ref('default')
const keySubmitting = ref(false)
const secretDialogVisible = ref(false)
const secretValue = ref('')

const rules: FormRules = {
  agent_id: [{ required: true, message: '请选择要发布的智能体', trigger: 'change' }],
  name: [{ required: true, message: '请输入应用名称', trigger: 'blur' }]
}

const filteredApps = computed(() => {
  const keyword = searchKeyword.value.trim().toLowerCase()
  return apps.value.filter((app) => {
    const matchesKeyword = !keyword || [app.name, app.slug, app.description]
      .some((value) => (value || '').toLowerCase().includes(keyword))
    const matchesStatus = !statusFilter.value || app.status === statusFilter.value
    return matchesKeyword && matchesStatus
  })
})

const errorText = (error: any) => error?.response?.data?.detail || error?.message || '操作失败'

const formatTime = (value?: string | null) => {
  if (!value) return '—'
  return new Date(value).toLocaleString('zh-CN', { hour12: false })
}

const agentId = (agent: AgentOption) => agent.id || agent.agent_id || ''

const loadApps = async () => {
  loading.value = true
  try {
    const response = await listPublishedAppsAPI()
    apps.value = response.data.data || []
  } catch (error) {
    ElMessage.error(errorText(error))
  } finally {
    loading.value = false
  }
}

const loadAgents = async () => {
  try {
    const response = await getAgentsAPI()
    agents.value = (response.data.data || []) as unknown as AgentOption[]
  } catch (error) {
    ElMessage.error(`智能体列表加载失败：${errorText(error)}`)
  }
}

const resetForm = () => {
  Object.assign(appForm, {
    agent_id: '',
    name: '',
    slug: '',
    description: '',
    api_enabled: true,
    web_enabled: true
  })
  editingAppId.value = ''
  formRef.value?.clearValidate()
}

const openCreate = () => {
  resetForm()
  formVisible.value = true
}

const openEdit = (app: PublishedApp) => {
  editingAppId.value = app.app_id
  Object.assign(appForm, {
    agent_id: app.agent_id,
    name: app.name,
    slug: app.slug,
    description: app.description,
    api_enabled: app.api_enabled,
    web_enabled: app.web_enabled
  })
  formVisible.value = true
}

const submitForm = async () => {
  const valid = await formRef.value?.validate().catch(() => false)
  if (!valid) return
  submitting.value = true
  try {
    if (editingAppId.value) {
      await updatePublishedAppAPI(editingAppId.value, {
        name: appForm.name,
        slug: appForm.slug,
        description: appForm.description,
        api_enabled: appForm.api_enabled,
        web_enabled: appForm.web_enabled
      })
      ElMessage.success('应用信息已更新')
    } else {
      await publishAppAPI({
        agent_id: appForm.agent_id,
        name: appForm.name,
        slug: appForm.slug || undefined,
        description: appForm.description,
        api_enabled: appForm.api_enabled,
        web_enabled: appForm.web_enabled
      })
      ElMessage.success('应用已发布，版本 1 已冻结')
    }
    formVisible.value = false
    await loadApps()
  } catch (error) {
    ElMessage.error(errorText(error))
  } finally {
    submitting.value = false
  }
}

const changeStatus = async (app: PublishedApp, active: boolean) => {
  try {
    if (active) {
      await resumePublishedAppAPI(app.app_id)
      app.status = 'active'
      ElMessage.success('应用已恢复')
    } else {
      await pausePublishedAppAPI(app.app_id)
      app.status = 'paused'
      ElMessage.success('应用已暂停')
    }
  } catch (error) {
    ElMessage.error(errorText(error))
  }
}

const republish = async (app: PublishedApp) => {
  try {
    await ElMessageBox.confirm(
      '将当前智能体配置冻结为新版本，已有版本不会被覆盖。',
      '重新发布',
      { confirmButtonText: '发布新版本', cancelButtonText: '取消', type: 'warning' }
    )
    const response = await republishAppAPI(app.app_id)
    ElMessage.success(`版本 ${response.data.data.version} 已发布`)
    await loadApps()
    if (detail.value?.app_id === app.app_id) await loadDetail(app.app_id)
  } catch (error: any) {
    if (error !== 'cancel') ElMessage.error(errorText(error))
  }
}

const rollbackVersion = async (versionId: string, versionNumber: number) => {
  if (!detail.value) return
  try {
    await ElMessageBox.confirm(
      `将版本 ${versionNumber} 的配置复制为新的最新版本，历史版本不会被修改。`,
      '回滚应用版本',
      { confirmButtonText: '确认回滚', cancelButtonText: '取消', type: 'warning' }
    )
    const response = await rollbackAppVersionAPI(detail.value.app_id, versionId)
    ElMessage.success(`已回滚并发布为版本 ${response.data.data.version}`)
    await Promise.all([loadApps(), loadDetail(detail.value.app_id)])
  } catch (error: any) {
    if (error !== 'cancel') ElMessage.error(errorText(error))
  }
}

const loadDetail = async (appId: string) => {
  detailLoading.value = true
  try {
    const [detailResponse, auditResponse] = await Promise.all([
      getPublishedAppAPI(appId),
      listAppAuditLogsAPI(appId)
    ])
    detail.value = detailResponse.data.data
    audits.value = auditResponse.data.data || []
  } catch (error) {
    ElMessage.error(errorText(error))
  } finally {
    detailLoading.value = false
  }
}

const openDetail = async (app: PublishedApp) => {
  activeDetailTab.value = 'overview'
  detailVisible.value = true
  await loadDetail(app.app_id)
}

const openCreateKey = () => {
  keyName.value = 'default'
  keyDialogVisible.value = true
}

const showSecret = (value: string) => {
  secretValue.value = value
  secretDialogVisible.value = true
}

const createKey = async () => {
  if (!detail.value || !keyName.value.trim()) return
  keySubmitting.value = true
  try {
    const response = await createAppApiKeyAPI(detail.value.app_id, keyName.value.trim())
    keyDialogVisible.value = false
    showSecret(response.data.data.api_key || '')
    await loadDetail(detail.value.app_id)
  } catch (error) {
    ElMessage.error(errorText(error))
  } finally {
    keySubmitting.value = false
  }
}

const revokeKey = async (key: AppApiKey) => {
  if (!detail.value) return
  try {
    await ElMessageBox.confirm(`吊销后，前缀为 ${key.key_prefix} 的调用将立即失效。`, '吊销 API Key', {
      confirmButtonText: '确认吊销', cancelButtonText: '取消', type: 'warning'
    })
    await revokeAppApiKeyAPI(detail.value.app_id, key.key_id)
    ElMessage.success('API Key 已吊销')
    await loadDetail(detail.value.app_id)
  } catch (error: any) {
    if (error !== 'cancel') ElMessage.error(errorText(error))
  }
}

const rotateKey = async (key: AppApiKey) => {
  if (!detail.value) return
  try {
    await ElMessageBox.confirm('轮换会立即吊销旧 Key，并生成一个只展示一次的新 Key。', '轮换 API Key', {
      confirmButtonText: '确认轮换', cancelButtonText: '取消', type: 'warning'
    })
    const response = await rotateAppApiKeyAPI(detail.value.app_id, key.key_id)
    showSecret(response.data.data.api_key || '')
    await loadDetail(detail.value.app_id)
  } catch (error: any) {
    if (error !== 'cancel') ElMessage.error(errorText(error))
  }
}

const copySecret = async () => {
  await navigator.clipboard.writeText(secretValue.value)
  ElMessage.success('已复制到剪贴板')
}

onMounted(() => {
  void Promise.all([loadApps(), loadAgents()])
})
</script>

<template>
  <div class="app-management">
    <header class="page-header">
      <div>
        <h2>应用管理</h2>
        <p>发布智能体、管理调用凭证，并追踪每次配置变更与调用。</p>
      </div>
      <el-button type="primary" :icon="Plus" @click="openCreate">发布应用</el-button>
    </header>

    <div class="toolbar">
      <el-input v-model="searchKeyword" clearable placeholder="搜索名称、Slug 或描述" :prefix-icon="Search" />
      <el-select v-model="statusFilter" clearable placeholder="全部状态">
        <el-option label="运行中" value="active" />
        <el-option label="已暂停" value="paused" />
      </el-select>
      <el-button :icon="Refresh" :loading="loading" @click="loadApps">刷新</el-button>
    </div>

    <el-table :data="filteredApps" v-loading="loading" row-key="app_id" class="apps-table">
      <el-table-column label="应用" min-width="220">
        <template #default="{ row }">
          <div class="app-name">{{ row.name }}</div>
          <div class="secondary">{{ row.description || '暂无描述' }}</div>
        </template>
      </el-table-column>
      <el-table-column label="Slug" min-width="150">
        <template #default="{ row }"><code>{{ row.slug }}</code></template>
      </el-table-column>
      <el-table-column label="版本" width="90" align="center">
        <template #default="{ row }">v{{ row.current_version || 1 }}</template>
      </el-table-column>
      <el-table-column label="访问方式" width="150">
        <template #default="{ row }">
          <el-tag v-if="row.web_enabled" size="small" effect="plain">Web</el-tag>
          <el-tag v-if="row.api_enabled" size="small" effect="plain" class="tag-gap">API</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="状态" width="120">
        <template #default="{ row }">
          <el-switch
            :model-value="row.status === 'active'"
            inline-prompt
            active-text="运行"
            inactive-text="暂停"
            @change="(value: boolean | string | number) => changeStatus(row, Boolean(value))"
          />
        </template>
      </el-table-column>
      <el-table-column label="更新时间" width="180">
        <template #default="{ row }">{{ formatTime(row.update_time) }}</template>
      </el-table-column>
      <el-table-column label="操作" width="280" fixed="right">
        <template #default="{ row }">
          <el-button link type="primary" :icon="View" @click="openDetail(row)">管理</el-button>
          <el-button link :icon="Edit" @click="openEdit(row)">编辑</el-button>
          <el-button link :icon="Upload" @click="republish(row)">重新发布</el-button>
        </template>
      </el-table-column>
      <template #empty>
        <el-empty description="还没有发布应用" />
      </template>
    </el-table>

    <el-dialog v-model="formVisible" :title="editingAppId ? '编辑应用' : '发布应用'" width="560px" @closed="resetForm">
      <el-form ref="formRef" :model="appForm" :rules="rules" label-position="top">
        <el-form-item label="智能体" prop="agent_id">
          <el-select v-model="appForm.agent_id" filterable :disabled="Boolean(editingAppId)" placeholder="选择智能体">
            <el-option v-for="agent in agents" :key="agentId(agent)" :label="agent.name" :value="agentId(agent)" />
          </el-select>
        </el-form-item>
        <el-form-item label="应用名称" prop="name">
          <el-input v-model="appForm.name" maxlength="128" show-word-limit />
        </el-form-item>
        <el-form-item label="Slug" prop="slug">
          <el-input v-model="appForm.slug" maxlength="128" placeholder="留空时根据名称生成" />
        </el-form-item>
        <el-form-item label="描述" prop="description">
          <el-input v-model="appForm.description" type="textarea" :rows="3" maxlength="2000" show-word-limit />
        </el-form-item>
        <div class="switch-row">
          <label><el-switch v-model="appForm.web_enabled" /> Web 访问</label>
          <label><el-switch v-model="appForm.api_enabled" /> API 调用</label>
        </div>
      </el-form>
      <template #footer>
        <el-button @click="formVisible = false">取消</el-button>
        <el-button type="primary" :loading="submitting" @click="submitForm">
          {{ editingAppId ? '保存' : '发布' }}
        </el-button>
      </template>
    </el-dialog>

    <el-drawer v-model="detailVisible" :title="detail?.name || '应用详情'" size="70%">
      <div v-loading="detailLoading" class="detail-content">
        <el-tabs v-if="detail" v-model="activeDetailTab">
          <el-tab-pane label="概览" name="overview">
            <el-descriptions :column="2" border>
              <el-descriptions-item label="状态">{{ detail.status === 'active' ? '运行中' : '已暂停' }}</el-descriptions-item>
              <el-descriptions-item label="当前版本">v{{ detail.current_version || 1 }}</el-descriptions-item>
              <el-descriptions-item label="Slug"><code>{{ detail.slug }}</code></el-descriptions-item>
              <el-descriptions-item label="Agent ID"><code>{{ detail.agent_id }}</code></el-descriptions-item>
              <el-descriptions-item label="API 地址" :span="2">
                <code>/api/v1/public/apps/{{ detail.slug }}/completion</code>
              </el-descriptions-item>
              <el-descriptions-item label="描述" :span="2">{{ detail.description || '暂无描述' }}</el-descriptions-item>
            </el-descriptions>
          </el-tab-pane>

          <el-tab-pane label="API Key" name="keys">
            <div class="tab-actions">
              <el-button type="primary" :icon="Key" @click="openCreateKey">创建 Key</el-button>
            </div>
            <el-table :data="detail.api_keys" row-key="key_id">
              <el-table-column prop="name" label="名称" min-width="140" />
              <el-table-column prop="key_prefix" label="前缀" min-width="140" />
              <el-table-column label="状态" width="90">
                <template #default="{ row }">
                  <el-tag :type="row.enabled ? 'success' : 'info'" size="small">{{ row.enabled ? '有效' : '已吊销' }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column label="最后使用" width="180">
                <template #default="{ row }">{{ formatTime(row.last_used_at) }}</template>
              </el-table-column>
              <el-table-column label="创建时间" width="180">
                <template #default="{ row }">{{ formatTime(row.create_time) }}</template>
              </el-table-column>
              <el-table-column label="操作" width="150" fixed="right">
                <template #default="{ row }">
                  <template v-if="row.enabled">
                    <el-button link type="primary" @click="rotateKey(row)">轮换</el-button>
                    <el-button link type="danger" @click="revokeKey(row)">吊销</el-button>
                  </template>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>

          <el-tab-pane label="发布版本" name="versions">
            <el-table :data="detail.versions" row-key="version_id">
              <el-table-column label="版本" width="90"><template #default="{ row }">v{{ row.version }}</template></el-table-column>
              <el-table-column prop="config_hash" label="配置摘要" min-width="240">
                <template #default="{ row }"><code>{{ row.config_hash.slice(0, 16) }}</code></template>
              </el-table-column>
              <el-table-column prop="created_by" label="发布人" min-width="160" />
              <el-table-column label="发布时间" width="180"><template #default="{ row }">{{ formatTime(row.create_time) }}</template></el-table-column>
              <el-table-column label="操作" width="90" fixed="right">
                <template #default="{ row, $index }">
                  <el-button v-if="$index > 0" link type="primary" @click="rollbackVersion(row.version_id, row.version)">回滚</el-button>
                </template>
              </el-table-column>
            </el-table>
          </el-tab-pane>

          <el-tab-pane label="审计日志" name="audits">
            <el-table :data="audits" row-key="audit_id">
              <el-table-column prop="action" label="动作" min-width="150" />
              <el-table-column prop="actor_id" label="调用方" min-width="180" show-overflow-tooltip />
              <el-table-column prop="trace_id" label="Trace / Run ID" min-width="190" show-overflow-tooltip />
              <el-table-column label="状态" width="90">
                <template #default="{ row }"><el-tag :type="row.status === 'success' ? 'success' : 'danger'" size="small">{{ row.status }}</el-tag></template>
              </el-table-column>
              <el-table-column prop="ip" label="来源 IP" min-width="130" />
              <el-table-column label="时间" width="180"><template #default="{ row }">{{ formatTime(row.create_time) }}</template></el-table-column>
            </el-table>
          </el-tab-pane>
        </el-tabs>
      </div>
    </el-drawer>

    <el-dialog v-model="keyDialogVisible" title="创建 API Key" width="420px">
      <el-form label-position="top">
        <el-form-item label="Key 名称">
          <el-input v-model="keyName" maxlength="128" placeholder="例如：生产环境 BFF" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="keyDialogVisible = false">取消</el-button>
        <el-button type="primary" :disabled="!keyName.trim()" :loading="keySubmitting" @click="createKey">创建</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="secretDialogVisible" title="请立即保存 API Key" width="560px" :close-on-click-modal="false">
      <el-alert title="该密钥只展示一次，关闭后无法再次查看。" type="warning" show-icon :closable="false" />
      <div class="secret-row">
        <el-input :model-value="secretValue" readonly />
        <el-button :icon="CopyDocument" @click="copySecret">复制</el-button>
      </div>
      <template #footer><el-button type="primary" @click="secretDialogVisible = false">我已保存</el-button></template>
    </el-dialog>
  </div>
</template>

<style scoped lang="scss">
.app-management {
  min-height: calc(100vh - 64px);
  padding: 24px;
  background: #f5f7fa;
  color: #1f2937;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
}

.page-header h2 { margin: 0 0 6px; font-size: 24px; }
.page-header p { margin: 0; color: #6b7280; }

.toolbar {
  display: grid;
  grid-template-columns: minmax(220px, 360px) 150px auto;
  gap: 12px;
  align-items: center;
  margin-bottom: 14px;
}

.apps-table { width: 100%; }
.app-name { font-weight: 600; margin-bottom: 4px; }
.secondary { color: #909399; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tag-gap { margin-left: 6px; }
.switch-row { display: flex; gap: 28px; }
.switch-row label { display: inline-flex; align-items: center; gap: 8px; color: #4b5563; }
.detail-content { min-height: 240px; }
.tab-actions { display: flex; justify-content: flex-end; margin-bottom: 12px; }
.secret-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 10px; margin-top: 16px; }
code { color: #374151; background: #eef1f5; padding: 2px 5px; border-radius: 4px; overflow-wrap: anywhere; }

@media (max-width: 760px) {
  .app-management { padding: 16px; }
  .page-header { align-items: flex-start; }
  .toolbar { grid-template-columns: 1fr; }
  .switch-row { flex-direction: column; gap: 12px; }
  .secret-row { grid-template-columns: 1fr; }
}
</style>
