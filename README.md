# AgentFlow

AgentFlow 是一个面向多智能体协作、知识库检索和工具调用的全栈平台。项目提供可视化管理界面、FastAPI 服务端、模型与工具配置、RAG 检索、MCP 接入，以及面向高并发场景的限流、超时、熔断和可观测能力。

## 核心能力

- **多智能体编排**：配置智能体、子智能体与任务路由，支持复杂任务的分解和协作执行。
- **多模型接入**：分别配置对话、工具调用、推理、Embedding、Rerank、视觉和文生图模型。
- **知识库与 RAG**：支持文档分块、向量检索、重排序，以及可选的 Elasticsearch 关键词检索。
- **工具与 MCP**：管理内置工具、外部搜索服务和 MCP 服务，统一处理工具发现与调用。
- **会话与记忆**：保存对话历史、上下文和长期记忆，支持基于 Redis 与向量数据库的状态管理。
- **可靠性治理**：提供请求准入、用户级限流、工具并发控制、超时、幂等和流式输出预算。
- **可观测性**：内置 Prometheus 指标，并可通过 OTLP 接入外部追踪系统。
- **Web 管理端**：基于 Vue 3、TypeScript、Element Plus 和 Vite，覆盖对话、智能体、知识库与系统配置界面。

## 技术架构

```text
Web 管理端（Vue 3）
        │ HTTP / SSE
        ▼
API 服务（FastAPI）
        ├── 智能体编排与意图路由
        ├── 模型、工具与 MCP 调用
        ├── RAG 检索与知识库管理
        ├── 会话、记忆与权限管理
        └── 限流、可靠性与可观测组件
             │
             ├── MySQL
             ├── Redis
             ├── ChromaDB / Milvus
             ├── MinIO / OSS
             └── Elasticsearch（可选）
```

## 项目目录

```text
agentflow/
├── src/
│   ├── backend/
│   │   ├── agentflow/          # FastAPI 应用、编排、RAG、MCP 与基础设施代码
│   │   ├── pyproject.toml      # Python 项目与依赖定义
│   │   ├── requirements.txt    # pip 依赖清单
│   │   └── uv.lock             # uv 锁定文件
│   └── frontend/               # Vue 3 管理端
├── scripts/
│   └── migrations/             # 数据库增量迁移脚本
└── README.md
```

## 环境要求

- Python 3.12 或更高版本
- Node.js 20 或更高版本
- MySQL 8.x
- Redis 7.x
- MinIO 或兼容的对象存储
- ChromaDB、Milvus Lite 或独立 Milvus 服务
- Elasticsearch 8.x（仅在启用关键词检索时需要）

## 后端启动

进入后端目录并创建虚拟环境：

```bash
cd src/backend
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux 或 macOS：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

编辑 `src/backend/agentflow/config.yaml`，至少完成以下配置：

1. `mysql.endpoint` 和 `mysql.async_endpoint`：MySQL 同步、异步连接地址。
2. `redis.endpoint`：Redis 连接地址。
3. `multi_models`：所使用模型的 API Key、Base URL 和模型名称。
4. `rag.vector_db`：向量数据库模式与连接信息。
5. `storage`：MinIO 或 OSS 的访问参数。

配置完成后启动服务：

```bash
uvicorn agentflow.main:app --host 127.0.0.1 --port 7860
```

默认配置下，后端监听 `http://127.0.0.1:7860`。修改数据结构时，请按文件名顺序执行 `scripts/migrations/` 中需要的 SQL 脚本。

## 前端启动

```bash
cd src/frontend
npm ci
npm run dev
```

生产构建：

```bash
npm run build
```

前端请求地址和开发代理可在 `src/frontend/vite.config.ts` 中调整。

## 关键配置

| 配置项 | 用途 |
| --- | --- |
| `server` | 服务监听地址、端口、环境和版本 |
| `mysql` | 业务数据的同步与异步连接 |
| `redis` | 会话、限流、配额和缓存 |
| `multi_models` | 对话、推理、工具调用、视觉、Embedding 与 Rerank 模型 |
| `tools` | 天气、搜索、快递等外部工具凭据 |
| `rag` | 文档切分、召回阈值、向量库和关键词检索 |
| `storage` | MinIO 或 OSS 文件存储 |
| `reliability` | 并发上限、令牌桶、工具超时与输出预算 |
| `observability` | Prometheus 指标与 OTLP 上报 |
| `kafka` | 可选的长期记忆事件队列 |

## 数据与密钥安全

- 不要把真实 API Key、数据库密码、访问令牌或对象存储密钥提交到仓库。
- 部署前请替换 `config.yaml` 中的示例账号、密码和服务地址。
- 生产环境建议通过密钥管理系统或部署平台注入敏感配置，并限制 MySQL、Redis、Milvus 和 MinIO 的网络访问范围。
- 日志、上传文件、向量数据、数据库文件和实验输出应保存在仓库之外。

## 开发检查

前端类型检查：

```bash
cd src/frontend
npm run lint
```

前端构建检查：

```bash
npm run build
```

后端依赖以 `pyproject.toml`、`requirements.txt` 和 `uv.lock` 为准。增加功能时，请同步更新对应依赖清单和数据库迁移脚本。

## 许可证

本项目采用 MIT License，版权归 `ssssvbdd` 所有。完整条款请参阅 [LICENSE](LICENSE)。
