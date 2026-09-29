"""
RAG 检索链路抽象层：6 步可插拔管线（对齐 trpc-agent-go DefaultRetriever）。

    查询增强(可选) -> 向量化 -> 混合检索(BM25+Dense 融合) -> 融合去重 -> 重排(可选) -> Top-K

组件全部接口化（Embedder / VectorStore / QueryEnhancer / Reranker / ChunkingStrategy），
收益是可灰度、可 A/B、可插拔后端。

融合策略：加权融合（score 归一化后加权）或 RRF（Reciprocal Rank Fusion），
对齐 trpc-agent-go pgvector 后端的两种融合参考。

Agentic 过滤（对齐 NewAgenticFilterSearchTool）：
Agent 静态 filter + Runner 级 filter + LLM 动态 filter 用 AND 合并，既灵活又可控。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from agentflow.telemetry import metrics
from agentflow.telemetry.tracing import span_scope


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

class SearchMode(str, Enum):
    VECTOR = "vector"        # Dense（Milvus/Chroma）
    KEYWORD = "keyword"      # BM25（Elasticsearch）
    HYBRID = "hybrid"        # 融合检索


class MergeStrategy(str, Enum):
    WEIGHTED = "weighted"    # 分数归一化加权
    RRF = "rrf"              # Reciprocal Rank Fusion


@dataclass
class RagQuery:
    """检索请求：query + 元数据过滤 + 检索参数。"""
    query: str
    knowledge_ids: Sequence[str] = field(default_factory=list)
    top_k: int = 10
    min_score: Optional[float] = None
    search_field: str = "content"
    needs_query_rewrite: bool = True
    # Agentic 过滤：多来源 filter AND 合并（Agent 静态 + Runner 级 + LLM 动态）
    filters: List[Dict[str, Any]] = field(default_factory=list)
    include_content: bool = True
    inv: Any = None   # 可选 Invocation，用于链路追踪归因


@dataclass
class RetrievedDocument:
    chunk_id: str
    content: str
    score: float
    source: str = ""          # 命中后端：milvus / es / chroma
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieveResult:
    documents: List[RetrievedDocument]
    latency_ms: float = 0.0
    recall_count: int = 0
    rerank_count: int = 0

    def join_contents(self, separator: str = "\n") -> str:
        return separator.join(doc.content for doc in self.documents)


def merge_filters(*filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Agentic 过滤合并：多来源 filter 的键冲突时合并为 AND 列表（对齐 trpc-agent-go）。"""
    merged: Dict[str, Any] = {}
    for f in filters:
        if not f:
            continue
        for key, value in f.items():
            if key not in merged:
                merged[key] = value
            elif merged[key] != value:
                # 冲突 -> AND 语义：取交集表达
                existing = merged[key]
                merged[key] = {"$and": [existing, value]}
    return merged


# ---------------------------------------------------------------------------
# 可插拔组件契约
# ---------------------------------------------------------------------------

@runtime_checkable
class QueryEnhancer(Protocol):
    """查询增强（重写/扩展），对齐管线第 1 步（可选）。"""

    async def enhance(self, query: str) -> List[str]: ...


@runtime_checkable
class Embedder(Protocol):
    """向量化契约，对齐管线第 2 步。"""

    async def embed(self, texts: List[str]) -> List[List[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """向量库契约，对齐管线第 3 步。search(mode=hybrid|vector|keyword|filter)。"""

    async def search(
        self,
        query: str,
        collection_names: Sequence[str],
        top_k: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        filter: Optional[Dict[str, Any]] = None,
        search_field: str = "content",
    ) -> List[RetrievedDocument]: ...


@runtime_checkable
class RerankerComponent(Protocol):
    """重排契约，对齐管线第 4 步（可选）。"""

    async def rerank(self, query: str, documents: List[str]) -> List[Any]: ...


@runtime_checkable
class ChunkingStrategy(Protocol):
    """切分策略接口化：Token 步长+重叠滑窗 / DeepDOC 版面分析 各为一个 Strategy。"""

    def chunk(self, doc: Any) -> List[Any]: ...


class TokenSlidingWindowChunker:
    """固定窗口切分（Token 步长 + 重叠滑窗 naive_merge 的 Strategy 封装）。"""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Any) -> List[Any]:
        text = doc if isinstance(doc, str) else getattr(doc, "content", str(doc))
        tokens = text.split()
        if not tokens:
            return []
        step = max(self.chunk_size - self.overlap, 1)
        return [
            " ".join(tokens[i:i + self.chunk_size])
            for i in range(0, len(tokens), step)
        ]


# ---------------------------------------------------------------------------
# 融合算法
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[RetrievedDocument]],
    k: int = 60,
) -> List[RetrievedDocument]:
    """RRF：score = sum(1 / (k + rank))，对多路召回做排名融合。"""
    scores: Dict[str, float] = {}
    best: Dict[str, RetrievedDocument] = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            scores[doc.chunk_id] = scores.get(doc.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            best.setdefault(doc.chunk_id, doc)
    fused = [
        RetrievedDocument(
            chunk_id=cid, content=best[cid].content,
            score=score, source=best[cid].source, metadata=best[cid].metadata,
        )
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda d: d.score, reverse=True)
    return fused


def weighted_fusion(
    result_lists: Sequence[Sequence[RetrievedDocument]],
    weights: Optional[Sequence[float]] = None,
) -> List[RetrievedDocument]:
    """加权融合：各路分数 min-max 归一化后按权重相加，去重保留最高分。"""
    n = len(result_lists)
    weights = weights or [1.0 / n] * n
    seen: Dict[str, RetrievedDocument] = {}
    for results, weight in zip(result_lists, weights):
        if not results:
            continue
        lo = min(d.score for d in results)
        hi = max(d.score for d in results)
        span = (hi - lo) or 1.0
        for doc in results:
            normalized = (doc.score - lo) / span
            if doc.chunk_id in seen:
                seen[doc.chunk_id].score += weight * normalized
            else:
                doc_copy = RetrievedDocument(
                    chunk_id=doc.chunk_id, content=doc.content,
                    score=weight * normalized, source=doc.source, metadata=doc.metadata,
                )
                seen[doc.chunk_id] = doc_copy
    fused = list(seen.values())
    fused.sort(key=lambda d: d.score, reverse=True)
    return fused


# ---------------------------------------------------------------------------
# 默认实现：适配现有组件（MixRetrival / Reranker / query_rewriter）
# ---------------------------------------------------------------------------

class _ExistingHybridStore:
    """VectorStore 适配器：复用现有 MixRetrival（ES BM25 + Milvus Dense）。

    依赖（ES/Milvus 客户端）延迟到首次 search 时才导入，
    保证 abstractions 模块本身可独立导入（可测试性）。
    """

    def __init__(self, enable_elasticsearch: bool = True):
        self._mix = None
        self._es_enabled_flag = enable_elasticsearch

    def _ensure_mix(self):
        if self._mix is None:
            from agentflow.services.rag.retrieval import MixRetrival
            from agentflow.settings import app_settings
            self._mix = MixRetrival
            self._enable_es = self._es_enabled_flag and app_settings.rag.enable_elasticsearch
        return self._mix

    async def search(
        self,
        query: str,
        collection_names: Sequence[str],
        top_k: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        filter: Optional[Dict[str, Any]] = None,
        search_field: str = "content",
    ) -> List[RetrievedDocument]:
        # 现有 MixRetrival 按多 query 检索；单 query 场景包装为列表
        self._ensure_mix()
        query_list = query if isinstance(query, list) else [query]
        if mode == SearchMode.KEYWORD and self._enable_es:
            docs = await self._mix.retrival_es_documents(query_list, collection_names, search_field)
        elif mode == SearchMode.VECTOR or not self._enable_es:
            docs = await self._mix.retrival_milvus_documents(query_list, list(collection_names), search_field)
        else:
            es_docs, milvus_docs = await self._mix.mix_retrival_documents(
                query_list, list(collection_names), search_field
            )
            es_docs.sort(key=lambda x: x.score, reverse=True)
            milvus_docs.sort(key=lambda x: x.score, reverse=True)
            docs = es_docs + milvus_docs
        return [
            RetrievedDocument(
                chunk_id=getattr(d, "chunk_id", ""),
                content=getattr(d, "content", ""),
                score=getattr(d, "score", 0.0),
            )
            for d in (docs or [])
        ]


class _ExistingQueryEnhancer:
    """QueryEnhancer 适配器：复用现有 query_rewriter。"""

    async def enhance(self, query: str) -> List[str]:
        from agentflow.services.rewrite.query_write import query_rewriter
        return await query_rewriter.rewrite(query)


class _ExistingReranker:
    """Reranker 适配器：复用现有 Reranker.rerank_documents。"""

    async def rerank(self, query: str, documents: List[str]) -> List[Any]:
        from agentflow.services.rag.rerank import Reranker
        return await Reranker.rerank_documents(query, documents)


# ---------------------------------------------------------------------------
# DefaultRetriever：6 步管线编排
# ---------------------------------------------------------------------------

class DefaultRetriever:
    """可插拔检索管线（对齐 trpc-agent-go DefaultRetriever）。

    每一步埋 RAG 指标（召回耗时 / rerank 耗时 / Top-K 命中）+ span，
    量化检索质量与各阶段瓶颈。
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        query_enhancer: Optional[QueryEnhancer] = None,
        reranker: Optional[RerankerComponent] = None,
        merge_strategy: MergeStrategy = MergeStrategy.RRF,
    ):
        # 默认组件惰性构造（首次 retrieve 时才装配，导入零依赖）
        self.store = store
        self.query_enhancer = query_enhancer
        self.reranker = reranker
        self.merge_strategy = merge_strategy

    def _ensure_components(self) -> None:
        if self.store is None:
            self.store = _ExistingHybridStore()
        if self.query_enhancer is None:
            self.query_enhancer = _ExistingQueryEnhancer()
        if self.reranker is None:
            self.reranker = _ExistingReranker()

    async def retrieve(self, query: RagQuery) -> RetrieveResult:
        start = time.perf_counter()
        self._ensure_components()
        inv = getattr(query, "inv", None)

        with span_scope(inv, "rag.retrieve", **{"agchat.rag.top_k": query.top_k}):
            # 1. 查询增强（可选）
            enhance_start = time.perf_counter()
            if query.needs_query_rewrite and self.query_enhancer:
                queries = await self.query_enhancer.enhance(query.query)
            else:
                queries = [query.query]
            metrics.record_rag_stage("enhance", (time.perf_counter() - enhance_start) * 1000)

            # 2-3. 混合检索：多 query 各自检索后融合（加权 / RRF）
            search_start = time.perf_counter()
            per_query_hits: List[List[RetrievedDocument]] = []
            merged_filter = merge_filters(*query.filters) if query.filters else None
            for q in queries:
                hits = await self.store.search(
                    q, query.knowledge_ids,
                    top_k=query.top_k,
                    mode=SearchMode.HYBRID,
                    filter=merged_filter,
                    search_field=query.search_field,
                )
                per_query_hits.append(hits)
            metrics.record_rag_stage("search", (time.perf_counter() - search_start) * 1000)

            # 4. 融合 + 去重（保留分数更高者）
            if self.merge_strategy == MergeStrategy.RRF:
                candidates = reciprocal_rank_fusion(per_query_hits)
            else:
                candidates = weighted_fusion(per_query_hits)
            seen: set = set()
            deduped: List[RetrievedDocument] = []
            for doc in candidates:
                if doc.chunk_id and doc.chunk_id in seen:
                    continue
                if doc.chunk_id:
                    seen.add(doc.chunk_id)
                deduped.append(doc)
                if len(deduped) >= max(query.top_k * 3, 10):  # 召回池供 rerank 筛选
                    break
            recall_count = len(deduped)

            # 5. 重排（可选）
            rerank_start = time.perf_counter()
            if self.reranker and deduped:
                reranked = await self.reranker.rerank(
                    query.query, [doc.content for doc in deduped]
                )
                metrics.record_rag_stage("rerank", (time.perf_counter() - rerank_start) * 1000)
                content_to_doc = {doc.content: doc for doc in deduped}
                results: List[RetrievedDocument] = []
                for item in reranked or []:
                    content = getattr(item, "content", "")
                    score = getattr(item, "score", 0.0)
                    base = content_to_doc.get(content)
                    results.append(RetrievedDocument(
                        chunk_id=base.chunk_id if base else "",
                        content=content, score=score,
                        source=base.source if base else "",
                    ))
            else:
                results = deduped

            # 6. Top-K + min_score 过滤
            final = results[: query.top_k]
            if query.min_score is not None:
                final = [d for d in final if d.score >= query.min_score]

            metrics.record_rag_hits(rerank_count=len(results), top_k=len(final))
            latency_ms = (time.perf_counter() - start) * 1000
            return RetrieveResult(
                documents=final,
                latency_ms=latency_ms,
                recall_count=recall_count,
                rerank_count=len(results),
            )


# 进程级默认实例（惰性装配，组件可替换以做 A/B）
_default_retriever: Optional[DefaultRetriever] = None


def get_default_retriever() -> DefaultRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = DefaultRetriever()
    return _default_retriever
