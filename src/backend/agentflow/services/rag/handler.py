from loguru import logger
from typing import Optional
from agentflow.core.run_context import get_current_invocation
from agentflow.reliability.circuit_breaker import circuit_registry
from agentflow.reliability.retry import RAG_RETRY_POLICY
from agentflow.services.rag.abstractions import RagQuery, get_default_retriever
from agentflow.services.rag.retrieval import MixRetrival
from agentflow.services.rewrite.query_write import query_rewriter
from agentflow.services.rag.es_client import client as es_client
from agentflow.services.rag.vector_stores import milvus_client
from agentflow.services.rag.rerank import Reranker
from agentflow.settings import app_settings

class RagHandler:

    @classmethod
    async def query_rewrite(cls, query):
        query_list = await query_rewriter.rewrite(query)
        return query_list

    @classmethod
    async def index_milvus_documents(cls, collection_name, chunks):
        await milvus_client.insert(collection_name, chunks)

    @classmethod
    async def index_es_documents(cls, index_name, chunks):
        await es_client.index_documents(index_name, chunks)

    @classmethod
    async def mix_retrival_documents(cls, query_list, knowledges_id, search_field="summary"):

        if app_settings.rag.enable_elasticsearch:
            es_documents, milvus_documents = await MixRetrival.mix_retrival_documents(query_list, knowledges_id, search_field)
            # 先对ES和Milvus结果分别排序
            es_documents.sort(key=lambda x: x.score, reverse=True)
            milvus_documents.sort(key=lambda x: x.score, reverse=True)
            all_documents = es_documents + milvus_documents
        else:
            all_documents = await MixRetrival.retrival_milvus_documents(query_list, knowledges_id, search_field)

        # 合并并去重，保留分数更高的文档
        documents = []
        seen_chunk_ids = set()

        # 按分数从高到低排序
        all_documents.sort(key=lambda x: x.score, reverse=True)
        
        # 去重，保留分数最高的
        for doc in all_documents:
            if doc.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(doc.chunk_id)
                documents.append(doc)
                if len(documents) >= 10:  # 限制返回10个文档
                    break
        
        return documents

    @classmethod
    async def _retrieve_via_pipeline(cls, query, knowledges_id, search_field, min_score, top_k, needs_query_rewrite):
        """委托给 6 步可插拔检索管线（查询增强 -> 混合检索 -> RRF 融合 -> 重排 -> Top-K）。

        管线内自动埋 RAG 指标（召回/rerank 耗时、Top-K 命中）与 span。
        """
        rag_query = RagQuery(
            query=query,
            knowledge_ids=list(knowledges_id) if knowledges_id else [],
            top_k=top_k if top_k is not None else 10,
            min_score=min_score,
            search_field=search_field,
            needs_query_rewrite=needs_query_rewrite,
            inv=get_current_invocation(),
        )
        result = await get_default_retriever().retrieve(rag_query)
        return result

    @classmethod
    async def rag_query_summary(cls, query, knowledges_id, min_score: Optional[float]=None,
                                top_k: Optional[int]=None, needs_query_rewrite: bool=True):
        if min_score is None:
            min_score = app_settings.rag.retrival.get('min_score')
        if top_k is None:
            top_k = app_settings.rag.retrival.get('top_k')

        async def retrieve():
            return await cls._retrieve_via_pipeline(
                query, knowledges_id, "summary", min_score, top_k, needs_query_rewrite
            )

        result = await circuit_registry.call(
            "rag:retrieve",
            lambda: RAG_RETRY_POLICY.run(retrieve),
            failure_threshold=5,
            cooldown_seconds=30.0,
        )

        if not result.documents:
            # summary 字段召回不足，降级用 content 字段召回（保持原有兜底行为）
            logger.info(f"Recall for summary Field numbers < top k, Start recall use content Field")
            return await cls.retrieve_ranked_documents(query, knowledges_id, knowledges_id)

        final_result = "\n".join(doc.content for doc in result.documents)
        return final_result


    @classmethod
    async def retrieve_ranked_documents(cls, query, collection_names, index_names=None, min_score: Optional[float]=None,
                        top_k: Optional[int]=None, needs_query_rewrite: bool=True):
        """
        处理 RAG 流程：查询重写、文档检索、重排序、结果过滤和拼接。

        内部已升级为可插拔检索管线（Retriever/VectorStore/Reranker 均可替换，
        融合策略支持加权与 RRF），对外签名保持不变。

        Args:
            query (str): 用户查询。
            collection_names (list[str]): 向量知识库 集合ID。
            index_names (list[str]): ES关键词库 集合ID。
            min_score (float): 文档最低分数阈值，默认为配置中的值。
            top_k (int): 召回文档的个数。
            needs_query_rewrite (bool): 是否需要开启Query重写，默认开启

        Returns:
            str: 拼接后的最终结果。
            """
        if min_score is None:
            min_score = app_settings.rag.retrival.get('min_score')
        if top_k is None:
            top_k = app_settings.rag.retrival.get('top_k')

        async def retrieve():
            return await cls._retrieve_via_pipeline(
                query, collection_names, "content", min_score, top_k, needs_query_rewrite
            )

        result = await circuit_registry.call(
            "rag:retrieve",
            lambda: RAG_RETRY_POLICY.run(retrieve),
            failure_threshold=5,
            cooldown_seconds=30.0,
        )

        # 处理空结果
        if not result.documents:
            return "No relevant documents found."

        final_result = "\n".join(doc.content for doc in result.documents)
        return final_result

    @classmethod
    async def delete_documents_es_milvus(cls, file_id, knowledge_id):
        if app_settings.rag.enable_elasticsearch:
            await es_client.delete_documents(file_id, knowledge_id)
        await milvus_client.delete_by_file_id(file_id, knowledge_id)
