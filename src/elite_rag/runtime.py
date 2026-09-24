from __future__ import annotations

from dataclasses import dataclass

from elite_rag.chunking import HierarchicalChunker
from elite_rag.config import Settings
from elite_rag.embedding import EmbeddingGemma
from elite_rag.generation import EnterpriseGenerationEngine
from elite_rag.ingestion import IngestionEngine
from elite_rag.intent import DeterministicIntentRouter
from elite_rag.missing_documents import MissingDocumentRecorder
from elite_rag.parsing import default_registry
from elite_rag.reranking import JinaReranker
from elite_rag.retrieval import TwoPassDeterministicRetriever
from elite_rag.sparse_embedding import MiniCOILSparseEmbedder
from elite_rag.vector_store import QdrantVectorStore


@dataclass(slots=True)
class Runtime:
    store: QdrantVectorStore
    ingestion: IngestionEngine
    retrieval: TwoPassDeterministicRetriever
    generation: EnterpriseGenerationEngine


def build_runtime(settings: Settings) -> Runtime:
    embedder = EmbeddingGemma(
        settings.embedding_model,
        settings.embedding_base_url,
        settings.embedding_api_key,
    )
    store = QdrantVectorStore(
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.embedding_dimension,
        settings.qdrant_api_key,
        dense_vector_name=settings.qdrant_dense_vector_name,
        sparse_vector_name=settings.qdrant_sparse_vector_name,
        upsert_max_attempts=settings.qdrant_upsert_max_attempts,
        upsert_initial_backoff_seconds=settings.qdrant_upsert_initial_backoff_seconds,
        upsert_max_backoff_seconds=settings.qdrant_upsert_max_backoff_seconds,
        sparse_embedder=MiniCOILSparseEmbedder(
            settings.sparse_embedding_url,
            settings.sparse_embedding_model,
            settings.sparse_embedding_batch_size,
            settings.sparse_embedding_timeout_seconds,
            settings.sparse_embedding_max_attempts,
            settings.sparse_embedding_initial_backoff_seconds,
            settings.sparse_embedding_max_backoff_seconds,
        ),
        hybrid_candidate_limit=settings.hybrid_candidate_limit,
    )
    chunker = HierarchicalChunker(
        embedder.count_tokens,
        child_max_tokens=settings.child_max_tokens,
        parent_max_tokens=settings.parent_max_tokens,
    )
    ingestion = IngestionEngine(
        default_registry(),
        chunker,
        embedder,
        store,
        settings.ingest_batch_size,
        MissingDocumentRecorder(settings.missing_docs_file),
    )
    reranker = (
        JinaReranker(
            settings.jina_rerank_url,
            settings.jina_rerank_model,
            settings.rerank_top_n,
        )
    )
    retrieval = TwoPassDeterministicRetriever(
        store,
        embedder,
        DeterministicIntentRouter(),
        reranker,
        confidence_threshold=settings.rerank_confidence_threshold,
        primary_limit=settings.primary_vector_limit,
        fallback_limit=settings.fallback_vector_limit,
        top_n=settings.rerank_top_n,
    )
    generation = EnterpriseGenerationEngine(
        settings.llm_base_url,
        settings.llm_api_key,
        settings.llm_model,
        temperature=settings.llm_temperature,
        top_p=settings.llm_top_p,
        max_tokens=settings.llm_max_tokens,
    )
    return Runtime(store, ingestion, retrieval, generation)
