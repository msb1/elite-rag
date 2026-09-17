from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from the single project ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    qdrant_url: str = "http://192.168.1.50:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "elite_rag"
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "sparse"
    sparse_embedding_model: str = "Qdrant/minicoil-v1"
    sparse_embedding_url: str = "http://192.168.1.50:8000/v1/embeddings/sparse"
    sparse_embedding_batch_size: int = Field(default=8, ge=1, le=64)
    sparse_embedding_timeout_seconds: float = Field(default=30.0, gt=0)
    hybrid_candidate_limit: int = Field(default=100, ge=1, le=1_000)

    # rag-bench's local OpenAI-compatible embedding service.
    embedding_model: str = "text-embedding-embeddinggemma-300m"
    embedding_dimension: int = Field(default=768, ge=1)
    embedding_base_url: str = Field(
        default="http://192.168.1.50:1234/v1",
        validation_alias=AliasChoices("OPENAI_LOCAL_ENDPOINT", "EMBEDDING_BASE_URL"),
    )
    embedding_api_key: str = Field(default="lm-studio", validation_alias="EMBEDDING_API_KEY")
    child_max_tokens: int = Field(default=150, ge=16)
    parent_max_tokens: int = Field(default=2000, ge=128)
    ingest_batch_size: int = Field(default=64, ge=1)
    log_file: str = "logs/elite-rag.log"
    log_level: str = "INFO"
    missing_docs_file: str = "logs/missing_docs.json"

    # Keep internal names stable while accepting rag-bench's S3 names directly.
    rustfs_endpoint: str = Field(
        default="http://192.168.1.50:9000",
        validation_alias=AliasChoices("S3_ENDPOINT_URL", "RUSTFS_ENDPOINT"),
    )
    rustfs_access_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("S3_ACCESS_KEY", "RUSTFS_ACCESS_KEY"),
    )
    rustfs_secret_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("S3_SECRET_KEY", "RUSTFS_SECRET_KEY"),
    )
    rustfs_region: str = "us-east-1"
    rustfs_bucket: str = Field(
        default="enterprise-rag",
        validation_alias=AliasChoices("S3_BUCKET", "RUSTFS_BUCKET"),
    )
    rustfs_prefix: str = Field(
        default="documents",
        validation_alias=AliasChoices("S3_PATH_MARKDOWN_PREFIX", "RUSTFS_PREFIX"),
    )

    # rag-bench's local FastAPI service backed by Jina reranker v3.5.
    jina_rerank_url: str = Field(
        default="http://192.168.1.50:8000/v1/rerank",
        validation_alias="JINA_RERANK_URL",
    )
    jina_rerank_model: str = Field(
        default="jina-reranker-v3.5",
        validation_alias="JINA_RERANK_MODEL",
    )
    rerank_top_n: int = Field(default=5, ge=1)
    rerank_confidence_threshold: float = Field(default=0.35, ge=0, le=1)
    primary_vector_limit: int = Field(default=25, ge=1)
    fallback_vector_limit: int = Field(default=50, ge=1)

    # rag-bench's local Qwen generation model.
    llm_base_url: str = Field(
        default="http://127.0.0.1:1234/v1",
        validation_alias=AliasChoices("OPENAI_LOCAL_ENDPOINT", "LLM_BASE_URL"),
    )
    llm_api_key: str = Field(default="lm-studio", validation_alias="LLM_API_KEY")
    llm_model: str = Field(
        default="qwen2.5-7b-instruct-mlx",
        validation_alias=AliasChoices("RAG_MODEL", "LLM_MODEL"),
    )
    llm_max_tokens: int = Field(default=1024, ge=1)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_top_p: float = Field(default=0.1, gt=0, le=1)

    # rag-bench's local Llama 3.1 evaluator uses the same endpoint.
    eval_model: str = Field(
        default="meta-llama-3.1-8b-instruct",
        validation_alias="EVAL_MODEL",
    )
    eval_base_url: str = Field(
        default="http://127.0.0.1:1234/v1",
        validation_alias=AliasChoices("OPENAI_LOCAL_ENDPOINT", "EVAL_BASE_URL"),
    )
    eval_api_key: str = Field(default="lm-studio", validation_alias="EVAL_API_KEY")
    eval_timeout_seconds: float = Field(default=30.0, gt=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
