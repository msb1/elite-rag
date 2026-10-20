from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Configuration comes from the project .env file (or explicit constructor
        # values), never inherited shell variables or ambient secret files.
        return init_settings, dotenv_settings

    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "elite_rag_v2"
    qdrant_dense_vector_name: str = "dense"
    qdrant_sparse_vector_name: str = "sparse"
    rerank_url: str = Field(
        default="http://192.168.1.50:8000/v1/rerank",
        validation_alias="BGE_RERANK_URL",
    )
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_timeout_seconds: float = Field(default=180, gt=0)
    rerank_max_length: int = Field(default=512, ge=64)
    qdrant_bm25_vector_name: str = "sparse_bm25"
    bm25_model: str = "Qdrant/bm25"
    multi_source_types: str = (
        "confluence,fireflies,github,gmail,google_drive,hubspot,jira,linear,slack"
    )
    multi_chunk_max_tokens: int = Field(default=300, ge=96)
    multi_ingestion_report: str = "results/ingestion_multi.json"
    questions_file: str = "benchmark/questions.jsonl"
    multi_retrieval_output: str = "results/multi_retrieval_results.jsonl"
    retrieval_number: int = Field(default=50, ge=1)
    multi_document_score: str = "max"
    bge_embedding_max_length: int = Field(default=1024, ge=96)

    openai_local_endpoint: str = Field(
        default="http://127.0.0.1:1234/v1", validation_alias="OPENAI_LOCAL_ENDPOINT"
    )
    rag_model: str = "qwen2.5-7b-instruct-mlx"
    rag_api_key: str = Field(default="lm-studio", validation_alias="LLM_API_KEY")
    rag_temperature: float = Field(default=0.7, ge=0, le=2)
    rag_max_tokens: int = Field(default=2048, ge=1)
    rerank_top_n: int = Field(default=5, ge=1)
    progress_file: str = "results/progress.json"
    dataset_file: str = "results/results.jsonl"
    eval_output_file: str = Field(
        default="results/output.jsonl", validation_alias="OUTPUT_FILE"
    )
    hallucinations_output_file: str = "results/hallucinations.jsonl"
    retrieval_misses_output_file: str = "results/retrieval_misses.jsonl"
    analysis_output_file: str = "results/analysis.json"
    dashboard_output_file: str = "results/rag_metrics_dashboard.svg"
    server_host: str = "127.0.0.1"
    server_port: int = Field(default=8080, ge=1, le=65535)
    api_key: str = ""

    eval_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("EVAL_BASE_URL", "OPENAI_LOCAL_ENDPOINT"),
    )
    eval_model: str = "meta-llama-3.1-8b-instruct"
    eval_api_key: str = "local"

    embedding_model: str = "bge-m3"
    embedding_dimension: int = Field(default=1024, ge=1)
    embedding_base_url: str = Field(
        default="http://192.168.1.50:8000/v1",
        validation_alias=AliasChoices("OPENAI_REMOTE_ENDPOINT", "EMBEDDING_BASE_URL"),
    )
    embedding_api_key: str = "lm-studio"
    tokenizer_repository: str | None = None
    tokenizer_json_path: str | None = None
    source_types: str = Field(
        default=("confluence,fireflies,github,gmail,google_drive,hubspot,jira,linear,slack"),
        validation_alias="SOURCE_TYPES",
    )
    embedding_batch_size: int = Field(default=16, ge=1, le=128)

    chunk_target_tokens: int = Field(default=320, ge=64)
    chunk_max_tokens: int = Field(default=420, ge=96)
    chunk_overlap_tokens: int = Field(default=40, ge=0)
    ingestion_pipeline_version: str = "v3"

    rustfs_endpoint: str = Field(
        default="http://127.0.0.1:9000",
        validation_alias=AliasChoices("S3_ENDPOINT_URL", "RUSTFS_ENDPOINT"),
    )
    rustfs_access_key: str | None = Field(
        default=None, validation_alias=AliasChoices("S3_ACCESS_KEY", "RUSTFS_ACCESS_KEY")
    )
    rustfs_secret_key: str | None = Field(
        default=None, validation_alias=AliasChoices("S3_SECRET_KEY", "RUSTFS_SECRET_KEY")
    )
    rustfs_region: str = Field(
        default="us-east-1", validation_alias=AliasChoices("S3_REGION", "RUSTFS_REGION")
    )
    rustfs_bucket: str = Field(
        default="enterprise-rag", validation_alias=AliasChoices("S3_BUCKET", "RUSTFS_BUCKET")
    )
    rustfs_prefix: str = Field(
        default="docs",
        validation_alias=AliasChoices("S3_PATH_MARKDOWN_PREFIX", "RUSTFS_PREFIX"),
    )
    ingestion_database_url: str = ""
    log_file: str = "logs/elite-rag.log"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def valid_chunk_policy(self) -> Settings:
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("CHUNK_TARGET_TOKENS must not exceed CHUNK_MAX_TOKENS")
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TARGET_TOKENS")
        if self.tokenizer_repository is None:
            name = self.embedding_model.lower()
            if "embeddinggemma" in name:
                self.tokenizer_repository = "google/embeddinggemma-300m"
            elif name in {"bge-m3", "baai/bge-m3"}:
                self.tokenizer_repository = "BAAI/bge-m3"
            else:
                raise ValueError("Set TOKENIZER_REPOSITORY for the selected embedding model")
        return self

    @property
    def pipeline_signature(self) -> str:
        policy = {
            "version": self.ingestion_pipeline_version,
            "section_evidence_version": 1,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "tokenizer_repository": self.tokenizer_repository,
            "chunk_target_tokens": self.chunk_target_tokens,
            "chunk_max_tokens": self.chunk_max_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "sparse_model": "bge-m3-lexical",
        }
        digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:16]
        return f"{self.ingestion_pipeline_version}:{digest}"

    @property
    def configured_sources(self) -> tuple[str, ...]:
        return tuple(
            source.strip().lower() for source in self.source_types.split(",") if source.strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
