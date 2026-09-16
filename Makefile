.PHONY: install check test serve qdrant

install:
	uv sync --extra dev

check:
	uv run ruff check src tests
	uv run mypy src

test:
	uv run pytest -q

serve:
	uv run uvicorn elite_rag.api:app --host 0.0.0.0 --port 8080

qdrant:
	docker compose up -d qdrant
