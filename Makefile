.PHONY: sync check test test-fast format

sync:
	uv sync

check:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy src/chord_unilab
	uv run pyright

test:
	CHORD_TEST_REQUIRE_LFS=1 uv run pytest -ra

test-fast:
	uv run pytest -ra -m "not newton"

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests
