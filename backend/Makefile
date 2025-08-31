.PHONY: install lint format type-check test ci ci-with-summary

install:
	@echo "📦 Installing dependencies..."
	uv sync --frozen

lint:
	@echo "🔍 Running linting..."
	uv run ruff check app tests

format:
	@echo "✨ Formatting code..."
	uv run ruff format app tests

type-check:
	@echo "🔧 Type checking..."
	uv run mypy app --ignore-missing-imports --explicit-package-bases

test:
	@echo "🧪 Running tests..."
	uv run pytest -v --cov=app --cov-report=xml --cov-report=html

ci: install lint format type-check test
	@echo "✅ All CI checks passed!"
