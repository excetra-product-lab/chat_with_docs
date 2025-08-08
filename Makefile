.PHONY: help install-backend install-frontend install dev-backend dev-frontend dev test format lint clean

help:
	@echo "Available commands:"
	@echo "  make install        - Install all dependencies"
	@echo "  make dev           - Run both backend and frontend in development"
	@echo "  make dev-backend   - Run backend only"
	@echo "  make dev-frontend  - Run frontend only"
	@echo "  make test          - Run all tests"
	@echo "  make format        - Format code"
	@echo "  make lint          - Lint code"
	@echo "  make clean         - Clean temporary files"

install-backend:
	cd backend && uv sync

install-frontend:
	cd frontend && npm install

install: install-backend install-frontend

dev-backend:
	cd backend && . venv/bin/activate && \
	uvicorn app.main:app --reload

dev-frontend:
	cd frontend && npm run dev

dev:
	@echo "Starting backend and frontend..."
	@make -j 2 dev-backend dev-frontend

test:
	cd backend && . .venv/bin/activate && pytest
	cd frontend && npm test

format:
	cd backend && . venv/bin/activate && \
	black app tests && \
	isort app tests
	cd frontend && npm run format

lint:
	cd backend && . venv/bin/activate && \
	flake8 app tests && \
	mypy app
	cd frontend && npm run lint

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf backend/htmlcov
	rm -rf backend/.coverage
	rm -rf frontend/.next
	rm -rf frontend/out
