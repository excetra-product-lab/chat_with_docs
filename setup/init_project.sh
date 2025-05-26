#!/bin/bash
# Complete project initialization script

echo "🚀 Initializing Chat With Docs (RAG) Project"
echo "==========================================="

# Create root project directory
# mkdir -p chat-with-docs
# cd chat-with-docs

# Initialize git repository
# git init

# Create the main README (copy from documents)
cat > README.md << 'EOF'
# Chat With Docs (RAG)
## Tech Stack
| Layer             | Choice                          | Rationale                           |
| ----------------- | ------------------------------- | ----------------------------------- |
| Vector store      | **pgvector (managed Supabase)** | 1-click, UK region, SQL familiarity |
| Embeddings & chat | Azure OpenAI - gpt-4o           | Enterprise SLA, GDPR-aligned        |
| API               | FastAPI                         | Async, quick setup                  |
| Front-end         | Next.js + shadcn/ui             | Rapid UI, SSR                       |
| Auth              | Clerk.dev free tier             | Offload security; JWT passthrough   |
| Hosting           | Fly.io UK or Railway            | Minutes to deploy, EU data          |

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL with pgvector extension (or Supabase account)
- Azure OpenAI API access
- Clerk.dev account

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
cp .env.local.example .env.local
# Edit .env.local with your Clerk keys
npm run dev
```

### Development
Both servers should now be running:
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

## Architecture

See the architecture diagram in the full README for details on the RAG pipeline and system design.

## Deployment

### Backend (Fly.io)
```bash
cd backend
fly launch
fly deploy
```

### Frontend (Vercel)
```bash
cd frontend
vercel
```

## Features
- 🔒 Secure authentication with Clerk
- 📄 Multi-format document upload (PDF, DOCX, TXT)
- 🔍 Semantic search with pgvector
- 💬 AI-powered Q&A with GPT-4o
- 📚 Automatic citation generation
- ✅ Source verification
- 🇪🇺 GDPR compliant hosting
EOF

# Copy the provided gitignore
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml
.pdm-python
.pdm-build/

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# Ruff
.ruff_cache/

# PyPI configuration file
.pypirc

# Node modules
node_modules/

# Next.js
.next/
out/

# Environment variables
.env.local
.env.development.local
.env.test.local
.env.production.local

# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*

# OS
.DS_Store
Thumbs.db

# Editor
.vscode/
*.swp
*.swo
*~

# Testing
coverage/
.nyc_output/
EOF

# Create .env.example for docker-compose
cat > .env.example << 'EOF'
# Database Configuration for Docker Compose
POSTGRES_USER=chatwithdocs
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=chatwithdocs
EOF

# Create docker-compose.yml for local development
cat > docker-compose.yml << 'EOF'
version: '3.8'

# Load environment variables from .env file
# Copy .env.example to .env and customize your values
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-chatwithdocs}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-your_secure_password_here}
      POSTGRES_DB: ${POSTGRES_DB:-chatwithdocs}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-chatwithdocs}:${POSTGRES_PASSWORD:-your_secure_password_here}@postgres:5432/${POSTGRES_DB:-chatwithdocs}
    depends_on:
      - postgres
    volumes:
      - ./backend/app:/app/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

volumes:
  postgres_data:
EOF

# Create Makefile for common tasks
cat > Makefile << 'EOF'
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
	cd backend && python -m venv venv && \
	. venv/bin/activate && \
	pip install -r requirements.txt

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
	cd backend && . venv/bin/activate && pytest
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
EOF

# Create VS Code workspace settings
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "python.defaultInterpreterPath": "./backend/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
EOF

# Create launch.json for debugging
cat > .vscode/launch.json << 'EOF'
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Backend: FastAPI",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "app.main:app",
        "--reload"
      ],
      "cwd": "${workspaceFolder}/backend",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/backend"
      }
    },
    {
      "name": "Frontend: Next.js",
      "type": "node",
      "request": "launch",
      "runtimeExecutable": "npm",
      "runtimeArgs": ["run", "dev"],
      "cwd": "${workspaceFolder}/frontend",
      "skipFiles": ["<node_internals>/**"]
    }
  ],
  "compounds": [
    {
      "name": "Full Stack",
      "configurations": ["Backend: FastAPI", "Frontend: Next.js"]
    }
  ]
}
EOF

# Create setup script
cat > setup.sh << 'EOF'
#!/bin/bash

echo "Setting up Chat With Docs project..."

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting." >&2; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed. Aborting." >&2; exit 1; }

# Backend setup
echo "Setting up backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
echo "✅ Backend setup complete"

# Frontend setup
echo "Setting up frontend..."
cd ../frontend
npm install
cp .env.local.example .env.local
echo "✅ Frontend setup complete"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit backend/.env with your Azure OpenAI and database credentials"
echo "2. Edit frontend/.env.local with your Clerk.dev keys"
echo "3. Run 'make dev' to start both servers"
echo ""
echo "Happy coding! 🚀"
EOF

chmod +x setup.sh

echo ""
echo "✅ Project structure created successfully!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and set your database credentials"
echo "2. Run './setup.sh' to install dependencies"
echo "3. Configure your environment variables"
echo "4. Set up your Supabase database with pgvector"
echo "5. Create a Clerk.dev application"
echo "6. Run 'make dev' to start development"
echo ""
echo "Project structure:"
echo "- Backend: FastAPI with Python"
echo "- Frontend: Next.js with TypeScript and Tailwind CSS"
echo "- Authentication: Clerk.dev"
echo "- Vector Database: PostgreSQL with pgvector"
echo "- AI: Azure OpenAI"
