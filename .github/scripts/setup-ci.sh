#!/bin/bash

# Setup script for GitHub Actions CI/CD
# This script helps set up the CI environment locally for testing

set -e

echo "🚀 Setting up CI/CD for Chat With Docs"
echo "======================================"

# Create .github directories if they don't exist
# mkdir -p .github/workflows
# mkdir -p .github/scripts

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

if ! command_exists node; then
    echo "❌ Node.js is not installed"
    exit 1
fi

if ! command_exists docker; then
    echo "⚠️  Docker is not installed (optional, needed for act)"
fi

echo "✅ Prerequisites checked"

# Install pre-commit
echo ""
echo "📦 Installing pre-commit..."
if ! command_exists pre-commit; then
    uv add pre-commit
fi

# Install pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  No .pre-commit-config.yaml found"
fi

# Install act for local GitHub Actions testing (optional)
echo ""
echo "📦 Installing act (GitHub Actions local runner)..."
if ! command_exists act && command_exists docker; then
    case "$(uname -s)" in
        Darwin)
            if command_exists brew; then
                brew install act
            else
                echo "⚠️  Please install act manually: https://github.com/nektos/act"
            fi
            ;;
        Linux)
            curl -s https://api.github.com/repos/nektos/act/releases/latest | \
                grep "browser_download_url.*Linux_x86_64.tar.gz" | \
                cut -d : -f 2,3 | \
                tr -d \" | \
                wget -qi - -O - | \
                sudo tar -xz -C /usr/local/bin act
            ;;
        *)
            echo "⚠️  Please install act manually: https://github.com/nektos/act"
            ;;
    esac
fi

# Create .secrets file for act (local testing)
if command_exists act && [ ! -f ".secrets" ]; then
    echo ""
    echo "📝 Creating .secrets file for local testing with act..."
    cat > .secrets << EOF
# Secrets for local GitHub Actions testing with act
# DO NOT COMMIT THIS FILE
CODECOV_TOKEN=test-codecov-token
EOF
    echo "✅ .secrets file created (remember to add real values for testing)"
fi

# Create .actrc file for act configuration
if command_exists act && [ ! -f ".actrc" ]; then
    echo ""
    echo "📝 Creating .actrc file for act configuration..."
    cat > .actrc << EOF
# Configuration for act (GitHub Actions local runner)
-P ubuntu-latest=ghcr.io/catthehacker/ubuntu:act-latest
-P ubuntu-22.04=ghcr.io/catthehacker/ubuntu:act-22.04
-P ubuntu-20.04=ghcr.io/catthehacker/ubuntu:act-20.04
--container-architecture linux/amd64
EOF
    echo "✅ .actrc file created"
fi

# Set up backend testing environment
echo ""
echo "🔧 Setting up backend testing environment..."
cd backend
if [ ! -f ".env.test" ]; then
    cp .env.example .env.test
    echo "✅ Created .env.test for backend"
fi
cd ..

# Set up frontend testing environment
echo ""
echo "🔧 Setting up frontend testing environment..."
cd frontend
if [ ! -f ".env.test" ]; then
    cp .env.local.example .env.test
    echo "✅ Created .env.test for frontend"
fi
cd ..

# Run initial checks
echo ""
echo "🏃 Running initial CI checks..."

# Backend checks
echo ""
echo "Backend checks:"
cd backend
if [ -f "pyproject.toml" ]; then
    echo "- Installing dependencies with uv..."
    uv sync
    echo "- Running black..."
    uv run black --check app tests || echo "  ⚠️  Some files need formatting"
    echo "- Running isort..."
    uv run isort --check-only app tests || echo "  ⚠️  Some imports need sorting"
    echo "- Running flake8..."
    uv run flake8 app tests --max-line-length=100 --extend-ignore=E203,W503 || echo "  ⚠️  Some linting issues found"
else
    echo "⚠️  No pyproject.toml found, skipping backend checks"
fi
cd ..

# Frontend checks
echo ""
echo "Frontend checks:"
cd frontend
if [ -f "package.json" ]; then
    echo "- Installing dependencies..."
    npm install
    echo "- Running ESLint..."
    npm run lint || echo "  ⚠️  Some linting issues found"
    echo "- Running TypeScript check..."
    npx tsc --noEmit || echo "  ⚠️  Some type errors found"
else
    echo "⚠️  No package.json found, skipping frontend checks"
fi
cd ..

# Summary
echo ""
echo "✅ CI/CD setup complete!"
echo ""
echo "📚 Next steps:"
echo ""
echo "1. Test GitHub Actions locally:"
echo "   act pull_request    # Test PR workflows"
echo "   act push           # Test push workflows"
echo ""
echo "2. Run pre-commit hooks:"
echo "   pre-commit run --all-files"
echo ""
echo "3. Fix any issues found:"
echo "   cd backend && uv run black app tests"
echo "   cd backend && uv run isort app tests"
echo "   cd frontend && npm run lint -- --fix"
echo ""
echo "4. Commit the workflow files to your repository"
echo ""
echo "Happy coding! 🎉"
