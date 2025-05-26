#!/bin/bash

# Script to run all CI checks locally before pushing
# This mimics what GitHub Actions will run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
OVERALL_STATUS=0

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        OVERALL_STATUS=1
    fi
}

print_header() {
    echo ""
    echo -e "${YELLOW}$1${NC}"
    echo "----------------------------------------"
}

# Main script
echo "🏃 Running CI checks locally"
echo "============================"

# Backend checks
if [ -d "backend" ]; then
    print_header "🐍 Backend CI Checks"

    cd backend

    # Check if using uv
    if [ -f "pyproject.toml" ]; then
        # Install/sync dependencies
        echo "Installing dependencies..."
        uv sync

        # Black formatting check
        echo -n "Running Black formatter check... "
        if uv run black --check app tests >/dev/null 2>&1; then
            print_status 0 "Black formatting"
        else
            print_status 1 "Black formatting (run: uv run black app tests)"
        fi

        # isort import sorting check
        echo -n "Running isort import check... "
        if uv run isort --check-only app tests >/dev/null 2>&1; then
            print_status 0 "Import sorting"
        else
            print_status 1 "Import sorting (run: uv run isort app tests)"
        fi

        # Flake8 linting
        echo -n "Running Flake8 linter... "
        if uv run flake8 app tests --max-line-length=100 --extend-ignore=E203,W503 >/dev/null 2>&1; then
            print_status 0 "Flake8 linting"
        else
            print_status 1 "Flake8 linting"
            echo "  Run: uv run flake8 app tests --max-line-length=100 --extend-ignore=E203,W503"
        fi

        # MyPy type checking
        echo -n "Running MyPy type checker... "
        if uv run mypy app --ignore-missing-imports >/dev/null 2>&1; then
            print_status 0 "MyPy type checking"
        else
            print_status 1 "MyPy type checking"
            echo "  Run: uv run mypy app --ignore-missing-imports"
        fi

        # Run tests
        echo -n "Running pytest... "
        # Set test environment variables safely
        export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}./"
        export DATABASE_URL="sqlite:///./test.db"
        export SECRET_KEY="test-secret-key"
        export AZURE_OPENAI_API_KEY="test-key"
        export AZURE_OPENAI_ENDPOINT="test-endpoint"

        # Run pytest with timeout (using Python's timeout mechanism)
        if python -c "
import subprocess
import sys
try:
    result = subprocess.run(['uv', 'run', 'pytest', '-q'],
                          timeout=30,
                          capture_output=True,
                          text=True)
    sys.exit(result.returncode)
except subprocess.TimeoutExpired:
    print('Tests timed out after 30 seconds')
    sys.exit(1)
" >/dev/null 2>&1; then
            print_status 0 "Backend tests"
        else
            print_status 1 "Backend tests"
            echo "  Run: uv run pytest -v"
        fi
    else
        echo "⚠️  No pyproject.toml found, using pip..."
        # Fallback to regular pip/pytest
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt
        fi
        pytest -q || print_status 1 "Backend tests"
    fi

    cd ..
fi

# Frontend checks
if [ -d "frontend" ]; then
    print_header "🎨 Frontend CI Checks"

    cd frontend

    if [ -f "package.json" ]; then
        # Install dependencies
        echo "Installing dependencies..."
        npm install --silent

        # ESLint
        echo -n "Running ESLint... "
        if npm run lint >/dev/null 2>&1; then
            print_status 0 "ESLint"
        else
            print_status 1 "ESLint (run: npm run lint)"
        fi

        # TypeScript check
        echo -n "Running TypeScript check... "
        if npx tsc --noEmit >/dev/null 2>&1; then
            print_status 0 "TypeScript"
        else
            print_status 1 "TypeScript (run: npx tsc --noEmit)"
        fi

        # Build check
        echo -n "Running build... "
        if npm run build >/dev/null 2>&1; then
            print_status 0 "Build"
        else
            print_status 1 "Build (run: npm run build)"
        fi

        # Tests (if configured)
        if npm run | grep -q " test"; then
            echo -n "Running tests... "
            if npm test >/dev/null 2>&1; then
                print_status 0 "Frontend tests"
            else
                print_status 1 "Frontend tests (run: npm test)"
            fi
        fi

        # Security audit
        echo -n "Running security audit... "
        # Run audit and capture output (allow non-zero exit codes)
        audit_output=$(npm audit --production 2>&1 || true)
        if echo "$audit_output" | grep -q "found 0 vulnerabilities"; then
            print_status 0 "Security audit"
        elif echo "$audit_output" | grep -q "critical\|high"; then
            print_status 1 "Security audit (run: npm audit fix)"
        else
            print_status 0 "Security audit (minor issues)"
        fi
    fi

    cd ..
fi

# Summary
print_header "📊 Summary"

if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}All CI checks passed! 🎉${NC}"
    echo "Your code is ready to be pushed."
else
    echo -e "${RED}Some CI checks failed! 😞${NC}"
    echo ""
    echo "Please fix the issues above before pushing."
    echo "You can use the suggested commands to fix most issues automatically."
    echo ""
    echo "For a quick fix of formatting issues, run:"
    echo "  cd backend && uv run black app tests && uv run isort app tests"
    echo "  cd frontend && npm run lint -- --fix"
fi

exit $OVERALL_STATUS
