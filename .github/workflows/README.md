# GitHub Actions Workflows

This directory contains the CI/CD workflows for the Chat With Docs project.

## Workflows

### 1. Backend CI (`backend-ci.yml`)

- **Triggers**: Pull requests and pushes that modify backend code
- **Actions**:
  - Sets up Python 3.11 environment with `uv`
  - Installs dependencies using `uv sync`
  - Runs code formatting checks (Black)
  - Runs import sorting checks (isort)
  - Runs linting (Flake8)
  - Runs type checking (MyPy)
  - Runs tests with pytest and coverage
  - Uploads coverage reports to Codecov
- **Database**: Uses PostgreSQL with pgvector for integration tests

### 2. Frontend CI (`frontend-ci.yml`)

- **Triggers**: Pull requests and pushes that modify frontend code
- **Actions**:
  - Tests on Node.js 18.x and 20.x
  - Installs dependencies with `npm ci`
  - Runs ESLint for code quality
  - Performs TypeScript type checking
  - Builds the Next.js application
  - Runs security audit

### 3. Full Stack CI (`ci.yml`)

- **Triggers**: All pull requests and pushes to main/develop branches
- **Actions**:
  - Runs both backend and frontend checks in parallel
  - Provides a unified CI status check
  - Uploads test artifacts

### 4. Dependency Review (`dependency-review.yml`)

- **Triggers**: Pull requests that modify dependency files
- **Actions**:
  - Reviews dependency changes for security vulnerabilities
  - Runs Python Safety check
  - Performs npm audit for frontend dependencies
  - Generates vulnerability reports

## Local Development

### Pre-commit Hooks

To ensure code quality before pushing, install pre-commit:

```bash
pip install pre-commit
pre-commit install
```

This will run the following checks automatically:

- Code formatting (Black for Python, ESLint for JS/TS)
- Import sorting (isort)
- Linting (Flake8, ESLint)
- Type checking (MyPy, TypeScript)
- Security checks (detect-secrets)
- And more...

### Running Checks Locally

#### Backend

```bash
cd backend
uv run black app tests
uv run isort app tests
uv run flake8 app tests
uv run mypy app
uv run pytest
```

#### Frontend

```bash
cd frontend
npm run lint
npx tsc --noEmit
npm run build
npm test  # If tests are configured
```

## Configuration

### Required Secrets

The following secrets should be configured in your GitHub repository:

- `CODECOV_TOKEN` (optional): For uploading coverage reports
- Any deployment-related secrets for production workflows

### Environment Variables

The workflows automatically create test environment files with placeholder
values. For production deployments, ensure proper secrets are
configured.

## Best Practices

1. **Keep workflows DRY**: Use composite actions for repeated steps
2. **Cache dependencies**: All workflows use caching for faster runs
3. **Run in parallel**: Backend and frontend checks run simultaneously
4. **Fail fast**: Linting issues don't block test runs but are reported
5. **Matrix testing**: Frontend tests against multiple Node.js versions

## Troubleshooting

### Common Issues

1. **uv sync fails**: Ensure `uv.lock` is committed and up to date
2. **Database connection errors**: Check PostgreSQL service is healthy
3. **ESLint errors**: Run `npm run lint -- --fix` locally to auto-fix
4. **Type errors**: Ensure all TypeScript types are properly defined

### Debugging Workflows

- Check the Actions tab in GitHub for detailed logs
- Use `act` locally to test workflows: `act pull_request`
- Add `continue-on-error: true` temporarily for debugging

## Contributing

When adding new workflows:

1. Test locally using `act` if possible
2. Keep workflows focused and single-purpose
3. Add appropriate caching strategies
4. Document any new requirements here
5. Consider the impact on CI runtime and costs
