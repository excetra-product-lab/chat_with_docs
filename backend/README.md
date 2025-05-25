# Chat With Docs Backend

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run the application:
```bash
uv run uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000
API documentation at http://localhost:8000/docs

## Testing

Run tests with:
```bash
pytest
```

## Deployment

Build and run with Docker:
```bash
docker build -t chat-with-docs-backend .
docker run -p 8000:8000 --env-file .env chat-with-docs-backend
```
