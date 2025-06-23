# Chat With Docs Backend

## Setup

1. Install dependencies:

```bash
uv sync
```

1. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

1. Run the application:

```bash
uv run uvicorn app.main:app --reload
```

## API Testing & Documentation

Once your backend server is running, you can view and test the API endpoints:

1. **Interactive API Documentation**: Visit <http://localhost:8000/docs> to access the automatically generated Swagger UI documentation
2. **Test API Endpoints**: The Swagger UI allows you to test all API endpoints directly from your browser:
   - Click on any endpoint to expand its details
   - Use the "Try it out" button to make actual API calls
   - View request/response schemas and examples
   - Test authentication flows and data operations
3. **Alternative Documentation**: Visit <http://localhost:8000/redoc> for ReDoc-style documentation (read-only)

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
