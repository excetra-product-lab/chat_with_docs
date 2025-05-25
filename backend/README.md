# Chat With Docs Backend

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the application:
```bash
uvicorn app.main:app --reload
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
