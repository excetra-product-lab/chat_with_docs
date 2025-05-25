#!/bin/bash

echo "Setting up Chat With Docs project..."

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting." >&2; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed. Aborting." >&2; exit 1; }

# Backend setup
echo "Setting up backend with uv..."
cd backend
# Initialize uv project (if not already)
if [ ! -f pyproject.toml ]; then
  uv init --non-interactive
  # Add existing deps (optional)
  # uv add -r requirements.txt
fi
# Sync env
uv sync
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
