#!/bin/bash

echo "🔍 Checking TypeScript build..."

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Run type check
echo "🔧 Running TypeScript type check..."
npm run type-check

echo "✅ Build check complete!"