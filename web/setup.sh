#!/bin/bash
# Setup script for the web application

set -e

echo "🚀 Setting up Brain Stroke Segmentation Web App..."

# Backend setup
echo "📦 Setting up Backend..."
cd web/backend

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file from .env.example"
fi

# Install Python dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Run migrations
echo "🗄️ Running database migrations..."
python manage.py migrate

echo "✅ Backend setup complete!"

# Frontend setup
echo "📦 Setting up Frontend..."
cd ../frontend

# Install Node dependencies
echo "📥 Installing Node dependencies..."
npm install

echo "✅ Frontend setup complete!"

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ Setup Complete!"
echo "═══════════════════════════════════════════════"
echo ""
echo "To start the application:"
echo ""
echo "1. Backend (Terminal 1):"
echo "   cd web/backend"
echo "   python manage.py runserver"
echo ""
echo "2. Frontend (Terminal 2):"
echo "   cd web/frontend"
echo "   npm start"
echo ""
echo "Then open http://localhost:3000 in your browser"
echo ""
