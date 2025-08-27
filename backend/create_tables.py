#!/usr/bin/env python3
"""
Database table creation script for Chat with Docs.

This script creates all necessary tables in the database.
Run this once after setting up your PostgreSQL database.
"""

import sys
import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.settings import settings
from app.models.database import Base

def create_tables():
    """Create all database tables."""
    print(f"🔗 Connecting to database: {settings.DATABASE_URL}")
    
    try:
        # Create database engine
        engine = create_engine(settings.DATABASE_URL)
        
        # Test connection
        with engine.connect() as conn:
            print("✅ Database connection successful")
            
            # Enable pgvector extension if using PostgreSQL
            if "postgresql" in settings.DATABASE_URL:
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                    print("✅ pgvector extension enabled")
                except SQLAlchemyError as e:
                    print(f"⚠️  Could not enable pgvector extension: {e}")
                    print("   This is normal if pgvector is not installed")
        
        # Create all tables
        print("🛠️  Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ All tables created successfully!")
        
        # List created tables
        with engine.connect() as conn:
            if "postgresql" in settings.DATABASE_URL:
                result = conn.execute(text("""
                    SELECT tablename FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """))
                tables = [row[0] for row in result]
            else:
                # SQLite
                result = conn.execute(text("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table'
                    ORDER BY name
                """))
                tables = [row[0] for row in result]
            
            print(f"📋 Created tables: {', '.join(tables)}")
        
        return True
        
    except SQLAlchemyError as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Chat with Docs database...")
    
    success = create_tables()
    
    if success:
        print("\n✅ Database setup complete!")
        print("   You can now run your FastAPI application.")
    else:
        print("\n❌ Database setup failed!")
        print("   Please check your database connection and try again.")
        sys.exit(1)
