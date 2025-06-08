#!/usr/bin/env python3
"""
Simple script to run the Pokémon RAG Chat Flask application.
"""
import os
import sys
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Set a default OpenAI API key if none is provided
if not os.getenv('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = 'your_openai_api_key_here'

from src.app import app

if __name__ == '__main__':
    print("🔥 Starting Pokémon RAG Chat Server...")
    print("📱 Open your browser to: http://localhost:5001")
    print("💡 Note: Add your OpenAI API key to .env file for LLM responses")
    print("🤖 Without an API key, you'll get rule-based responses")
    print("-" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
