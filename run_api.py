"""
Run the FastAPI application from the root directory.
This ensures proper module imports.
"""
import uvicorn
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Import the app from api.main
    from api.main import app
    
    print("🚀 Starting Smart Loan Recovery API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)