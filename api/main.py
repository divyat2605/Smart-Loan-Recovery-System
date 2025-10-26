from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn
import sys
import os

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import routes using the full path from api package
from api.routes import train, predict, monitoring, features  # Add features

app = FastAPI(
    title="Smart Loan Recovery System API",
    description="ML-powered loan recovery prediction and strategy system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(train.router, prefix="/api/v1", tags=["training"])
app.include_router(predict.router, prefix="/api/v1", tags=["prediction"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["monitoring"])
app.include_router(features.router, prefix="/api/v1", tags=["features"])  # Add this

@app.get("/")
async def root():
    return {
        "message": "Smart Loan Recovery System API",
        "version": "1.0.0",
        "endpoints": {
            "training": "/api/v1/train",
            "prediction": "/api/v1/predict",
            "monitoring": "/api/v1/monitoring/status",
            "features": "/api/v1/features/analysis"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    print("🚀 Starting Smart Loan Recovery API...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)