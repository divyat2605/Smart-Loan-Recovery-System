from fastapi import APIRouter
import os
import json
from datetime import datetime

router = APIRouter()

@router.get("/monitoring/logs")
async def get_logs():
    """
    Retrieve the last 20 lines from the application log file.
    """
    try:
        if os.path.exists("logs/app.log"):
            with open("logs/app.log", "r") as file:
                logs = file.readlines()[-20:]
                return {"logs": logs, "total_lines": len(logs)}
        return {"message": "No logs found yet.", "logs": []}
    except Exception as e:
        return {"error": str(e)}

@router.get("/monitoring/status")
async def get_system_status():
    """
    Check the status of all required system components.
    """
    components = {
        "model": os.path.exists("models/model.pkl"),
        "scaler": os.path.exists("models/scaler.pkl"),
        "encoders": os.path.exists("models/encoders.pkl"),
        "feature_names": os.path.exists("models/feature_names.pkl"),
        "target_encoder": os.path.exists("models/target_encoder.pkl"),
        "kmeans": os.path.exists("models/kmeans_model.pkl"),
        "cluster_scaler": os.path.exists("models/cluster_scaler.pkl"),
        "logs": os.path.exists("logs/app.log"),
        "evaluation": os.path.exists("evaluation_results.json")
    }
    
    all_healthy = all(components.values())
    
    return {
        "status": "operational" if all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "components": components
    }

@router.get("/monitoring/evaluation")
async def get_evaluation_results():
    """
    Get the latest model evaluation results.
    """
    try:
        if os.path.exists("evaluation_results.json"):
            with open("evaluation_results.json", "r") as f:
                evaluation_data = json.load(f)
            return {
                "status": "success",
                "evaluation": evaluation_data
            }
        return {
            "status": "not_found",
            "message": "No evaluation results found. Please train the model first."
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/monitoring/evaluation/report")
async def get_evaluation_report():
    """
    Get the human-readable evaluation report.
    """
    try:
        if os.path.exists("evaluation_report.txt"):
            with open("evaluation_report.txt", "r") as f:
                report = f.read()
            return {
                "status": "success",
                "report": report
            }
        return {
            "status": "not_found",
            "message": "No evaluation report found. Please train the model first."
        }
    except Exception as e:
        return {"error": str(e)}