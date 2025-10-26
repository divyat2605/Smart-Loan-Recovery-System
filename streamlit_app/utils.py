import requests
import streamlit as st

def check_api_health():
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_status():
    """Get current system status from API."""
    try:
        response = requests.get("http://localhost:8000/api/v1/monitoring/status")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def format_currency(amount):
    """Format number as currency."""
    return f"${amount:,.2f}"

def get_risk_color(risk_level):
    """Get color code for risk levels."""
    colors = {
        "High": "#ff6b6b",
        "Medium": "#ffa500", 
        "Low": "#51cf66"
    }
    return colors.get(risk_level, "#666666")