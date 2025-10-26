import logging
import os
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_event(event_type, details):
    msg = f"[{event_type}] {datetime.now()} | {details}"
    logging.info(msg)
    print(msg)

def log_error(error_msg, details=None):
    msg = f"[ERROR] {error_msg} | {details if details else 'N/A'}"
    logging.error(msg)
    print(msg)

def log_prediction(input_data, prediction, probability):
    log_event("PREDICTION", {
        "input": input_data,
        "prediction": prediction,
        "confidence": probability
    })

def log_training_metrics(metrics):
    log_event("TRAINING_COMPLETE", {
        "accuracy": metrics.get('accuracy', 0),
        "precision": metrics.get('precision', 0),
        "recall": metrics.get('recall', 0),
        "f1_score": metrics.get('f1_score', 0)
    })