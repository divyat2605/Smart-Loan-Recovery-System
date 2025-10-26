import logging
import os
from typing import Tuple
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_PATH = "models/model.pkl"
N_ESTIMATORS = 100
RANDOM_STATE = 42

def train_recovery_model(X_train, y_train, X_test, y_test) -> Tuple[RandomForestClassifier, float]:
    """
    Trains a RandomForestClassifier model, evaluates it, and saves it.
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        joblib.dump(model, MODEL_PATH)
        logging.info(f"Model trained and saved to {MODEL_PATH} with accuracy: {acc:.3f}")
        
        return model, acc
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def load_model() -> RandomForestClassifier:
    """
    Loads the trained model from the specified path.
    """
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    
    logging.info(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def tune_hyperparameters(X_train, y_train):
    """
    Performs hyperparameter tuning for the RandomForestClassifier model.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    
    # Save the best model
    joblib.dump(grid_search.best_estimator_, MODEL_PATH)
    logging.info(f"Best model saved to {MODEL_PATH}")
    
    return grid_search.best_estimator_