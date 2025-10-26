import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib
import json
import os

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Comprehensive model evaluation with multiple metrics and visualization.
    """
    # Load target encoder to get class names
    target_encoder = None
    if os.path.exists("models/target_encoder.pkl"):
        target_encoder = joblib.load("models/target_encoder.pkl")
        class_names = target_encoder.classes_.tolist()
    else:
        class_names = None
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # ROC AUC (for binary/multiclass classification)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        roc_auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    if class_names:
        class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    else:
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Feature importance
    if hasattr(model, 'feature_importances_') and feature_names:
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    else:
        sorted_features = []
    
    # Compile results
    evaluation_results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'top_features': sorted_features,
        'class_names': class_names
    }
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Generate report
    generate_evaluation_report(evaluation_results, model, X_test.shape[0])
    
    return evaluation_results

def generate_evaluation_report(results, model, test_size):
    """
    Generate a human-readable evaluation report.
    """
    report = f"""
SMART LOAN RECOVERY SYSTEM - MODEL EVALUATION REPORT
{'=' * 60}

MODEL INFORMATION:
- Model Type: {type(model).__name__}
- Test Set Size: {test_size} samples
- Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE METRICS:
- Accuracy: {results['accuracy']:.4f}
- Precision: {results['precision']:.4f}
- Recall: {results['recall']:.4f}
- F1-Score: {results['f1_score']:.4f}
- ROC AUC: {results['roc_auc'] if results['roc_auc'] else 'N/A'}

CONFUSION MATRIX:
{np.array(results['confusion_matrix'])}

"""
    
    if results.get('class_names'):
        report += f"CLASS LABELS: {', '.join(results['class_names'])}\n\n"
    
    report += "TOP 10 FEATURE IMPORTANCES:\n"
    
    for feature, importance in results['top_features']:
        report += f"- {feature}: {importance:.4f}\n"
    
    report += f"\nCLASSIFICATION REPORT:\n"
    for class_name, metrics in results['classification_report'].items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            if isinstance(metrics, dict):
                report += f"Class {class_name}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}\n"
    
    # Save report
    with open('evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print("✅ Evaluation report generated: evaluation_report.txt")
    return report