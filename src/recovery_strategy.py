def generate_recovery_strategy(prediction_result: dict) -> str:
    """
    Generate personalized recovery strategy based on prediction results and risk factors.
    """
    prediction = prediction_result.get('prediction', 0)
    probability = prediction_result.get('probability', 0)
    risk_factors = prediction_result.get('risk_factors', {})
    
    missed_payments = risk_factors.get('Num_Missed_Payments', 0)
    days_past_due = risk_factors.get('Days_Past_Due', 0)
    monthly_income = risk_factors.get('Monthly_Income', 0)
    outstanding_loan = risk_factors.get('Outstanding_Loan_Amount', 0)
    
    # Base strategy on prediction
    if prediction == 2:  # Fully Recovered
        return "Low Priority - Standard Monitoring"
    
    elif prediction == 1:  # Partially Recovered
        if probability > 0.7:
            if missed_payments <= 2 and days_past_due <= 30:
                return "Payment Plan Restructuring + Settlement Offer"
            else:
                return "Enhanced Settlement Offer + Payment Reminders"
        else:
            return "Collection Agency Referral + Legal Review"
    
    else:  # Not Recovered / High Risk
        if probability > 0.8:
            if days_past_due > 90:
                return "Immediate Legal Action + Debt Collection"
            elif missed_payments > 3:
                return "Aggressive Collection + Legal Notice"
            else:
                return "Settlement Negotiation + Payment Plan"
        else:
            return "Write-off Consideration + Legal Proceedings"

def get_strategy_actions(strategy: str) -> list:
    """
    Get detailed actions for each recovery strategy.
    """
    strategy_actions = {
        "Low Priority - Standard Monitoring": [
            "Monthly account review",
            "Standard payment reminders",
            "Annual financial health check"
        ],
        "Payment Plan Restructuring + Settlement Offer": [
            "Contact borrower for settlement discussion",
            "Restructure payment plan",
            "Offer 10-20% settlement discount",
            "Set up automated payment schedule"
        ],
        "Enhanced Settlement Offer + Payment Reminders": [
            "Send formal settlement offer",
            "Increase collection call frequency",
            "Offer 20-30% settlement discount",
            "Set payment deadline"
        ],
        "Collection Agency Referral + Legal Review": [
            "Refer to collection agency",
            "Legal department review",
            "Prepare demand notice",
            "Monitor for 30 days"
        ],
        "Immediate Legal Action + Debt Collection": [
            "Issue legal notice immediately",
            "Engage debt collection agency",
            "Freeze additional credit",
            "Weekly progress monitoring"
        ],
        "Aggressive Collection + Legal Notice": [
            "Send legal demand notice",
            "Daily collection calls",
            "Credit bureau reporting",
            "Collateral assessment"
        ],
        "Settlement Negotiation + Payment Plan": [
            "Urgent settlement negotiation",
            "Payment plan with guarantees",
            "Collateral verification",
            "Bi-weekly follow-ups"
        ],
        "Write-off Consideration + Legal Proceedings": [
            "Evaluate for write-off",
            "Initiate legal proceedings",
            "Asset investigation",
            "Final settlement attempt"
        ]
    }
    
    return strategy_actions.get(strategy, ["Contact financial advisor for custom strategy"])