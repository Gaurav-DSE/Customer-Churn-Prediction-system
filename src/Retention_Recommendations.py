def get_recommendations(customer):
    recommendations = []

    # Contract
    if customer["Contract"] == "Month-to-month":
        recommendations.append(
            "Offer an incentive to switch to a long-term contract."
        )

    # Early tenure
    if customer["tenure"] <= 3:
        recommendations.append(
            "Provide an early-tenure retention offer or onboarding support."
        )

    # Online Security
    if customer["OnlineSecurity"] == "No":
        recommendations.append(
            "Recommend an Online Security plan."
        )

    # Tech Support
    if customer["TechSupport"] == "No":
        recommendations.append(
            "Offer technical support assistance."
        )

    # High monthly charges
    if customer["MonthlyCharges"] > 70:
        recommendations.append(
            "Review the customer's plan and offer a suitable pricing option."
        )

    # Fiber optic
    if customer["InternetService"] == "Fiber optic":
        recommendations.append(
            "Review the fiber optic plan and ensure the customer is receiving suitable value."
        )

    # Electronic check
    if customer["PaymentMethod"] == "Electronic check":
        recommendations.append(
            "Consider offering a more convenient automatic payment option."
        )

    # No online backup
    if customer["OnlineBackup"] == "No":
        recommendations.append(
            "Consider recommending an Online Backup service."
        )

    # No dependents
    if customer["Dependents"] == "No":
        recommendations.append(
            "Consider personalized offers based on the customer's individual usage."
        )

    # No recommendations
    if not recommendations:
        recommendations.append(
            "Continue regular engagement and monitor the customer's churn risk."
        )

    return recommendations

if __name__ == "__main__":

    test_customer = {
        "Contract": "Month-to-month",
        "tenure": 1,
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "MonthlyCharges": 85,
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "OnlineBackup": "No",
        "Dependents": "No"
    }

    recommendations = get_recommendations(test_customer)

    for recommendation in recommendations:
        print("•", recommendation)