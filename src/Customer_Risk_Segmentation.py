def risk_level(probability):
    if probability < 0.20:
        return '🟢 Low Risk'
    elif probability < 0.35:
        return "🟡 Medium Risk"
    elif probability < 0.60:
        return "🟠 High Risk(Action Needed)"
    else:
        return "🔴 Very High Risk(Critical)"

if __name__ == "__main__":
    print(risk_level(0.15))
    print(risk_level(0.30))
    print(risk_level(0.50))
    print(risk_level(0.80))