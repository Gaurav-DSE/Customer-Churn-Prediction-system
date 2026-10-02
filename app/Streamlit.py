import joblib
import pandas as pd
import altair as alt
import shap
import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.Customer_Risk_Segmentation import risk_level
from src.Retention_Recommendations import get_recommendations

# ============================================================
# CONFIG
# ============================================================

THRESHOLD = 0.35

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# LOAD ARTIFACTS (cached so they load once, not on every click)
# ============================================================


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT/"artifacts"/"xgb_model.pkl")
    preprocessor = joblib.load(ROOT/"artifacts"/"preprocessor.pkl")
    feature_names = joblib.load(ROOT/"artifacts"/"feature_names.pkl")
    explainer = shap.TreeExplainer(model)
    return model, preprocessor, feature_names, explainer


model, preprocessor, feature_names, explainer = load_artifacts()

# ============================================================
# STYLE (works in light and dark themes)
# ============================================================

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .hero {
        padding: 30px 34px; border-radius: 18px; margin-bottom: 18px;
        background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 60%, #0e7490 100%);
        color: #fff;
    }
    .hero h1 {margin: 0 0 6px 0; font-size: 38px; color: #fff;}
    .hero p {margin: 0; font-size: 17px; color: #cbd5e1;}
    .pill {
        display: inline-block; padding: 4px 12px; margin: 14px 8px 0 0;
        border-radius: 999px; font-size: 13px;
        background: rgba(255,255,255,.14); color: #e2e8f0;
    }
    .card {
        padding: 18px 20px; border-radius: 14px;
        border: 1px solid rgba(128,128,128,.28);
        background: rgba(128,128,128,.07); height: 100%;
    }
    .card .label {font-size: 13px; opacity: .7; margin-bottom: 4px;}
    .card .value {font-size: 28px; font-weight: 700;}
    .verdict {
        padding: 26px 28px; border-radius: 16px; margin-bottom: 16px; color: #fff;
    }
    .verdict.stay {background: linear-gradient(120deg,#065f46,#047857);}
    .verdict.churn {background: linear-gradient(120deg,#991b1b,#dc2626);}
    .verdict .small {font-size: 14px; opacity: .85;}
    .verdict .big {font-size: 32px; font-weight: 700; margin: 4px 0;}
    .gauge {
        position: relative; height: 16px; border-radius: 999px; margin: 34px 0 6px 0;
        background: linear-gradient(90deg,#22c55e 0%,#22c55e 20%,#eab308 20%,
                    #eab308 35%,#f97316 35%,#f97316 60%,#dc2626 60%,#dc2626 100%);
    }
    .gauge .marker {
        position: absolute; top: -9px; width: 4px; height: 34px;
        background: currentColor; border-radius: 2px;
    }
    .gauge .marker span {
        position: absolute; top: -24px; left: -26px; width: 56px;
        text-align: center; font-size: 13px; font-weight: 700;
    }
    .gauge .thr {
        position: absolute; top: -4px; width: 2px; height: 24px; background: #fff;
        left: 35%; opacity: .9;
    }
    .gauge-legend {display:flex; justify-content:space-between; font-size:12px; opacity:.7;}
    .rec {
        padding: 12px 16px; border-radius: 10px; margin-bottom: 8px;
        border-left: 4px solid #3b82f6; background: rgba(59,130,246,.10);
    }
    .step {
        padding: 12px 10px; border-radius: 10px; text-align: center; font-size: 14px;
        border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.07);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SAMPLE CUSTOMERS (one-click demo presets)
# ============================================================

PRESETS = {
    "High-risk example": dict(
        gender="Male", senior_citizen=0, partner="No", dependents="No", tenure=2,
        phone_service="Yes", multiple_lines="No", internet_service="Fiber optic",
        online_security="No", online_backup="No", device_protection="No",
        tech_support="No", streaming_tv="No", streaming_movies="No",
        contract="Month-to-month", paperless_billing="Yes",
        payment_method="Electronic check", monthly_charges=85.0, total_charges=170.0,
    ),
    "Loyal customer example": dict(
        gender="Female", senior_citizen=0, partner="Yes", dependents="Yes", tenure=60,
        phone_service="Yes", multiple_lines="Yes", internet_service="DSL",
        online_security="Yes", online_backup="Yes", device_protection="Yes",
        tech_support="Yes", streaming_tv="No", streaming_movies="No",
        contract="Two year", paperless_billing="No",
        payment_method="Credit card (automatic)", monthly_charges=65.0, total_charges=3900.0,
    ),
}


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>📊 Customer Churn Prediction System</h1>
        <p>Predict who is likely to leave, understand why, and decide what to do about it.</p>
        <span class="pill">Tuned XGBoost</span>
        <span class="pill">SHAP explanations</span>
        <span class="pill">Risk segmentation</span>
        <span class="pill">Retention actions</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🤖 Model information")
    st.write("**Model:** Tuned XGBoost")
    st.write(f"**Decision threshold:** {THRESHOLD}")
    st.write("**Explainability:** SHAP (TreeExplainer)")
    st.write("**Data:** Telco Customer Churn, 7,010 customers after cleaning")
    st.divider()
    st.subheader("Try a sample customer")
    for name in PRESETS:
        st.button(name, on_click=apply_preset, args=(name,), use_container_width=True)
    st.divider()
    st.caption(
        "SHAP explains the model's behaviour. It does not prove that a feature causes churn."
    )

tab_predict, tab_insights, tab_model, tab_about = st.tabs(
    ["🔍 Predict churn", "📈 Data insights", "🏆 Model performance", "🧭 Project overview"]
)

# ============================================================
# TAB 1: PREDICT
# ============================================================

with tab_predict:
    st.subheader("👤 Customer information")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
        senior_citizen = st.selectbox("Senior citizen", [0, 1], key="senior_citizen")
        partner = st.selectbox("Partner", ["Yes", "No"], key="partner")
        dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents")
        tenure = st.number_input("Tenure (months)", 0, 100, 12, key="tenure")

    with c2:
        phone_service = st.selectbox("Phone service", ["Yes", "No"], key="phone_service")
        multiple_lines = st.selectbox(
            "Multiple lines", ["Yes", "No", "No phone service"], key="multiple_lines")
        internet_service = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"], key="internet_service")
        online_security = st.selectbox(
            "Online security", ["Yes", "No", "No internet service"], key="online_security")
        online_backup = st.selectbox(
            "Online backup", ["Yes", "No", "No internet service"], key="online_backup")

    with c3:
        device_protection = st.selectbox(
            "Device protection", ["Yes", "No", "No internet service"], key="device_protection")
        tech_support = st.selectbox(
            "Tech support", ["Yes", "No", "No internet service"], key="tech_support")
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"], key="streaming_tv")
        streaming_movies = st.selectbox(
            "Streaming movies", ["Yes", "No", "No internet service"], key="streaming_movies")

    st.subheader("💳 Billing information")
    b1, b2, b3 = st.columns(3)
    with b1:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"], key="contract")
    with b2:
        paperless_billing = st.selectbox(
            "Paperless billing", ["Yes", "No"], key="paperless_billing")
    with b3:
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            key="payment_method",
        )

    b4, b5 = st.columns(2)
    with b4:
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, value=70.0, step=0.1,
            key="monthly_charges")
    with b5:
        if "total_charges" not in st.session_state:
            st.session_state["total_charges"] = float(monthly_charges * tenure)
        total_charges = st.number_input(
            "Total charges ($)", min_value=0.0, step=0.1, key="total_charges")

    analyze = st.button("🔍 Analyze customer", type="primary", use_container_width=True)

    if analyze:
        customer = {
            "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_service,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        customer_df = pd.DataFrame([customer])
        processed = preprocessor.transform(customer_df)
        prob = float(model.predict_proba(processed)[0][1])
        prediction = int(prob >= THRESHOLD)
        risk = risk_level(prob)

        st.divider()
        st.header("🎯 Prediction result")

        # ---------- verdict + gauge ----------
        css = "churn" if prediction else "stay"
        text = "⚠️ Likely to churn" if prediction else "✅ Likely to stay"
        st.markdown(
            f"""
            <div class="verdict {css}">
                <div class="small">MODEL PREDICTION</div>
                <div class="big">{text}</div>
                <div class="small">Churn probability {prob:.1%}
                 · decision threshold {THRESHOLD:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pos = min(max(prob * 100, 1), 99)
        st.markdown(
            f"""
            <div class="gauge">
                <div class="thr"></div>
                <div class="marker" style="left:{pos}%;"><span>{prob:.0%}</span></div>
            </div>
            <div class="gauge-legend">
                <span>0%</span><span>Low &lt;20%</span><span>Medium 20–35%</span>
                <span>High 35–60%</span><span>Very high ≥60%</span><span>100%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        # ---------- KPI cards ----------
        k1, k2, k3, k4 = st.columns(4)
        for col, label, value in [
            (k1, "Churn probability", f"{prob:.1%}"),
            (k2, "Risk level", risk),
            (k3, "Margin vs threshold", f"{(prob - THRESHOLD) * 100:+.1f} pts"),
            (k4, "Monthly revenue at stake", f"${monthly_charges:,.2f}"),
        ]:
            col.markdown(
                f'<div class="card"><div class="label">{label}</div>'
                f'<div class="value" style="font-size:22px">{value}</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        if prob < 0.20:
            st.success("Low predicted churn probability. Regular engagement and monitoring are appropriate.")
        elif prob < 0.35:
            st.warning("Moderate churn probability. Proactive engagement could reduce future risk.")
        elif prob < 0.60:
            st.warning("High risk. Retention action should be considered.")
        else:
            st.error("Very high predicted churn probability. Immediate retention attention is recommended.")

        # ---------- SHAP ----------
        st.header("🧠 Why did the model predict this?")
        shap_values = explainer(processed)
        values = shap_values.values[0]
        if values.ndim > 1:
            values = values[:, 1]

        shap_df = pd.DataFrame({"Feature": feature_names, "SHAP value": values})
        shap_df["abs"] = shap_df["SHAP value"].abs()
        top = shap_df.sort_values("abs", ascending=False).head(8).copy()
        top["Effect"] = top["SHAP value"].apply(
            lambda x: "Increases churn risk" if x > 0 else "Decreases churn risk")

        left, right = st.columns([3, 2])
        with left:
            chart = (
                alt.Chart(top)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("SHAP value:Q", title="Impact on churn risk"),
                    y=alt.Y("Feature:N", sort=alt.EncodingSortField("abs", order="descending"), title=None),
                    color=alt.Color(
                        "Effect:N",
                        scale=alt.Scale(
                            domain=["Increases churn risk", "Decreases churn risk"],
                            range=["#dc2626", "#16a34a"]),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    tooltip=["Feature", alt.Tooltip("SHAP value:Q", format=".3f"), "Effect"],
                )
                .properties(height=330)
            )
            st.altair_chart(chart, use_container_width=True)
        with right:
            table = top[["Feature", "SHAP value", "Effect"]].copy()
            table["SHAP value"] = table["SHAP value"].round(3)
            st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption("Positive values push the prediction toward churn; negative values push toward staying.")

        # ---------- Profile ----------
        st.header("👤 Customer profile")
        p1, p2, p3 = st.columns(3)
        p1.markdown(
            f"**Contract:** {contract}  \n**Tenure:** {tenure} months  \n"
            f"**Internet:** {internet_service}  \n**Monthly charges:** ${monthly_charges:,.2f}")
        p2.markdown(
            f"**Online security:** {online_security}  \n**Tech support:** {tech_support}  \n"
            f"**Online backup:** {online_backup}  \n**Payment:** {payment_method}")
        p3.markdown(
            f"**Senior citizen:** {'Yes' if senior_citizen else 'No'}  \n**Partner:** {partner}  \n"
            f"**Dependents:** {dependents}  \n**Paperless billing:** {paperless_billing}")

        # ---------- Recommendations ----------
        st.header("💡 Recommended retention actions")
        recs = get_recommendations(customer)
        if recs:
            for i, r in enumerate(recs, 1):
                st.markdown(f'<div class="rec"><strong>{i}.</strong> {r}</div>', unsafe_allow_html=True)
        else:
            st.info("No specific retention actions triggered for this customer.")

        # ---------- Download ----------
        report = pd.DataFrame([{
            **customer,
            "ChurnProbability": round(prob, 4),
            "Prediction": "Churn" if prediction else "Stay",
            "RiskLevel": risk,
            "TopDrivers": "; ".join(top["Feature"].head(3)),
            "Recommendations": " | ".join(recs),
        }])
        st.download_button(
            "⬇️ Download customer report (CSV)",
            report.to_csv(index=False).encode("utf-8"),
            file_name="churn_report.csv",
            mime="text/csv",
        )
        st.caption(
            "Recommendations are rule-based suggestions from customer attributes, "
            "not guaranteed outcomes.")

# ============================================================
# TAB 2: DATA INSIGHTS (numbers from your EDA)
# ============================================================


def churn_chart(df, cat, title):
    return (
        alt.Chart(df, title=title)
        .mark_bar(cornerRadiusEnd=4, color="#3b82f6")
        .encode(
            x=alt.X(f"{cat}:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Churn %:Q", scale=alt.Scale(domain=[0, 50])),
            tooltip=[cat, alt.Tooltip("Churn %:Q", format=".2f")],
        )
        .properties(height=260)
    )


with tab_insights:
    st.subheader("What the data says")
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in [
        (m1, "Customers (cleaned)", "7,010"),
        (m2, "Original churners", "1,869 of 7,043"),
        (m3, "Avg tenure: stayed vs churned", "37.7 vs 18.1 mo"),
        (m4, "Avg monthly: stayed vs churned", "$61.39 vs $74.60"),
    ]:
        col.markdown(
            f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value" style="font-size:22px">{val}</div></div>',
            unsafe_allow_html=True)
    st.write("")

    r1, r2 = st.columns(2)
    with r1:
        st.altair_chart(churn_chart(pd.DataFrame({
            "Contract": ["Month-to-month", "One year", "Two year"],
            "Churn %": [42.64, 11.28, 2.85]}), "Contract", "Churn rate by contract"),
            use_container_width=True)
    with r2:
        st.altair_chart(churn_chart(pd.DataFrame({
            "Internet": ["Fiber optic", "DSL", "No internet"],
            "Churn %": [41.78, 18.93, 7.24]}), "Internet", "Churn rate by internet service"),
            use_container_width=True)

    r3, r4 = st.columns(2)
    with r3:
        st.altair_chart(churn_chart(pd.DataFrame({
            "Group": ["Electronic check", "No tech support", "No online security"],
            "Churn %": [45.15, 41.51, 41.65]}), "Group", "Higher-churn customer groups"),
            use_container_width=True)
    with r4:
        st.altair_chart(churn_chart(pd.DataFrame({
            "Age group": ["Senior", "Non-senior"],
            "Churn %": [41.63, 23.55]}), "Age group", "Churn rate by senior citizen status"),
            use_container_width=True)

    st.info(
        "Month-to-month contracts, fiber optic, electronic check and short tenure are linked "
        "with higher churn in this dataset. These are patterns, not proof of cause.")

# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================

with tab_model:
    st.subheader("Final model: tuned XGBoost at a 0.35 threshold")
    perf = pd.DataFrame({
        "Metric": ["Recall", "Precision", "Accuracy", "F1 score", "ROC-AUC"],
        "Tuned XGBoost": [0.7062, 0.5822, 0.7882, 0.6382, 0.8450],
        "Logistic Regression": [0.6927, 0.5636, 0.7767, 0.6215, 0.8472],
    })

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recall", "70.6%", "+1.4 pts vs LR")
    k2.metric("Precision", "58.2%", "+1.9 pts vs LR")
    k3.metric("F1 score", "63.8%", "+1.7 pts vs LR")
    k4.metric("ROC-AUC", "0.845", "-0.002 vs LR", delta_color="off")

    long = perf.melt("Metric", var_name="Model", value_name="Score")
    st.altair_chart(
        alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset="Model:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Model:N", scale=alt.Scale(range=["#2563eb", "#94a3b8"]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Metric", "Model", alt.Tooltip("Score:Q", format=".4f")],
        ).properties(height=320),
        use_container_width=True)

    st.markdown(
        """
        **Why 0.35 instead of 0.50?** The dataset has an imbalanced churn distribution,
          with churners representing about 27% of customers. A 0.50 cutoff produced lower recall.
          We selected 0.35 to improve the model's ability to identify potential churners while maintaining 
          a reasonable precision level for retention actions.

        **Why XGBoost?** I compared Logistic Regression with XGBoost. Their ROC-AUC scores were very close,
          but at my selected threshold of 0.35, XGBoost gave slightly better recall, precision, accuracy and 
          F1 score. So I used XGBoost as my final model.

        **Threshold trade-off:** Lowering the threshold increases the number of customers flagged as potential churners.
          This improves recall but can also increase false positives. The 0.35 threshold was selected based on the
            observed precision–recall trade-off
        """
    )

    st.markdown("""**Risk segments-->**    
                **NOTE:-** Risk levels are used to divide customers into different risk groups based on their predicted churn probability.
    
    
                """)
    st.dataframe(pd.DataFrame({
        "Probability": ["< 20%", "20–35%", "35–60%", "≥ 60%"],
        "Risk level": ["🟢 Low", "🟡 Medium", "🟠 High (action needed)", "🔴 Very high (critical)"],
        "Basis": ["Business threshold", "Business threshold", "Starts at model threshold",
                  "Business threshold"],
    }), hide_index=True, use_container_width=True)

# ============================================================
# TAB 4: PROJECT OVERVIEW
# ============================================================

with tab_about:
    st.subheader("The business problem")
    st.write(
        "Telecom companies lose customers when they leave their service."
        "This project predicts the probability that a customer may churn "
        "based on their contract, tenure, charges, services, payment method,"
        "and other customer information."
    )

    st.subheader("End-to-end pipeline")
    steps = ["Data cleaning", "EDA", "Preprocessing", "Model Training",
             "Tuning + threshold", "XGBoost", "SHAP", "Risk Segmentation",
            "Retention Recommendations","Streamlit"]
    cols = st.columns(len(steps))
    for c, s in zip(cols, steps):
        c.markdown(f'<div class="step">{s}</div>', unsafe_allow_html=True)

    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
            **How each part is used**
            - **XGBoost** → predicts the probability of churn
            - **SHAP** → explains why the model made the prediction
            - **Risk rules** → divide customers into risk levels
            - **Recommendation rules** → suggest possible retention actions
            - **Streamlit** → provides the interactive interface            
            """
        )
    with b:
        st.markdown(
            """
            **Data preparation**
            - Original data: 7,043 rows, 21 columns
            - Removed `customerID` and 22 duplicate rows
            - Cleaned blank `TotalCharges` values
            - Final data: 7,010 rows, 20 columns
            """
        )

    st.markdown(
        """**tech Stack**
                **Python · NumPy · pandas · Matplotlib · Seaborn · scikit-learn · XGBoost · SHAP · Streamlit
                """
    )

st.divider()
st.caption("Customer Churn Prediction · Tuned XGBoost · SHAP explainability")