














from pathlib import Path

import pandas as pd
import streamlit as st
from xgboost import XGBClassifier


MODEL_PATH = Path(__file__).resolve().parent / "student_stress_xgb.json"
FEATURES = [
    "Student_Type",
    "Sleep_Hours",
    "Study_Hours",
    "Social_Media_Hours",
    "Attendance",
    "Exam_Pressure",
    "Family_Support",
    "Month",
]
STUDENT_TYPE_CODES = {"School student": 0, "College student": 1, "Working student": 2}


st.set_page_config(page_title="StressSense | Student Stress Predictor", page_icon="🌿", layout="wide")


@st.cache_resource
def load_model() -> XGBClassifier:
    """Load the trained XGBoost classifier once per app session."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def add_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .stApp { background: radial-gradient(circle at 92% 4%, #dcefe9 0, transparent 26rem), linear-gradient(135deg, #f8fbfa 0%, #edf5f2 100%); }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 2rem; }
        .brand { display:flex; align-items:center; gap:12px; margin-bottom: 1.6rem; }
        .brand-icon { display:grid; place-items:center; width:46px; height:46px; border-radius:15px; background:#16715c; font-size:24px; box-shadow:0 10px 24px rgba(22,113,92,.18); }
        .brand-name { font-family:Manrope, sans-serif; color:#163b34; font-weight:800; font-size:22px; }
        .eyebrow { color:#26806a; font-size:12px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; }
        .hero-title { font-family:Manrope, sans-serif; color:#143a33; font-size:clamp(34px, 4.5vw, 56px); line-height:1.06; letter-spacing:-2px; margin:8px 0 12px; }
        .hero-copy { max-width:680px; color:#627871; font-size:16px; line-height:1.6; margin-bottom:26px; }
        [data-testid="stForm"] { background:rgba(255,255,255,.95); border:1px solid #dce9e4; border-radius:22px; padding:27px; box-shadow:0 18px 44px rgba(23,78,65,.09); }
        .section-label { color:#7b918a; font-size:11px; font-weight:700; letter-spacing:1.35px; text-transform:uppercase; margin:3px 0 10px; }
        [data-testid="stFormSubmitButton"] button { width:100%; min-height:52px; border:0; border-radius:12px; background:#16715c; color:white; font-weight:700; font-size:16px; }
        [data-testid="stFormSubmitButton"] button:hover { background:#0f5948; color:white; border:0; }
        .result-card { min-height:385px; border-radius:22px; padding:31px; color:white; background:linear-gradient(145deg, rgba(255,255,255,.10), transparent 38%), linear-gradient(135deg, #123e36, #1c7460); box-shadow:0 18px 44px rgba(17,69,57,.19); }
        .result-label { color:#b7ded3; font-size:12px; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; }
        .result-value { font-family:Manrope, sans-serif; font-size:48px; line-height:1; margin:16px 0 8px; letter-spacing:-1.5px; }
        .result-copy { color:#e0f2ec; line-height:1.55; font-size:15px; }
        .divider { height:1px; background:rgba(255,255,255,.17); margin:27px 0 20px; }
        .detail { color:#c4e4da; font-size:13px; margin:11px 0; display:flex; justify-content:space-between; gap:16px; }
        .detail strong { color:white; font-weight:600; text-align:right; }
        .empty { min-height:350px; display:grid; place-content:center; text-align:center; color:#d5ebe5; }
        .empty .icon { font-size:48px; margin-bottom:13px; }
        /* Streamlit's dark-theme widget labels can otherwise inherit white text. */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label { color:#244f45 !important; font-weight:600 !important; opacity:1 !important; }
        [data-testid="stWidgetLabel"] { color:#244f45 !important; opacity:1 !important; }
        [data-testid="stSlider"] [data-testid="stThumbValue"] { color:#16715c !important; }
        .stNumberInput input { border-radius:10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_message(probability: float) -> tuple[str, str]:
    if probability >= 0.70:
        return "Higher predicted stress", "The model sees several patterns associated with the positive stress class. Consider checking in with a trusted person or support service."
    if probability >= 0.40:
        return "Moderate predicted stress", "The model sees a mixed pattern. Small changes to rest, workload, and support may be worth exploring."
    return "Lower predicted stress", "The model did not identify a strong positive-stress pattern in these inputs. Keep maintaining supportive routines."


def main() -> None:
    add_styles()
    st.markdown('<div class="brand"><div class="brand-icon">🌿</div><div class="brand-name">StressSense</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">XGBoost-powered wellbeing check</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Understand your study-stress pattern.</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-copy">Enter a few lifestyle and study details to receive an instant prediction from your trained student stress model.</div>', unsafe_allow_html=True)

    form_column, result_column = st.columns([1.5, 1], gap="large")
    with form_column:
        with st.form("stress_prediction_form"):
            st.markdown('<div class="section-label">Student profile</div>', unsafe_allow_html=True)
            profile_col, month_col = st.columns(2)
            with profile_col:
                student_type = st.selectbox("Student type", list(STUDENT_TYPE_CODES))
            with month_col:
                month = st.selectbox("Current month", list(range(1, 13)), index=0, format_func=lambda value: pd.Timestamp(2026, value, 1).strftime("%B"))

            st.markdown('<div class="section-label">Daily routine</div>', unsafe_allow_html=True)
            routine_left, routine_right = st.columns(2)
            with routine_left:
                sleep_hours = st.number_input("Sleep hours per day", min_value=0.0, max_value=24.0, value=7.0, step=0.25)
                social_media_hours = st.number_input("Social media hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.25)
            with routine_right:
                study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=4.0, step=0.25)
                attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)

            st.markdown('<div class="section-label">Support and pressure</div>', unsafe_allow_html=True)
            scale_left, scale_right = st.columns(2)
            with scale_left:
                exam_pressure = st.slider("Exam pressure (1 = low, 10 = high)", 1, 10, 5)
            with scale_right:
                family_support = st.slider("Family support (1 = low, 10 = high)", 1, 10, 6)

            submitted = st.form_submit_button("Check predicted stress level")

    with result_column:
        if submitted:
            try:
                values = [[STUDENT_TYPE_CODES[student_type], sleep_hours, study_hours, social_media_hours, attendance, exam_pressure, family_support, month]]
                inputs = pd.DataFrame(values, columns=FEATURES)
                probability = float(load_model().predict_proba(inputs)[0][1])
                label, advice = risk_message(probability)
                st.markdown(
                    f'''<div class="result-card">
                    <div class="result-label">Prediction</div>
                    <div class="result-value">{label}</div>
                    <div class="result-copy">{advice}</div>
                    <div class="divider"></div>
                    <div class="detail"><span>Positive-stress probability</span><strong>{probability:.0%}</strong></div>
                    <div class="detail"><span>Model confidence signal</span><strong>{max(probability, 1 - probability):.0%}</strong></div>
                    <div class="detail"><span>Student profile</span><strong>{student_type}</strong></div>
                    </div>''',
                    unsafe_allow_html=True,
                )
            except Exception as error:
                st.error(f"The model could not generate a prediction: {error}")
        else:
            st.markdown('<div class="result-card"><div class="empty"><div class="icon">🧭</div><div><strong>Your prediction will appear here.</strong><br><br>Complete the form and select <em>Check predicted stress level</em>.</div></div></div>', unsafe_allow_html=True)

    st.caption("For educational use only. This prediction is not a medical or mental-health diagnosis; seek a qualified professional or local emergency support if you need immediate help.")


if __name__ == "__main__":
    main()
