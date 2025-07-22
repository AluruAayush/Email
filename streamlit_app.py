import streamlit as st
import joblib

# Load the wrapped EmailPredictor class
@st.cache_resource
def load_predictor():
    return joblib.load("EmailPredictor.joblib")

predictor = load_predictor()

# Streamlit UI
st.set_page_config(page_title="Email Performance Predictor", page_icon="📧")

st.title("📧 Email Performance Predictor")
st.markdown("Enter your email subject and body to get predictions for open rate and click-through rate (CTR).")

# Input fields
subject = st.text_input("✉️ Email Subject")
body = st.text_area("📝 Email Body", height=200)

# Predict button
if st.button("📊 Predict"):
    if not subject or not body:
        st.warning("Please fill in both the subject and the body.")
    else:
        try:
            open_rate, ctr = predictor.predict(subject, body)
            st.success(f"✅ **Predicted Open Rate:** {open_rate * 100:.2f}%")
            st.success(f"✅ **Predicted CTR:** {ctr * 100:.2f}%")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
