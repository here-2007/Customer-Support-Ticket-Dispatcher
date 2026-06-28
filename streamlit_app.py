import streamlit as st
from model import prediction as pred

st.set_page_config(
    page_title="Customer Support Ticket Dispatcher",
    page_icon="📩",
    layout="centered",
)

st.title("Customer Support Ticket Dispatcher")
st.caption("Predicts the most likely team and urgency score for a support email.")


@st.cache_resource
def load_model_once():
    pred.get_model()
    return True


def safe_predict(email_text: str):
    if not email_text.strip():
        return None

    return pred.predict_output([email_text.strip()])


with st.spinner("Loading model..."):
    try:
        load_model_once()
    except Exception as e:
        st.error("Model failed to load.")
        st.exception(e)
        st.stop()

st.success("Model loaded successfully.")
def clear_form():
    st.session_state.email = ""

email_text = st.text_area(
    "Paste the customer email here",
    height=220,
    placeholder="Write or paste the email message...",
    key="email",
)

col1, col2 = st.columns([1, 1])

with col1:
    predict_clicked = st.button("Predict", type="primary")

with col2:
    clear_clicked = st.button("Clear", on_click=clear_form)

if predict_clicked:
    try:
        result = safe_predict(email_text)

        if result is None:
            st.warning("Please enter an email first.")
        else:
            st.subheader("Prediction")

            st.metric("Predicted Team", result["predicted_team"])
            st.metric("Confidence", f'{result["confidence"]:.3f}')
            st.metric("Urgency Score", f'{result["urgency_score"]:.3f}')

            urgency = result["urgency_score"]

            if urgency < 0.4:
                st.info("Low urgency")
            elif urgency < 0.7:
                st.warning("Medium urgency")
            else:
                st.error("High urgency")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

st.divider()
st.caption(f"Model version: {pred.MODEL_VERSION}")