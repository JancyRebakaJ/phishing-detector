import streamlit as st
import joblib
import pandas as pd
import tldextract

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('model/model.pkl')

model = load_model()

def extract_url_features(url):
    features = {}
    features['url_length'] = len(url)
    features['hostname_length'] = len(tldextract.extract(url).domain)
    features['has_ip'] = 1 if any(char.isdigit() for char in url.split('/')[0]) else 0
    features['count_at'] = url.count('@')
    features['count_question'] = url.count('?')
    features['count_percent'] = url.count('%')
    features['count_dot'] = url.count('.')
    features['count_hyphen'] = url.count('-')
    features['is_https'] = 1 if url.startswith('https') else 0
    return pd.DataFrame([features])

# Page configuration
st.set_page_config(page_title="Phishing URL & Threat Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Phishing URL & Threat Detector")
st.markdown("Enter a suspicious URL below to analyze its structural features and check for potential phishing threats.")

# User input
url_input = st.text_input("Enter URL to scan:", placeholder="https://example.com/login")

if st.button("Analyze URL"):
    if not url_input.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Analyzing lexical features and scanning model..."):
            # Extract features and predict
            features_df = extract_url_features(url_input)
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("Scan Results")
            
            if prediction == 1:
                risk_score = probability[1] * 100
                st.error(f" **WARNING: Potential Phishing Detected!** (Confidence: {risk_score:.1f}%)")
                st.markdown("This URL exhibits structural anomalies commonly associated with malicious sites (e.g., unusual length, IP address usage, or misleading subdomains).")
            else:
                safe_score = probability[0] * 100
                st.success(f" **Safe / Legitimate URL** (Confidence: {safe_score:.1f}%)")
                st.markdown("No major structural indicators of phishing were identified in this URL.")
            
            # Display extracted feature breakdown
            with st.expander("View Extracted Lexical Features"):
                st.json(features_df.to_dict(orient="records")[0])