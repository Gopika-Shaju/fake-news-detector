import streamlit as st
import joblib
import re
import string
from transformers import pipeline

# Page Config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #ff4b4b;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .fake {
        background-color: #ff4b4b;
    }
    .real {
        background-color: #28a745;
    }
    .summary-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.title("📰 AI Fake News Detector")
st.markdown("### For Students & Researchers")
st.write("Paste a news article below to assess its credibility and get a concise summary.")

# Load Model/Assets
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except:
        return None, None

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

model, vectorizer = load_assets()
summarizer = load_summarizer()

# Cleaning Function (same as training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('\\\\[.*?\\]', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text

# Input Area
article_text = st.text_area("Enter News Article Content", height=300)

if st.button("Analyze Article"):
    if not article_text:
        st.warning("Please enter some text to analyze.")
    elif model is None:
        st.error("Model files not found. Please run 'train_model.py' first!")
    else:
        with st.spinner("Analyzing credibility and summarizing..."):
            # Detection
            cleaned_text = clean_text(article_text)
            vec_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vec_text)[0]
            probability = 0 # PAC doesn't support predict_proba by default, simpler classification
            
            # Summarization
            try:
                # Truncate for speed/memory if too long
                input_len = len(article_text)
                if input_len > 3000:
                    sum_text = article_text[:3000]
                else:
                    sum_text = article_text
                
                summary_result = summarizer(sum_text, max_length=130, min_length=30, do_sample=False)
                summary_text = summary_result[0]['summary_text']
            except Exception as e:
                summary_text = f"Error generating summary: {e}"

            # Display Results
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Credibility Assessment")
                if prediction == 0:
                    st.markdown('<div class="result-box fake">⚠️ FAKE NEWS DETECTED</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-box real">✅ RELIABLE NEWS DETECTED</div>', unsafe_allow_html=True)
                
            with col2:
                st.subheader("Result Details")
                st.info("Analysis complete based on linguistic patterns and verified datasets.")

            st.markdown("### 📝 Article Summary")
            st.markdown(f'<div class="summary-box">{summary_text}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Powered by Scikit-Learn & HuggingFace Transformers")
