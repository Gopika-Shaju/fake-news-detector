# Fake News Detector & Summarizer

This project is an AI-powered tool designed to help students identify fake news and summarize articles.

## Project Structure
- `fake_news_detector.ipynb`: detailed Jupyter Notebook with step-by-step implementation.
- `train_model.py`: Python script to train the model and save it for the app.
- `app.py`: A user-friendly web application (using Streamlit) to paste news and get real-time results.
- `requirements.txt`: List of required libraries.

## How to Run

1. **Install Dependencies**
   Open your terminal and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   Before using the app, you need to train the AI model on your datasets (`Fake.csv` and `True.csv`).
   ```bash
   python train_model.py
   ```
   *This will create `model.pkl` and `vectorizer.pkl`.*

3. **Run the App**
   Start the web interface:
   ```bash
   streamlit run app.py
   ```
   This will open the tool in your browser.

## Features
- **Credibility Analysis**: Uses Machine Learning (PassiveAggressiveClassifier) to label text as "REAL" or "FAKE".
- **Summarization**: Uses a Transformer model (DistilBART) to generate concise summaries.
"# fake-news-detector" 
"# fake-news-detector" 
