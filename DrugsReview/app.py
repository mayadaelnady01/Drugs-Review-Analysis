from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved SVM classifier and vectorizer
svm_classifier = joblib.load('svm_classifierr.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizerr.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    drug_name = request.form['drug_name']
    
    # Vectorize the drug name using the fitted TF-IDF Vectorizer
    drug_tfidf = tfidf_vectorizer.transform([drug_name])
    
    # Predict sentiment
    pred_sentiment = svm_classifier.predict(drug_tfidf)
    
    # Convert sentiment label to readable format
    sentiment_label = 'Positive' if pred_sentiment[0] == 'Positive' else 'Negative'
    
    return render_template('index.html', sentiment=f'Sentiment for {drug_name}: {sentiment_label}')

if __name__ == '__main__':
    app.run(debug=False)