from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Sample Dataset
data = {
    'url': [
        'https://www.google.com', 
        'http://malicious.com/badsite', 
        'https://facebook.com/login', 
        'http://phishing.com/login.php',
        'https://github.com', 
        'http://badwebsite.com/malware',
        'https://linkedin.com/in/login', 
        'http://fakebank.com/login',
        'https://twitter.com', 
        'http://harmfulsite.org/download'
    ],
    'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = safe, 1 = unsafe
}

# Prepare the Dataset
df = pd.DataFrame(data)
vectorizer = TfidfVectorizer(token_pattern=r'[A-Za-z0-9]+', max_features=3000)
X = vectorizer.fit_transform(df['url'])
y = df['label']

# Train the Model
model = LogisticRegression()
model.fit(X, y)

# Route for rendering the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML file

# Route for URL classification
@app.route('/predict', methods=['POST'])
def predict():
    # Get the URL from the POST request
    input_data = request.json
    url = input_data.get('url', '')

    # Vectorize and Predict
    url_features = vectorizer.transform([url])
    prediction = model.predict(url_features)[0]

    # Return the result as JSON
    return jsonify({'url': url, 'prediction': 'unsafe' if prediction == 1 else 'safe'})

if __name__ == '__main__':
    app.run(debug=True)
