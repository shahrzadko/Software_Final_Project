from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.utils import pad_sequences
import re
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from tensorflow.keras.utils import custom_object_scope
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model

#preprocess the reviews
# Initialize stopwords, stemmer, and lemmatizer
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Downloading required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the trained model
with custom_object_scope({'F1Score': tfa.metrics.F1Score}):
    model = load_model('model_RNN_LSTM.h5')


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove currency symbols
    text = re.sub(r'£|\$', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove email addresses
    text = re.sub(r'^.+@[^\.].*\.[a-z]{2,}$', '', text)
    #Tokenize text
    tokens = word_tokenize(text)
    #Remove stopwords
    tokens = [word for word in tokens if word not in STOPWORDS]
    #Join tokens back into string
    text = ' '.join(tokens)
    return text

#Define function for predicting sentiment
def predict_sentiment(text):
    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    max_length = 100
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    # Tokenize the preprocessed text
    input_text = tokenizer.texts_to_sequences([preprocessed_text])
    # Pad the input sequence to the same length as the training data
    input_text = pad_sequences(input_text, maxlen=max_length, padding='post')
    # Make the prediction
    predicted_sentiment = model.predict(input_text)
    arr = np.array(predicted_sentiment[0])
    max_index = np.argmax(arr)
    if max_index==2:
        sentiment = 'Positive'
    elif max_index==1:
        sentiment = 'Neutral'
    else:
        sentiment = 'Negative'
    return sentiment


print(predict_sentiment("worst")) 
print(predict_sentiment("best"))

"""Frontend with template"""

# Initialize Flask application
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    while True:
        if request.method == 'POST':
            review = request.form['review']
            prediction = predict_sentiment(review)
            return render_template('index.html', prediction=prediction)
        return render_template('index.html')

if __name__ == '__main__':
     app.run(debug=True)

"""********to use with postman**********"""
"""
app = Flask(__name__)

@app.route('/sentiment-analysis', methods=['POST'])
def sentiment_analysis():
     text = str(request.form.get('text'))
     sentiment = predict_sentiment(text)
     return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
     app.run(debug=True)
    
"""