import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from keras.layers import Dense, Input, Flatten, LSTM, Bidirectional,Embedding, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential, load_model
import tensorflow as tf
import tensorflow_addons as tfa
from nltk.corpus import wordnet
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import pickle
from tensorflow.keras.models import save_model

# Downloading required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

"""Loading the dataset and preparing for sentiment analysis"""

df_beauty = pd.read_json("reviews_Beauty_5.json", lines=True)

# Dropping the unwanted columns and rows with no reviewer name
review_df = df_beauty.drop(columns=[ 'unixReviewTime', 'reviewTime'], axis=1)
review_df = review_df.dropna()

# Randomly sampling the reviews from the dataset
df_sample = review_df.sample(np.random.randint(10000, 15000))

# Labelling the data as positive, negative or neutral based on the value of "rating of the product"
df_sample['Rating'] = ""
df_sample['Rating'] = np.where(df_sample.overall >= 4, 'Positive', np.where(df_sample.overall == 3, 'Neutral', np.where(df_sample.overall < 3, 'Negative', '')))
review_df = df_sample.drop(columns =['helpful','reviewerID', 'reviewerName','summary'])


"""  Balancing the data """
# Split the dataset by sentiment
df_pos = review_df[review_df['Rating'] == 'Positive']
df_neutral = review_df[review_df['Rating'] == 'Neutral']
df_neg = review_df[review_df['Rating'] == 'Negative']

# Define function for synonym replacement
def synonym_replacement(text):
    words = nltk.word_tokenize(text)
    new_words = []
    for word in words:
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if synonyms:
            new_word = synonyms[0]
        else:
            new_word = word
        new_words.append(new_word)
    new_text = ' '.join(new_words)
    return new_text

# Oversample minority classes using synonym replacement
desired_balance = 0.9
while len(df_neutral) / len(df_pos) < desired_balance or len(df_neg) / len(df_pos) < desired_balance:
    if len(df_neutral) / len(df_pos) < desired_balance:
        sample = df_neutral.sample(n=1)['reviewText'].iloc[0]
        new_review = synonym_replacement(sample)
        df_neutral = df_neutral.append({'reviewText': new_review, 'Rating': 'Neutral'}, ignore_index=True)
    if len(df_neg) / len(df_pos) < desired_balance:
        sample = df_neg.sample(n=1)['reviewText'].iloc[0]
        new_review = synonym_replacement(sample)
        df_neg = df_neg.append({'reviewText': new_review, 'Rating': 'Negative'}, ignore_index=True)

# Combine oversampled subsets back into a single dataframe
review_df = pd.concat([df_pos, df_neutral, df_neg], ignore_index=True)

# Shuffle the dataframe
review_df = review_df.sample(frac=1).reset_index(drop=True)

# Initialize stopwords, stemmer, and lemmatizer
STOPWORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

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

review_df['reviewText'] = review_df['reviewText'].apply(preprocess_text)


#Remove non-English reviews
#!pip install langdetect

from langdetect import detect
review_df['reviewText'] = review_df['reviewText'].apply(lambda x: '' if pd.isna(x) or not isinstance(x, str) else x)
review_df['language'] = review_df['reviewText'].apply(lambda x: detect(x) if x else '')
review_df = review_df[review_df['language'] == 'en']


#Select feature and label
X = review_df.reviewText
y = review_df.Rating

#Text Representation

#Tokenizing and Padding


max_words = 5000
max_len = 100


def tokenize_pad_sequences(text):
    '''
    This function tokenize the input text into sequnences of intergers and then
    pad each sequence to the same length
    '''
    # Text tokenization
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(text)
    # Transforms text to a sequence of integers
    X = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    X = pad_sequences(X, padding='post', maxlen=max_len)
    # return sequences
    return X, tokenizer

X, tokenizer = tokenize_pad_sequences(X)

#One-hot encoding the label
y = pd.get_dummies(review_df.Rating)

#Split Dataset

#Split dataset into Training, Validation, Testing data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42, stratify=y_train_val)

print('Train:         ', X_train.shape, y_train.shape)
print('Validation:', X_val.shape, y_val.shape)
print('Test:      ', X_test.shape, y_test.shape)


#!pip install tensorflow_addons


#Model RNN+LSTM
vocab_size = 5000
embedding_size = 32
epochs=10
batch_size = 32

#Create Model
model_lstm= Sequential()
model_lstm.add(Embedding(vocab_size, embedding_size, input_length=max_len))
model_lstm.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_lstm.add(MaxPooling1D(pool_size=2))
model_lstm.add(Bidirectional(LSTM(32)))
model_lstm.add(Dropout(0.1))
model_lstm.add(Dense(3, activation='softmax'))

#Model Summary
model_lstm.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tfa.metrics.F1Score(num_classes=3, average='macro'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])
print(model_lstm.summary())

#Model Plot
plot_model(model_lstm, show_shapes = True)

#Training
history = model_lstm.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=batch_size, epochs=epochs, verbose=1)

y_pred = model_lstm.predict(X_test)

# Checking validation metrics
# Plot accuracy and loss of training data vs validation data
history_frame = pd.DataFrame(history.history)
#history_frame.loc[:, ['sparse_categorical_accuracy','val_sparse_categorical_accuracy']].plot();
history_frame.loc[:, ['accuracy','val_accuracy']].plot();
history_frame.loc[:, ['loss','val_loss']].plot();

loss, accuracy, f1_score, precision, recall = model_lstm.evaluate(X_test, y_test, verbose=0)

print('Test loss: {:.4f}'.format(loss))
print('Test accuracy: {:.4f}'.format(accuracy))
print('Test F1 score: {:.4f}'.format(f1_score))
print('Test precision: {:.4f}'.format(precision))
print('Test recall: {:.4f}'.format(recall))

#Model RNN+LSTM
vocab_size = 5000
embedding_size = 32
epochs=10
batch_size = 32

#Best Model piched by gridsearch
#Best: using {'activation': 'relu', 'dropout_rate': 0.1, 'neurons': 32}

#Create Model using the best hyperparameters
model_best= Sequential()
model_best.add(Embedding(vocab_size, embedding_size, input_length=max_len))
model_best.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model_best.add(MaxPooling1D(pool_size=2))
model_best.add(Bidirectional(LSTM(32)))
model_best.add(Dropout(0.1))
model_best.add(Dense(3, activation='softmax'))

#Model Plot
plot_model(model_best, show_shapes = True)

#Model Summary
model_best.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', tfa.metrics.F1Score(num_classes=3, average='macro'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])
print(model_best.summary())

#Training
history = model_best.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=batch_size, epochs=epochs, verbose=1)

#save best model
save_model(model_best,"model_RNN_LSTM.h5")



loss, accuracy, f1_score, precision, recall = model_best.evaluate(X_test, y_test, verbose=0)

print('Test loss: {:.4f}'.format(loss))
print('Test accuracy: {:.4f}'.format(accuracy))
print('Test F1 score: {:.4f}'.format(f1_score))
print('Test precision: {:.4f}'.format(precision))
print('Test recall: {:.4f}'.format(recall))

#Define function for predicting sentiment
def predict_sentiment(text):
    max_length = 100
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    # Tokenize the preprocessed text
    input_text = tokenizer.texts_to_sequences([preprocessed_text])
    # Pad the input sequence to the same length as the training data
    input_text = pad_sequences(input_text, maxlen=max_length, padding='post')
    # Make the prediction
    predicted_sentiment = model_best.predict(input_text)
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
