# -*- coding: utf-8 -*-

import time
from flask import Flask, render_template, request
import nltk
import re
import string
nltk.download('wordnet')
from nltk.stem import SnowballStemmer
from sklearn.externals import joblib
from nltk.stem import WordNetLemmatizer

stemmer = SnowballStemmer('russian')
wordnet_lemmatizer = WordNetLemmatizer()

class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("../models/MultilNB.pkl")
        self.vectorizer = joblib.load("../models/vectorizer.pkl")
        self.classes_dict = {"neg": "negative",  "pos": "positive"}
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "neutral or uncertain"
        if probability < 0.7:
            return "probably"
        if probability > 0.95:
            return "certain"
        else:
            return ""

    def predict_text(self, text):
        vectorized = self.vectorizer.transform([self.prepare_text(text)])
        return self.model.predict(vectorized)[0], \
                self.model.predict_proba(vectorized)[0].max()


    def predict_list(self, list_of_texts):
        vectorized = self.vectorizer.transform(list_of_texts)
        return self.model.predict(vectorized), \
                self.model.predict_proba(vectorized)

    def get_prediction_message_and_score(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return (self.get_probability_words(prediction_probability) + " " +
                self.classes_dict[class_prediction], prediction_probability)

    def prepare_text(self, text):
        def hasNumbers(inputString):
            return any(char.isdigit() for char in inputString)

        def filt(sent):
            sent = self.regex.sub('', sent)
            words = [x.strip() for x in sent.split() if not hasNumbers(x)]
            result = []
            for word in words:
                word = stemmer.stem(word)
                result.append(word)
            return ' '.join(result)
        return filt(text.lower())


app = Flask(__name__)

print ("Preparing classifier")
start_time = time.time()
classifier = SentimentClassifier()
print ("Classifier is ready")
print (time.time() - start_time, "seconds")


@app.route("/", methods=["POST", "GET"])
def index(text="", prediction_message="", prob=0.0):
    if request.method == "POST":
        text = request.form["text"]
        print (text)
        prediction_message, prob = classifier.get_prediction_message_and_score(text)
        print (prediction_message)
        print (prob)

    return render_template('hello.html', text=text, prediction_message=prediction_message, prob=prob)


if __name__ == "__main__":
    app.run()
