#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:04:12 2019

@author: jesskim
"""

#import the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer # CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords')

"""
class Car:
    def __init__(self):
        self.wheel = 4
        self.handle = 'circle'
    
    def count_wheels(self):
        return self.wheel
"""

class SentimentAnalysis:
    def __init__(self):
        self.dataset = None
        self.corpus = []

        self.document_processor = None
        self.classifier = None

        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def import_dataset(self, tsv_file, quoting=3):
        self.dataset = pd.read_csv(tsv_file, delimiter = '\t', quoting = quoting)

    def clean_text(self, text):
        text = re.sub('[^a-zA-Z]', ' ', text)  # to substitute non letters
        # with spaces
        text= text.lower()  # making all the words lowercase
        # make the string into list. that is spliting the review.
        text = text.split()  # split itself this becomes  a list ## TOKENIZER
        ps = PorterStemmer()  ## Lemmatizer
        text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
        text = ' '.join(text)  # joining the words in reivew as string together.

        return text

    def create_corpus(self):
        for i in range(self.dataset.shape[0]):
            review = self.dataset['Review'][i]
            cleaned_review = self.clean_text(review)
            self.corpus.append(cleaned_review)

        return self.corpus

    def load_document_processor(self, mode='TFIDF', max_feat=1500):
        if mode == 'TFIDF':
            self.document_processor = TfidfVectorizer(max_features=max_feat)
        elif mode == 'COUNT':
            self.document_processor = CountVectorizer(max_features=max_feat)
        else:
            print('Invalid Model! Choose from [TIFDF, COUNT]')

    def get_data(self, test_size=0.2, random_state=0):
        X = self.document_processor.fit_transform(self.corpus)
        Y = self.dataset.iloc[:, 1].values

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state)

    def load_classifier(self, mode):
        if mode == 'GAUSSIAN':
            self.classifier = GaussianNB()
        elif mode == 'DT':
            self.classifier = DecisionTreeClassifier(max_depth=10)


    def train_classifier(self):
        self.classifier.fit(self.X_train, self.Y_train)

    def predict(self):
        y_pred = self.classifier.predict(self.X_test)

        return y_pred

    def evaluate(self, y_pred):
        return confusion_matrix(self.Y_test, y_pred)


def main():
    sentiment_analysis = SentimentAnalysis()

    sentiment_analysis.import_dataset(tsv_file='Restaurant_Reviews.tsv')
    sentiment_analysis.load_document_processor(mode='TFIDF')
    sentiment_analysis.load_classifier(mode='DT')

    sentiment_analysis.create_corpus()
    sentiment_analysis.get_data()
    sentiment_analysis.train_classifier()

    y_pred = sentiment_analysis.predict()

    score = sentiment_analysis.evaluate(y_pred)

    print(score)


if __name__ == '__main__':
    main()