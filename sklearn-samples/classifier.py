import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import time
import pandas as pd


## HELLO WORLD

AVAILABLE_CLASSIFIERS = {
    
    #################
    ### Total of 10 classifiers are tested.
    ### 6 are dropped.
    ### knn, gaussian_nb, qda: low scores.
    ### linearsvc, svc, gpc: long training time.
    #################

    #'knn': KNeighborsClassifier(6),
    #'linear_svc': SVC(kernel="linear", C=0.025),
    #'svc': SVC(gamma=2, C=1),
    #'gpc': GaussianProcessClassifier(1.0 * RBF(1.0)),
    #'gaussian_nb' : GaussianNB(),
    #'qda' : QuadraticDiscriminantAnalysis()

    'decision_tree': DecisionTreeClassifier(max_depth=10),
    'random_forest': RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
    'mlp' : MLPClassifier(alpha=1),
    'ada_boost' : AdaBoostClassifier(),
}

class Classifier:
    def __init__(self, user_features, book_features, ue_features, ratings, known_index):
        self.user_features = user_features
        self.book_features = book_features
        self.ue_features = ue_features

        self.ratings = ratings
        self.known_index = known_index

        self.ue_features = np.zeros((len(self.known_index), self.user_features.shape[1]+self.book_features.shape[1]))
        self.ratings_1d = np.zeros((len(self.known_index), 1))
        
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self.classifier = None
        self.proba_train = None
        self.score_train = None
        self.proba_test = None
        self.score_test = None
        self.logistic_regression = None

        self.prediction = None

    def get_dataset(self):
        """
        Reshaping dataset to fit classifier.
        """
        for i in range(len(self.known_index)):
            self.ratings_1d[i,] = self.ratings[self.known_index[i][0], self.known_index[i][1]]

        (self.X_train, self.X_test, self.Y_train, self.Y_test) = train_test_split(self.ue_features, self.ratings_1d)

        self.Y_train = self.Y_train * 5
        self.Y_test = self.Y_test * 5

        return self.X_train, self.X_test, self.Y_train, self.Y_test


    def train(self, clf='random_forest', save_model=''):
        """
        Parameter:
            clf - string. key of AVAILABLE_CLASSIFIERS
        Returns:
            self.classifier - One of the sklearn's classifier object.
            score - float. Evalution score.
            time - float. Time it took to train.
        """

        self.classifier = AVAILABLE_CLASSIFIERS[clf]
        
        start = time.time()
        self.classifier.fit(self.X_train, self.Y_train)
        score = self.classifier.score(self.X_test, self.Y_test)
        end = time.time()

        print(score)
        return (self.classifier, score, end-start)

    def to_logistic_regression(self):
        """
        Returns:
            self.logistic_regression model - One of the sklearn's regressor object.
            score - float. Evaluation score.
        """
        self.proba_train = self.classifier.predict_proba(self.X_train)
        self.score_train = self.Y_train / 5

        self.proba_test= self.classifier.predict_proba(self.X_test)
        self.score_test = self.Y_test / 5

        self.logistic_regression = DecisionTreeRegressor()
        self.logistic_regression.fit(self.proba_train, self.score_train)

        score = self.logistic_regression.score(self.proba_test, self.score_test)

        return (self.logistic_regression, score)

    def predict_by_user(self, user_idx):
        features = np.c_[np.tile(self.user_features[user_idx,:], (self.book_features.shape[0], 1)), self.book_features]

        proba_prediction = self.classifier.predict_proba(features)
        prediction = self.logistic_regression.predict(proba_prediction)

        return np.reshape(prediction, (1, prediction.shape[0]))

    def predict_rating(self):
        self.prediction = np.zeros((self.user_features.shape[0], self.book_features.shape[0]))

        for i in range(self.prediction.shape[0]):
            self.prediction[i,:] = self.predict_by_user(user_idx=i)

        return self.prediction

    def save_to_csv(self, out, userIds, bookIds):
        for index in self.known_index:
            self.prediction[index[0], index[1]] = self.ratings[index[0], index[1]]

        self.prediction = (self.prediction * 10).astype(int)

        df = pd.DataFrame(self.prediction, index=userIds, columns=bookIds)
        df.to_csv(out)

        print('Successfully saved the output at {}'.format(out))
        return None