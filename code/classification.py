#!/usr/bin/env python -W ignore:: _DataConversionWarning

from itertools import compress
import numpy as np
from scipy import stats
from sklearn import svm, preprocessing
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (Normalizer,
                                   StandardScaler,
                                   FunctionTransformer)
from sklearn.metrics import classification_report

def to_dense(X):
        # Vectorizer outputs sparse matrix X
        # This function returns X as a dense matrix
        X = X.todense()
        X = np.nan_to_num(X)
        return X

def deltavectorizer(X):
        # Function that normalizes X to Delta score
        # "An expression of pure difference is what we need"
        #  - Burrows -> absolute Z-scores
        X = stats.zscore(X)
        X = np.abs(X)
        X = np.nan_to_num(X)
        return X

class PipeGridClassifier:

    def __init__(self, authors, titles, texts, n_feats, test_dict, invalid_words):
        self.authors = authors
        self.titles = titles
        self.texts = texts
        self.test_dict = test_dict
        self.n_feats = n_feats
        self.invalid_words = invalid_words

    def fit_transform_classify(self, visualize_db):
        """
        Parameters
        ===========
        X_train : list of documents (where a document is a list of sents,
            where a sent is a str)
        Y_train : list of string labels
        X_test : same format as X_train
        """

        # Split up data into training and testing set

        test_titles = []
        train_titles =  []
        X_train = []
        Y_train = []
        x_test = []
        y_test = []

        for author, title, text in zip(self.authors, self.titles, self.texts):
            if author in self.test_dict and title.split('_')[0] in self.test_dict.values():
                test_titles.append(title)
                y_test.append(author)
                x_test.append(text)
            else:
                train_titles.append(title)
                Y_train.append(author)
                X_train.append(text)

        # Translate author names to labels
        le = preprocessing.LabelEncoder()
        Y_train = le.fit_transform(Y_train)

        # Initialize pipeline
        # Steps in the pipeline:
        # 1 Vectorize incoming training material
        # 2 Normalize counts
        # 3 Classify

        pipe = Pipeline(
            [('vectorizer', TfidfVectorizer()),
             ('to_dense', FunctionTransformer(to_dense, accept_sparse=True)),
             ('feature_scaling', StandardScaler()),
             ('reduce_dim', SelectKBest(f_regression)),
             ('classifier', svm.SVC())])

        # GridSearch parameters

        # C parameter: optimize towards smaller-margin hyperplane (large C)
        # or larger-margin hyperplane (small C)
        # C is the penalty parameter of the error term.
        
        c_options = [1, 10, 100, 1000]
        n_features_options = list(range(30, self.n_feats, 30))
        kernel_options = ['linear']

        param_grid = [
            {
                'vectorizer': [TfidfVectorizer(stop_words=self.invalid_words)],
                'feature_scaling': [StandardScaler()],
                'reduce_dim': [SelectKBest(f_regression)],
                'reduce_dim__k': n_features_options,
                'classifier__C': c_options,
                'classifier__kernel': kernel_options,
            },
        ]

        # Cross Validation: KFold, StratifiedKFold, LeaveOneOut
        # Folds make sure that the training set is not diminished because of train_test_split
        # StratifiedKFold makes sure that the same percentage for each class is represented under each iteration
        # LeaveOneOut: One sample is left out per iteration, this being the test set

        grid = GridSearchCV(pipe, cv=LeaveOneOut(), n_jobs=2, param_grid=param_grid, verbose=1)
        grid.fit(X_train, Y_train)

        # Make prediction with the best parameters
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        accuracy = grid.best_score_ * 100
        predictions = grid.predict(x_test)

        # Once the results are there, make the fits of the finest Feature Extractor and Scaler
        # Also make sure that the test data, strictly unseen by the model, is set to the same format

        doc_vectors = grid.best_params_['vectorizer'].fit_transform(X_train).toarray()
        doc_vectors = grid.best_params_['feature_scaling'].fit_transform(doc_vectors)

        # grid_vectors (including train and test data) downsize to the amount of features
        # that, according to the Dimensionality Reduction, yield the greatest variance.
        # The concatenation of train and test data (grid_vectors) can be fed to PCA

        features = grid.best_params_['vectorizer'].get_feature_names()
        features_booleans = grid.best_params_['reduce_dim'].get_support()
        grid_features = list(compress(features, features_booleans))
        grid_nfeats = len(grid_features)
        grid_vocab = {index: i for i, index in enumerate(grid_features)}

        # Terminal Results
        print()
        print("::: Best model :::")
        print()
        print(best_model)
        print()
        print(best_params)
        print()
        print("::: Score :::", "%g" % accuracy)
        print()
        print("::: Predictions :::")
        print()
        for prediction, title in zip(le.inverse_transform(predictions), test_titles):
            print(prediction, title)

        return grid_vocab, grid_nfeats
