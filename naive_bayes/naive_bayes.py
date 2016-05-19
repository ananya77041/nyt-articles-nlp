from __future__ import division
from collections import Counter, defaultdict
import numpy as np
import itertools

class NaiveBayes(object):
    def __init__(self, alpha=1):
        """
        INPUT:
        -alpha: float, laplace smoothing constant.
        """
        self.class_totals = defaultdict(int)
        self.class_feature_totals = defaultdict(Counter)
        self.class_counts = None
        self.alpha = alpha
        self.p = None

    def _compute_likelihood(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels

        Compute the totals for each class and the totals for each feature
        and class.
        '''

        self.class_totals = defaultdict(int)
        self.class_feature_totals = defaultdict(Counter)

        #YOUR CODE HERE
        
    def fit(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels

        OUTPUT: None
        '''
        #Compute priors
        self.class_counts = Counter(y)

        #Compute number of features.
        self.p = len(set(itertools.chain(*X)))

        #Compute likelihoods
        self._compute_likelihood(X, y)

    def posteriors(self, X):
        '''
        INPUT:
        - X: List of list of tokens.

        OUTPUT: A list of counters. The keys of the counter
        will correspond to the possible labels, and the values
        will be the likelihood. (This is so we can use
        most_common method in predict).
        '''
        #YOUR CODE HERE.
        
    def predict(self, X):
        """
        INPUT:
        - X A list of lists of tokens.
        
        OUTPUT:
        -predictions: a numpy array with predicted labels.
        
        """
        return np.array([label.most_common(1)[0][0]
                         for label in self.posteriors(X)])

    def score(self, X, y):
        '''
        INPUT:
        - X: List of list of tokens.
        - y: numpy array, labels

        OUTPUT:
        - accuracy: float between 0 and 1

        Calculate the accuracy, the percent predicted correctly.
        '''

        return sum(self.predict(X) == y) / float(len(y))
