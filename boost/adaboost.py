
from typing import Any, List
from learners.base import WeakLearner
import numpy as np


class Adaboost:

    def __init__(
            self, k:int, 
            estimator:WeakLearner,
            **estimator_param:Any
        ) -> None:
        self._K = k
        self._estimator = estimator
        self._estimators:List[WeakLearner] = []
        self._estimator_weights:List[float] = []
        self._estimator_param = estimator_param

    def fit(self, X:np.ndarray, y:np.ndarray):
        N, D = X.shape
        w_m = np.ones(shape=N) / N
        for m in range(self._K):
            # STEP 1: train a new estimator h_m
            # sample data according to weights
            idxs = np.random.choice(N, N, replace=True, p=w_m/np.sum(w_m))
            X_train, y_train = X[idxs, :], y[idxs]
            # make predictions using samples -> approximation
            h_m = self._estimator.copy()
            h_m.fit(X_train, y_train)
            y_pred = h_m.predict(X)
            self._estimators.append(h_m)
            wrong_class = np.not_equal(y_pred, y).astype(int)
            # STEP 2: calculate weighted average epsilon
            epsilon = (w_m.T @ wrong_class) / np.sum(w_m)
            # STEP 3: update estimator weight alpha a_m
            a_m = np.log((1-epsilon)/epsilon)
            self._estimator_weights.append(a_m)
            # STEP 4: update weights
            w_m *= np.exp(a_m * wrong_class)

    def predict(self, X:np.ndarray):
        N, D = X.shape
        r_sum = np.zeros(shape=N)
        for h, a in zip(self._estimators, self._estimator_weights):
            y_pred = h.predict(X)
            r_sum += a * y_pred
        return np.sign(r_sum)
    

def exponential_loss(y_true, y_pred):
    return np.sum(np.exp(-y_true * y_pred))


