from abc import ABC, abstractmethod
import numpy as np

class WeakLearner(ABC):

    def __init__(self, **params) -> None:
        pass

    @abstractmethod
    def copy(self) -> "WeakLearner":
        raise NotImplementedError('Implement this method')

    @abstractmethod
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        raise NotImplementedError('Implement this method')

    @abstractmethod
    def predict(self, X:np.ndarray) -> np.ndarray:
        raise NotImplementedError('Implement this method')