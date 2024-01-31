# AdaPy
AdaPy is a Python package providing a user-friendly implementation of the AdaBoost algorithm, designed for educational purposes. The algorithm is built on the concept of exponential loss as it was first introduced by Yoav Freund and Robert Schapire in 1995 [[1]](https://doi.org/10.1007%2F3-540-59119-2_166).


## Features
- Implementation of the AdaBoost algorithm with a focus on clarity.
- Includes a basic Decision Stump weak learner, with the flexibility to integrate other weak learners.
- Easy-to-use interface for training AdaBoost models and making predictions.


## Installation
To install `adapy`, simply run:
```bash
pip install adapy
```

## Usage
Here's a quick example of how to use AdaBoostKit:
```python
from adapy.boost import Adaboost
from adapy.learners import DecisionStump
from sklearn.datasets import make_classification
import numpy as np

# Create a sample dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0)
y = np.where(y == 0, -1, 1)  # Convert labels to {-1, +1}

# Initialize and train AdaBoost with DecisionStumps
ada = Adaboost(k=10, estimator=DecisionStump())
ada.fit(X, y)

# Make predictions
y_pred = ada.predict(X)
```

## Requirements
- Python 3.x
- NumPy


## Contributions
Feel free to add more feetures. Just sent me a PR and I will add it.
