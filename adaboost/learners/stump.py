from boost.learners.base import WeakLearner
import numpy as np

class DecisionStump(WeakLearner):

    def __init__(self):
        # Initialize the parameters of the decision stump
        self.feature_index = None
        self.threshold = None
        self.polarity = 1

    def copy(self):
        # Create a new instance of DecisionStump
        new_stump = DecisionStump()
        # Copy the parameters
        new_stump.feature_index = self.feature_index
        new_stump.threshold = self.threshold
        new_stump.polarity = self.polarity
        return new_stump

    def fit(self, X, y, sample_weight=None):
        # Initialize variables
        n_samples, n_features = X.shape
        # If no weights are provided, we assume uniform distribution
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples

        # Initialize variables to store the best stump parameters
        min_error = float('inf')
        # Loop over each feature
        for feature in range(n_features):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)
            # Try each threshold to split the data
            for threshold in thresholds:
                # Initialize polarity
                p = 1
                # Initialize prediction as all 1s
                prediction = np.ones(n_samples)
                # Label the samples whose value is less than the threshold as -1
                prediction[feature_values < threshold] = -1
                # Error equals sum of weights where predictions are wrong
                error = sum(sample_weight[prediction != y])
                # If the error is more than 50%, invert the polarity
                if error > 0.5:
                    error = 1 - error
                    p = -1
                # If this is the best threshold so far, store its parameters
                if error < min_error:
                    min_error = error
                    self.polarity = p
                    self.threshold = threshold
                    self.feature_index = feature

    def predict(self, X):
        # Make predictions using the decision stump
        n_samples = X.shape[0]
        feature_values = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1
        return predictions