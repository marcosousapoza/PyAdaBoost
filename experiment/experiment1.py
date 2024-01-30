from boost.adaboost import Adaboost
from learners.stump import DecisionStump
import numpy as np


def run():
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt

    # Create a sample dataset
    X, y = make_classification(n_samples=2000, n_features=2, n_informative=2, n_redundant=0)

    # Convert labels to {-1, +1}
    y = np.where(y == 0, -1, 1)

    # Initialize and train AdaBoost with DecisionStumps
    ada = Adaboost(k=10, estimator=DecisionStump())
    ada.fit(X, y)
    y_pred = ada.predict(X)

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red']  # Colors for different classes

    for class_value in [-1, 1]:
        color = colors[class_value == 1]  # Select color based on class value
        # Plot all points
        plt.scatter(X[y == class_value, 0], X[y == class_value, 1], label=f"True class {class_value}", color=color)

        # Circle the points with the predicted class
        plt.scatter(X[(y_pred == class_value) & (y == class_value), 0], 
                    X[(y_pred == class_value) & (y == class_value), 1], 
                    facecolors='none', edgecolors=color, s=100)
        
    error = accuracy_score(y, y_pred)

    plt.title(f"AdaBoost Classification | accuracy={error*100:2f}%")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()