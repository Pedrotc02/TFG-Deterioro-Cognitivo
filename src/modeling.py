from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

models = {
    'logistic_regression': LogisticRegression(max_iter=1000)
}

class modeling:

    def __init__(self, X, y):
        self.X = X
        self.y = y


    def trainModel(self, X_train, y_train):
        for name, model in models.items():
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            print(f"{name} - Accuracy media: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
            model.fit(X_train, y_train)
            joblib.dump(model, f"./models/{name}.pkl")

