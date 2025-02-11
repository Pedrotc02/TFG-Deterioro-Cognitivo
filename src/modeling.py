from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import joblib

models = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'svm': svm.SVC(kernel='rbf', C=1, gamma='scale'),
    'gradient-boosting': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, subsample=0.8, max_features='sqrt')
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

