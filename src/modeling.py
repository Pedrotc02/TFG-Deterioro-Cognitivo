from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import numpy as np
import joblib

models = {
    'logistic_regression': LogisticRegression(penalty='l2', C=1, solver='lbfgs', max_iter=30000, random_state=42),
    'svm': svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    'gradient-boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, max_features='sqrt', min_samples_split=5, random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=5, random_state=42)
}

class modeling:

    def __init__(self, X, y):
        self.X = X
        self.y = y


    def trainModel(self, X_train, y_train):
        for name, model in models.items():
            """cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)""" #StratifiedKFold - No mejoria
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            print(f"{name} - Accuracy media: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
            model.fit(X_train, y_train)
            joblib.dump(model, f"./models/{name}.pkl")

