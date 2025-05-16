from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class AutoML:
    def __init__(self, generationsBinary, populationSizeBinary, generationsMulti, populationMulti, cv=5):
        self.generationsBinary = generationsBinary
        self.populationSizeBinary = populationSizeBinary
        self.generationsMulti = generationsMulti
        self.populationMulti = populationMulti
        self.cv = cv

        self.modelBinary = TPOTClassifier(
            generations=self.generationsBinary, 
            population_size=self.populationSizeBinary, 
            cv=self.cv, 
            verbosity=2, 
            n_jobs=-1,
            scoring='f1_weighted',
            random_state=42
        )

        self.modelMulti = TPOTClassifier(
            generations=self.generationsMulti, 
            population_size=self.populationMulti, 
            cv=self.cv, 
            verbosity=2, 
            n_jobs=-1,
            scoring='f1_weighted',
            random_state=42
        )
        

    def splitSubjects(self, X, y, subjects):
        train_subjects, test_subjects = train_test_split(subjects.unique(), test_size=0.2, random_state=42)

        train_indices = subjects.isin(train_subjects)
        test_indices = subjects.isin(test_subjects)

        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


    def fit(self, X, y, subjects, modo):
        if modo == "binary":
            y_binary = y.apply(lambda x: 1 if x != 0 else 0)
            X_train, X_test, y_train, y_test = self.splitSubjects(X, y_binary, subjects)
            self.modelBinary.fit(X_train, y_train)
            score_bin = self.modelBinary.score(X_test, y_test)
            print(f"Best Score (Modelo binario): {score_bin:.4f}")
            
            return score_bin
        
        elif modo == "multilevel":
            mask = y != 0
            X_multilevel = X[mask]
            y_multilevel = y[mask]
            subjects_multilevel = subjects[mask]
            X_train, X_test, y_train, y_test = self.splitSubjects(X_multilevel, y_multilevel, subjects_multilevel)
            self.modelMulti.fit(X_train, y_train)
            score_multi = self.modelMulti.score(X_test, y_test)
            print(f"Best Score (Modelo multiclase): {score_multi:.4f}")

            return score_multi
        
        else:
            raise ValueError("Modo no válido. Debe ser 'binary' o 'multilevel'.")


    def saveModel(self, filePathBinary, filePathMultilevel): 
        joblib.dump(self.modelBinary.fitted_pipeline_, filePathBinary)
        joblib.dump(self.modelMulti.fitted_pipeline_, filePathMultilevel)
        print("Modelos guardados correctamente")


    def loadModel(self, filePathBinary, filePathMultilevel):
        try:
            self.modelBinary = joblib.load(filePathBinary)
            self.modelMulti = joblib.load(filePathMultilevel)
            print("Modelos cargados correctamente")
        except:
            raise ValueError("Error al cargar los modelos")
    

    def predict(self, X):
        if self.modelBinary is None or self.modelMulti is None:
            raise ValueError("Los modelos no han sido cargados o entrenados")

        pred_bin = self.modelBinary.predict(X)[0]

        if pred_bin == 0:
            return 0 
        else:
            return self.modelMulti.predict(X)[0]


