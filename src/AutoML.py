from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class AutoML:
    def __init__(self, generationsBinary, populationSizeBinary, generationsMulti, populationSizeMulti, cv=5):
        self.generationsBinary = generationsBinary
        self.populationSizeBinary = populationSizeBinary
        self.generationsMulti = generationsMulti
        self.populationSizeMulti = populationSizeMulti
        self.cv = cv

        self.binary_model = TPOTClassifier(
            generations=self.generationsBinary, 
            population_size=self.populationSizeBinary, 
            cv=self.cv, 
            verbosity=2, 
            n_jobs=-1,
            scoring='f1_weighted',
            random_state=42
        )
            
        self.multilevel_model = TPOTClassifier(
            generations=self.generationsMulti, 
            population_size=self.populationSizeMulti, 
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


    def fit(self, X, y, subjects):
        #Modelo binario
        y_binary = y.apply(lambda x: 0 if x == 0 else 1)

        X_train_bin, X_test_bin, y_train_bin, y_test_bin = self.splitSubjects(X, y_binary, subjects)

        self.binary_model.fit(X_train_bin, y_train_bin)
        score_bin = self.binary_model.score(X_test_bin, y_test_bin)
        print(f"Best Score (Modelo binario): {score_bin:.4f}")


        #Modelo multiclase
        group = y != 0
        X_multilevel = X[group]
        y_multilevel = y[group]
        subjects_multilevel = subjects[group]

        X_train_multi, X_test_multi, y_train_multi, y_test_multi = self.splitSubjects(X_multilevel, y_multilevel, subjects_multilevel)

        self.multilevel_model.fit(X_train_multi, y_train_multi)
        score_multi = self.multilevel_model.score(X_test_multi, y_test_multi)
        print(f"Best Score (Modelo multiclase): {score_multi:.4f}")


        #Puntuacion combinado
        num_bin = len(y_train_bin) + len(y_test_bin)
        num_multi = len(y_train_multi) + len(y_test_multi)
        total_samples = num_bin + num_multi

        weight_bin = num_bin / total_samples
        weight_multi = num_multi / total_samples

        score_combined = (score_bin * weight_bin) + (score_multi * weight_multi)

        print(f"Best Score (Modelo combinado): {score_combined:.4f}")

        return score_bin, score_multi, score_combined
    

    def saveModel(self, filePathBinary, filePathMultilevel): 
        joblib.dump(self.binary_model.fitted_pipeline_, filePathBinary)
        joblib.dump(self.multilevel_model.fitted_pipeline_, filePathMultilevel)
        print("Modelos guardados correctamente")


    def loadModel(self, filePathBinary, filePathMultilevel):
        try:
            self.binary_model = joblib.load(filePathBinary)
            self.multilevel_model = joblib.load(filePathMultilevel)
            print("Modelos cargados correctamente")
        except:
            raise ValueError("Error al cargar los modelos")
    

    def predict(self, X):
        if self.binary_model is None or self.multilevel_model is None:
            raise ValueError("Los modelos no han sido cargados o entrenados")

        pred_bin = self.model_binario.predict(X)[0]

        if pred_bin == 0:
            return 0 
        else:
            return self.model_multinivel.predict(X)[0]


