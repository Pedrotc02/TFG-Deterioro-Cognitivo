from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
import joblib
import warnings

class AutoML:
    def __init__(self, problemType, generations, populationSize, cv=5):
        self.problemType = problemType
        self.generations = generations
        self.populationSize = populationSize
        self.cv = cv
        
        if self.problemType == "classification":
            self.model = TPOTClassifier(
                generations=self.generations, 
                population_size=self.populationSize, 
                cv=self.cv, 
                verbosity=2, 
                n_jobs=-1,
                scoring='balanced_accuracy',
                random_state=42
            )
            
        elif self.problemType == "regression":
            self.model = TPOTRegressor(
                generations=self.generations, 
                population_size=self.populationSize, 
                cv=self.cv, 
                verbosity=2, 
                n_jobs=-1,
                scoring='balanced_accuracy',
                random_state=42
            )

        else:
            raise ValueError("Tipo de problema no válido. Usa classification o regression")


    def fit(self, X, y, subjects):
        train_subjects, test_subjects = train_test_split(subjects.unique(), test_size=0.2, random_state=42)

        if self.model is None:
            raise ValueError("No se ha inicializado el modelo")
        
        train_indices = subjects.isin(train_subjects)
        test_indices = subjects.isin(test_subjects)

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)
        print(f"Best Score: {score:.4f}")

        return score
    

    def saveModel(self, filePath):
        if self.model is None:
            raise ValueError("No se ha entrenado el modelo")
        
        joblib.dump(self.model.fitted_pipeline_, filePath)
    

    def loadModel(self, filePath):
        try:
            modeloCargado = joblib.load(filePath)
            if modeloCargado is not None:
                self.model = modeloCargado
            else:
                raise ValueError("El archivo del modelo esta vacio")
        except:
            raise ValueError("No se pudo cargar el modelo")
    

    def predict(self, X):
        if self.model is None:
            raise ValueError("No se ha entrenado el modelo")
        
        return self.model.predict(X)

