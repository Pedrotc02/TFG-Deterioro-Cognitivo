import numpy as np
import pandas as pd
from joblib import load


class Inference:
    def __init__(self, modelA_path, modelB_path):
        self.modelA_path = modelA_path
        self.modelB_path = modelB_path
        self.modelA = None
        self.modelB = None
    

    def loadModels(self):
        self.modelA = load(self.modelA_path)
        self.modelB = load(self.modelB_path)

        if self.modelA is None or self.modelB is None:
            raise ValueError("No se han cargado los modelos correctamente")


    def hierarchical_classification(self, X_input_modelA, X_input_modelB):
        pred_nivel1 = self.modelA.predict(X_input_modelA)
        resultado = []

        if X_input_modelA.shape[0] != X_input_modelB.shape[0]:
            raise ValueError("Las entradas para modelA y modelB no tienen el mismo número de muestras.")

        indices_modelB = [i for i, clase in enumerate(pred_nivel1) if clase != 0]

        if len(indices_modelB) > 0:
            X_modelB = X_input_modelB[indices_modelB]
            pred_nivel2 = self.modelB.predict(X_modelB)

            j = 0
            for i in range(len(pred_nivel1)):
                if pred_nivel1[i] == 0:
                    resultado.append(0)
                else:
                    resultado.append(pred_nivel2[j])
                    j += 1
        else:
            resultado = [0] * len(pred_nivel1)

        return np.array(resultado)


    

    def predict(self, X_input_modelA, X_input_modelB, output_path):
        if self.modelA is None or self.modelB is None:
            raise ValueError("Los modelos no están cargados. Carga los modelos primero.")
        
        pred = self.hierarchical_classification(X_input_modelA, X_input_modelB)

        data = {
            "Grupo-Prediccion": pred
        }

        df_result = pd.DataFrame(data)

        cols = ["Grupo-Prediccion"]
        df_result = df_result[cols]

        df_result.to_csv(output_path, sep='\t', index=False)

