import numpy as np
import pandas as pd
from joblib import load
import FeatureExtraction as fe
import ProcessingData as pData
import json


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


    def hierarchical_classification(self, df, output_path, df_complete):
        pred_nivel1 = self.modelA.predict(df)

        resultado = []
        indices_modelB = [i for i, pred in enumerate(pred_nivel1) if pred != 0]

        if indices_modelB:
            features_modelB = df.iloc[indices_modelB]
            pred_nivel2 = self.modelB.predict(features_modelB)

            j = 0
            for i in range(len(pred_nivel1)):
                if pred_nivel1[i] == 0:
                    resultado.append(0)  # No hay deterioro
                else:
                    resultado.append(pred_nivel2[j])
                    j += 1
        else:
            resultado = [0] * len(pred_nivel1)  # Todos sanos

        df["CodigoSujeto"] = df_complete["CodigoSujeto"]
        df["Grupo"] = df_complete["Grupo"]
        df["predicted_label"] = resultado

        df.to_csv(output_path, sep='\t', index=False)
    




# Prueba de la clase Inference
df = pd.read_csv("./data/dataset.tsv", sep="\t")

agrupado = df.groupby(["CodigoSujeto", "Edad", "Grupo"]).agg({
        "Sentence": lambda Sentence: " ".join(Sentence),
        "DuracionFrase": "mean",
        "DuracionPalabra": "mean"
    }).reset_index()

agrupado.to_csv("./data/dataset_agrupado.tsv", sep="\t", index=False)


processingData = pData.ProcessingData("./data/dataset_agrupado.tsv", "Sentence", "./data/dataset_agrupado_features.tsv", "./data/dataset_agrupado_featuresFilter.tsv")
processingData.processDataset()

with open("./data/commonFeatures/columns.json", "r") as f:
    selected_features = json.load(f)

inference = Inference("./models/model_automl_binaryClass.pkl", "./models/model_automl_multiclass.pkl")
inference.loadModels()
df = pd.read_csv("./data/dataset_agrupado_features.tsv", sep="\t")
dfComplete = df

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)
df.fillna(0, inplace=True)

df = df[selected_features].copy()

inference.hierarchical_classification(df, "./predictions/output_cognitive_automl.tsv", dfComplete)