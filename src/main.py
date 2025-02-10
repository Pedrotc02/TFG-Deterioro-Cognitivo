from processingData import processingData
from filterFeaturings import filterFeaturings
from modeling import modeling
import pandas as pd

interviewsPaths = {
            "chica_ocupada": "./data/chicaOcupada/chica_ocupada.csv",
            "fugu": "./data/fugu/fugu_v3_sp.csv",
            "lugar_favorito": "./data/lugarFavorito/lugar_favorito_v3_sp.csv"
}

outputPaths = {
            "chica_ocupada": "./data/chicaOcupada/chica_ocupada_features.csv",
            "fugu": "./data/fugu/fugu_features.csv",
            "lugar_favorito": "./data/lugarFavorito/lugar_favorito_features.csv"
}

outputPathsFilter = {
            "chica_ocupada": "./data/chicaOcupada/chica_ocupada_features_filter.csv",
            "fugu": "./data/fugu/fugu_features_filter.csv",
            "lugar_favorito": "./data/lugarFavorito/lugar_favorito_features_filter.csv"
}

psData = processingData(interviewsPaths["chica_ocupada"], "Sentence", outputPaths["chica_ocupada"], outputPathsFilter["chica_ocupada"])
psData.processDataset()
psData.saveSelectedFeaturings(outputPaths["chica_ocupada"])

psData.dataframe = pd.read_csv(outputPaths["chica_ocupada"])

X = psData.dataframe.drop(columns=["Audio", "Sentence", "Start", "End", "Class", "Type", "Offtopic", "CódigoSujeto", "Grupo"])
X = X.select_dtypes(include=['number']).dropna()
y = psData.dataframe["Grupo"]

filter = filterFeaturings()
X_filter = filter.bestFeaturings(X, y, 10)

model = modeling(X_filter, y)
model.trainModel(X_filter, y)