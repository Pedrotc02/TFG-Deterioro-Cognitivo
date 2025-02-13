from processingData import processingData
from filterFeaturings import filterFeaturings
from modeling import modeling
import pandas as pd

interviewsPaths = {
            "chica_ocupada": "./data/chicaOcupada/chica_ocupada.csv",
            "fugu": "./data/fugu/fugu.csv",
            "lugar_favorito": "./data/lugarFavorito/lugar_favorito.csv"
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


def processInterview(topic):
    psData = processingData(interviewsPaths[topic], "Sentence", outputPaths[topic], outputPathsFilter[topic])
    psData.processDataset()
    psData.saveSelectedFeaturings(outputPaths[topic])

    psData.dataframe = pd.read_csv(outputPathsFilter["chica_ocupada"])
    df = psData.dataframe

    X = df.drop(columns=["Grupo"])
    X = X.select_dtypes(include=['number']).dropna()
    y = df["Grupo"]

    print(topic.upper())
    model = modeling(X, y)
    model.trainModel(X, y)



for topic in interviewsPaths.keys():
    processInterview(topic)