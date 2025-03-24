from ProcessingData import ProcessingData
from Modeling import Modeling
from AutoML import AutoML
import pandas as pd
import numpy as np

PathsSentences = {
            "chica_ocupada": "./data/separacionTemas/chica_ocupada.tsv",
            "fugu": "./data/separacionTemas/fugu.tsv",
            "lugar_favorito": "./data/separacionTemas/lugar_favorito.tsv",
}

PathsFeatures = {
            "chica_ocupada": "./data/features/sentences_features_chica.tsv",
            "fugu": "./data/features/sentences_features_fugu.tsv",
            "lugar_favorito": "./data/features/sentences_features_lugar.tsv"
}

PathsFeaturesFilter = {
            "chica_ocupada": "./data/featuresFilter/features_filter_chica.tsv",
            "fugu": "./data/featuresFilter/features_filter_fugu.tsv",
            "lugar_favorito": "./data/featuresFilter/features_filter_lugar.tsv"
}

for key in PathsSentences.keys():

    print(f"********** Extracting Features {key} **********")
    psData = ProcessingData(PathsSentences[key], "Sentence", PathsFeatures[key], PathsFeaturesFilter[key])
    psData.processDataset()
    psData.saveSelectedFeaturings(PathsFeatures[key])

    # Selecionar las k mejores características
    psData.dataframe = pd.read_csv(PathsFeaturesFilter[key], sep='\t')
    df = psData.dataframe

    df = pd.read_csv(PathsFeaturesFilter[key], sep='\t')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)

    X = df.drop(columns=["Grupo", "CodigoSujeto"])
    X = X.select_dtypes(include=['number'])
    y = df["Grupo"]
    subjects = df["CodigoSujeto"]
    
    #AutoML
    print(f"********** AutoML {key}**********")
    automl = AutoML("classification", 10, 50)
    automl.fit(X, y, subjects)

    automl.saveModel(f"./models/model_automl_{key}.pkl")
