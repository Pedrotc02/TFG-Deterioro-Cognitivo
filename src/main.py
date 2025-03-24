from ProcessingData import ProcessingData
from Modeling import Modeling
from AutoML import AutoML
import pandas as pd
import numpy as np

Paths = {
            "sentences": "./data/dataset.tsv",
            "sentencesFeatures": "./data/features/sentences_features.tsv",
            "sentencesFeaturesFilter": "./data/features/sentences_features_filter.tsv",
}


"""print("********** Extracting Features **********")
psData = ProcessingData(Paths["sentences"], "Sentence", Paths["sentencesFeatures"], Paths["sentencesFeaturesFilter"])
psData.processDataset()
psData.saveSelectedFeaturings(Paths["sentencesFeatures"])"""

# Selecionar las k mejores características

"""psData.dataframe = pd.read_csv(Paths["sentencesFeaturesFilter"], sep='\t')
df = psData.dataframe"""


df = pd.read_csv(Paths["sentencesFeaturesFilter"], sep='\t')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)

X = df.drop(columns=["Grupo", "CodigoSujeto"])
X = X.select_dtypes(include=['number'])
y = df["Grupo"]
subjects = df["CodigoSujeto"]

print("********** AutoML **********")
automl = AutoML(10, 20, 10, 30)
automl.fit(X.values, y, subjects)

automl.saveModel(f"./models/model_automl_binary.pkl", "./models/model_automl_multilevel.pkl")
