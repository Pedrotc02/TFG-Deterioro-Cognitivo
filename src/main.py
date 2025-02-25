from ProcessingData import ProcessingData
from Modeling import Modeling
from AutoML import AutoML
import pandas as pd

Paths = {
            "sentences": "./data/dataset.tsv",
            "sentencesFeatures": "./data/features/sentences_features.tsv",
            "sentencesFeaturesFilter": "./data/features/sentences_features_filter.tsv",
}


psData = ProcessingData(Paths["sentences"], "Sentence", Paths["sentencesFeatures"], Paths["sentencesFeaturesFilter"])
psData.processDataset()
psData.saveSelectedFeaturings(Paths["sentencesFeatures"])

psData.dataframe = pd.read_csv(Paths["sentencesFeaturesFilter"])
df = psData.dataframe

X = df.drop(columns=["Grupo"])
X = X.select_dtypes(include=['number']).dropna()
y = df["Grupo"]

print("********** AutoML **********")
automl = AutoML("classification", 10, 30)
automl.fit(X, y)

automl.saveModel(f"./models/model_automl.pkl")
