import pandas as pd
from FeatureExtraction import FeatureExtraction
from FilterFeaturings import FilterFeaturings

class ProcessingData:

    def __init__(self, filePath, textColumn, outputFilePath, outputFilePathFilter):
        self.filePath = filePath
        self.textColumn = textColumn
        self.dataframe = {}
        self.outputFilePath = outputFilePath
        self.outputFilePathFilter = outputFilePathFilter


    def loadText(self):
        self.dataframe = pd.read_csv(self.filePath, sep="\t")
        return self.dataframe[self.textColumn]


    def processDataset(self): 
        text = self.loadText()
        featuresList = []

        for index, row in self.dataframe.iterrows():
            text = row["Sentence"]
            subjectId = row["CodigoSujeto"]
            sentenceDuration = row["DuracionFrase"]
            wordDuration = row["DuracionPalabra"]

            fExtraction = FeatureExtraction(text, sentenceDuration, wordDuration)
            features = fExtraction.extractFeatures()

            features["CodigoSujeto"] = subjectId
            featuresList.append(features)

        featureDf = pd.DataFrame(featuresList)
        aggFeatureDf = featureDf.groupby("CodigoSujeto").agg(["mean", "min", "max", "std"])

        aggFeatureDf.columns = ["_".join(col).strip() for col in aggFeatureDf.columns.values]
        aggFeatureDf.reset_index(inplace=True)

        finalDf = pd.merge(self.dataframe.drop(columns="Sentence"), aggFeatureDf, on="CodigoSujeto", how="left")
        finalDf.to_csv(self.outputFilePath, index=False, sep="\t")


    def saveSelectedFeaturings(self, filePath):
        pr = pd.DataFrame(pd.read_csv(filePath))
        filter = FilterFeaturings()
        dfFilter = filter.bestFeaturings(pr.drop(columns=["CodigoSujeto", "Grupo"]), self.dataframe["Grupo"], 10)

        finaldf = pd.concat([dfFilter, self.dataframe["Grupo"]], axis=1)
        finaldf.to_csv(self.outputFilePathFilter, index=False, sep="\t")


