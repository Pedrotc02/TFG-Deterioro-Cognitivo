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
        self.dataframe = pd.read_csv(self.filePath, sep=";")
        return self.dataframe[self.textColumn]


    def processDataset(self):
        texts = self.loadText()
        
        featuresList = []
        for text in texts:
            fExtraction = FeatureExtraction(text)
            featuresList.append(fExtraction.extractFeatures())

        featureDf = pd.DataFrame(featuresList)

        finalDf = pd.concat([self.dataframe, featureDf], axis=1)
        finalDf.to_csv(self.outputFilePath, index=False)


    def saveSelectedFeaturings(self, filePath):
        pr = pd.DataFrame(pd.read_csv(filePath))
        filter = FilterFeaturings()
        dfFilter = filter.bestFeaturings(pr.drop(columns=["Sentence", "Grupo"]), self.dataframe["Grupo"], 10)

        finaldf = pd.concat([dfFilter, self.dataframe["Grupo"]], axis=1)
        finaldf.to_csv(self.outputFilePathFilter, index=False)


