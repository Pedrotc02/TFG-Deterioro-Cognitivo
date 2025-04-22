from ProcessingData import ProcessingData
from AutoML import AutoML
import pandas as pd
import numpy as np
import json

Paths = {
            "sentencesBinaryClass": "./data/separacionGrupos/dataset_clasebinaria.tsv",
            "sentencesMultiClass": "./data/separacionGrupos/dataset_multiclase.tsv",

            "sentencesFeaturesBinaryClass": "./data/features/sentences_features_binaryClass.tsv",
            "sentencesFeaturesMultiClass": "./data/features/sentences_features_multiClass.tsv",

            "sentencesFeaturesFilterBinaryClass": "./data/featuresFilter/sentences_features_filter_binaryClass.tsv",
            "sentencesFeaturesFilterMultiClass": "./data/featuresFilter/sentences_features_filter_multiClass.tsv",
}


print("********** Extracting Features BinaryClass **********")
psDataBinaryClass = ProcessingData(Paths["sentencesBinaryClass"], "Sentence", Paths["sentencesFeaturesBinaryClass"], Paths["sentencesFeaturesFilterBinaryClass"])
psDataBinaryClass.processDataset()
psDataBinaryClass.saveSelectedFeaturings(Paths["sentencesFeaturesBinaryClass"])

print("********** Extracting Features Multiclass **********")
psDataMultiClass = ProcessingData(Paths["sentencesMultiClass"], "Sentence", Paths["sentencesFeaturesMultiClass"], Paths["sentencesFeaturesFilterMultiClass"])
psDataMultiClass.processDataset()
psDataMultiClass.saveSelectedFeaturings(Paths["sentencesFeaturesMultiClass"])


# Selecionar las k mejores características comunes
print("********** Selecting common features **********")
psDataBinaryClass.dataframe = pd.read_csv(Paths["sentencesFeaturesFilterBinaryClass"], sep='\t')
psDataMultiClass.dataframe = pd.read_csv(Paths["sentencesFeaturesFilterMultiClass"], sep='\t')

cols_exclude = ["Grupo", "CodigoSujeto"]

featuresBinary = set(psDataBinaryClass.dataframe.columns) - set(cols_exclude)
featuresMultiClass = set(psDataMultiClass.dataframe.columns) - set(cols_exclude)

commonFatures = featuresBinary.intersection(featuresMultiClass)
finalColumns = list(commonFatures) + cols_exclude
with open ("./data/commonFeatures/columns.json", "w") as f:
    json.dump(finalColumns, f)

df_binaryClass = psDataBinaryClass.dataframe[finalColumns]
df_multiClass = psDataMultiClass.dataframe[finalColumns]

df_binaryClass.to_csv(Paths["sentencesFeaturesFilterBinaryClass"], index=False, sep="\t")
df_multiClass.to_csv(Paths["sentencesFeaturesFilterMultiClass"], index=False, sep="\t")


# Rellenar valores NaN
df_binaryClass.replace([np.inf, -np.inf], np.nan, inplace=True)
df_binaryClass.fillna(df_binaryClass.mean(), inplace=True)
df_binaryClass.fillna(0, inplace=True)

df_multiClass.replace([np.inf, -np.inf], np.nan, inplace=True)
df_multiClass.fillna(df_multiClass.mean(), inplace=True)
df_multiClass.fillna(0, inplace=True)

# Separación jerárquica
X_A = df_binaryClass.drop(columns=["Grupo", "CodigoSujeto"])
X_A = X_A.select_dtypes(include=['number'])
y_A = df_binaryClass["Grupo"]
subjects = df_binaryClass["CodigoSujeto"]

X_B = df_multiClass.drop(columns=["Grupo", "CodigoSujeto"])
X_B = X_B.select_dtypes(include=['number'])
y_B = df_multiClass["Grupo"]
subjects = df_multiClass["CodigoSujeto"]


print("********** AutoML **********")
automl = AutoML(10, 20, 10, 30)

subjectsBinary = df_binaryClass["CodigoSujeto"]
automl.fit(X_A.values, y_A, subjectsBinary, modo="binary")

subjectsMulti = df_multiClass["CodigoSujeto"]
automl.fit(X_B.values, y_B, subjectsMulti, modo="multilevel")


print("********** Saving Models **********")
automl.saveModel(f"./models/model_automl_binaryClass.pkl", "./models/model_automl_multiClass.pkl")
