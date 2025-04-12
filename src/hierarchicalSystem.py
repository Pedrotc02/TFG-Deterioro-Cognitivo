from ProcessingData import ProcessingData
from AutoML import AutoML
import pandas as pd
import numpy as np

Paths = {
            "sentencesBinaryClass": "./data/separacionGrupos/dataset_clasebinaria.tsv",
            "sentencesMultiClass": "./data/separacionGrupos/dataset_multiclase.tsv",

            "sentencesFeaturesBinaryClass": "./data/features/sentences_features_binaryClass.tsv",
            "sentencesFeaturesMultiClass": "./data/features/sentences_features_multiClass.tsv",

            "sentencesFeaturesFilterBinaryClass": "./data/featuresFilter/sentences_features_filter_binaryClass.tsv",
            "sentencesFeaturesFilterMultiClass": "./data/featuresFilter/sentences_features_filter_multiClass.tsv",
}


print("********** Extracting Features BinaryClass**********")
psDataBinaryClass = ProcessingData(Paths["sentencesBinaryClass"], "TextoCompleto", Paths["sentencesFeaturesBinaryClass"], Paths["sentencesFeaturesFilterBinaryClass"])
psDataBinaryClass.processDataset()
psDataBinaryClass.saveSelectedFeaturings(Paths["sentencesFeaturesBinaryClass"])

print("********** Extracting Features Multiclass **********")
psDataMultiClass = ProcessingData(Paths["sentencesMultiClass"], "TextoCompleto", Paths["sentencesFeaturesMultiClass"], Paths["sentencesFeaturesFilterMultiClass"])
psDataMultiClass.processDataset()
psDataMultiClass.saveSelectedFeaturings(Paths["sentencesFeaturesMultiClass"])


# Selecionar las k mejores características
psDataBinaryClass.dataframe = pd.read_csv(Paths["sentencesFeaturesFilterBinaryClass"], sep='\t')
df_binaryClass = psDataBinaryClass.dataframe

psDataMultiClass.dataframe = pd.read_csv(Paths["sentencesFeaturesFilterMultiClass"], sep='\t')
df_multiClass = psDataMultiClass.dataframe


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
subjectsBinary = df_binaryClass["CodigoSujeto"]
automl_binaryClass = AutoML("classification",10, 20)
automl_binaryClass.fit(X_A.values, y_A, subjectsBinary, "./columns_test_train/binaryClass.pkl")

subjectsMulti = df_multiClass["CodigoSujeto"]
automl_multiclass = AutoML("classification",10, 30)
automl_multiclass.fit(X_B.values, y_B, subjectsMulti, "./columns_test_train/multiClass.pkl")


#Guardado de los modelos
automl_binaryClass.saveModel(f"./models/model_automl_binaryClass.pkl")
automl_multiclass.saveModel(f"./models/model_automl_multiClass.pkl")
