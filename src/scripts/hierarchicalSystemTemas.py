from ProcessingData import ProcessingData
from AutoML import AutoML
import pandas as pd
import numpy as np
import json

PathsSentencesBinary = {
            "chica_ocupada": "./data/separacionTemas/chica_ocupada_binary.tsv",
            "fugu": "./data/separacionTemas/fugu_binary.tsv",
            "lugar_favorito": "./data/separacionTemas/lugar_favorito_binary.tsv",
}

PathsSentencesMulticlass = {
            "chica_ocupada": "./data/separacionTemas/chica_ocupada_multiclass.tsv",
            "fugu": "./data/separacionTemas/fugu_multiclass.tsv",
            "lugar_favorito": "./data/separacionTemas/lugar_favorito_multiclass.tsv",
}

PathsFeaturesBinary = {
            "chica_ocupada": "./data/featuresTemas/sentences_features_chica_binary.tsv",
            "fugu": "./data/featuresTemas/sentences_features_fugu_binary.tsv",
            "lugar_favorito": "./data/featuresTemas/sentences_features_lugar_binary.tsv"
}

PathsFeaturesMulticlass = {
            "chica_ocupada": "./data/featuresTemas/sentences_features_chica_multiclass.tsv",
            "fugu": "./data/featuresTemas/sentences_features_fugu_multiclass.tsv",
            "lugar_favorito": "./data/featuresTemas/sentences_features_lugar_multiclass.tsv"
}

PathsFeaturesFilterBinary = {
            "chica_ocupada": "./data/featuresFilterTemas/features_filter_chica_binary.tsv",
            "fugu": "./data/featuresFilterTemas/features_filter_fugu_binary.tsv",
            "lugar_favorito": "./data/featuresFilterTemas/features_filter_lugar_binary.tsv"
}

PathsFeaturesFilterMulticlass = {
            "chica_ocupada": "./data/featuresFilterTemas/features_filter_chica_multiclass.tsv",
            "fugu": "./data/featuresFilterTemas/features_filter_fugu_multiclass.tsv",
            "lugar_favorito": "./data/featuresFilterTemas/features_filter_lugar_multiclass.tsv"
}



for key in PathsSentencesBinary.keys():

    print(f"********** Extracting Features BinaryClass {key} **********")
    psDataBinaryClass = ProcessingData(PathsSentencesBinary[key], "Sentence", PathsFeaturesBinary[key], PathsFeaturesFilterBinary[key])
    psDataBinaryClass.processDataset()
    psDataBinaryClass.saveSelectedFeaturings(PathsFeaturesBinary[key])

    print(f"********** Extracting Features Multiclass {key} **********")
    psDataMultiClass = ProcessingData(PathsSentencesMulticlass[key], "Sentence", PathsFeaturesMulticlass[key], PathsFeaturesFilterMulticlass[key])
    psDataMultiClass.processDataset()
    psDataMultiClass.saveSelectedFeaturings(PathsFeaturesMulticlass[key])


    # Selecionar las k mejores características comunes
    print(f"********** Selecting common features {key} **********")
    psDataBinaryClass.dataframe = pd.read_csv(PathsFeaturesFilterBinary[key], sep='\t')
    psDataMultiClass.dataframe = pd.read_csv(PathsFeaturesFilterMulticlass[key], sep='\t')

    cols_exclude = ["Grupo", "CodigoSujeto"]

    featuresBinary = set(psDataBinaryClass.dataframe.columns) - set(cols_exclude)
    featuresMultiClass = set(psDataMultiClass.dataframe.columns) - set(cols_exclude)

    commonFatures = featuresBinary.intersection(featuresMultiClass)
    finalColumns = list(commonFatures) + cols_exclude
    with open (f"./data/commonFeatures/columns_{key}.json", "w") as f:
        json.dump(finalColumns, f)

    df_binaryClass = psDataBinaryClass.dataframe[finalColumns]
    df_multiClass = psDataMultiClass.dataframe[finalColumns]

    df_binaryClass.to_csv(PathsFeaturesFilterBinary[key], index=False, sep="\t")
    df_multiClass.to_csv(PathsFeaturesFilterMulticlass[key], index=False, sep="\t")


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


    print(f"********** AutoML {key} **********")
    automl = AutoML(10, 20, 10, 30)

    subjectsBinary = df_binaryClass["CodigoSujeto"]
    automl.fit(X_A.values, y_A, subjectsBinary, modo="binary")

    subjectsMulti = df_multiClass["CodigoSujeto"]
    automl.fit(X_B.values, y_B, subjectsMulti, modo="multilevel")


    print(f"********** Saving Models {key} **********")
    automl.saveModel(f"./models/model_automl_{key}_binaryClass.pkl", f"./models/model_automl_{key}_multiClass.pkl")

