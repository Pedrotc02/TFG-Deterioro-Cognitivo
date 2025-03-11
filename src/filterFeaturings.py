from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

class FilterFeaturings:
    
    def __init__(self):
        self.dataframe = {}
    

    def bestFeaturings(self, X, y, k):
        X = X.drop(columns = X.columns[X.nunique() == 1], errors = 'ignore')
        X = X.select_dtypes(include=['number'])
        selector = SelectKBest(score_func=f_classif, k = k)
        selector.fit(X, y)
        featuresSelected = X.columns[selector.get_support()]
        
        return X[featuresSelected]


    def bestFeaturesPearson(self, df, target, threshold):
        correlations = {}
        for column in df.columns:
            if column != target:
                corr = df[column].corr(df[target])
                if not np.isnan(corr):
                    correlations[column] = corr

        bestFeatures = [feature for feature, corr in correlations.items() if abs(corr) > threshold]

        if not bestFeatures:
            print("No features found")

        return df[bestFeatures]
