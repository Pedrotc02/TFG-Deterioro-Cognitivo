from sklearn.feature_selection import SelectKBest, f_classif

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

