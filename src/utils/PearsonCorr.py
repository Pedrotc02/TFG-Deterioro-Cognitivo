import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

file_path = "./data/features/sentences_features_multiClass.tsv"

df = pd.read_csv(file_path, sep='\t')
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)
df = df.drop(columns=["CodigoSujeto"])

#Variable
target = "Grupo"

correlations = {}
for column in df.columns:
    if column != target:
        corr, _ = pearsonr(df[column], df[target])
        correlations[column] = corr


threshold = 0.1

bestFeatures = {feature: corr for feature, corr in correlations.items() if abs(corr) > threshold}

corr_df = pd.DataFrame.from_dict(bestFeatures, orient='index', columns=[target])

plt.figure(figsize=(10, 6))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlación de Pearson con Grupo")
plt.show()

for feature, corr_value in correlations.items():
    print(f"Correlación de {feature} con {target}: {corr_value:.4f}")
