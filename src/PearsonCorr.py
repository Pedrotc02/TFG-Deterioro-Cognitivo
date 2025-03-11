import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Cargar el CSV
file_path = "./data/features/sentences_features.tsv"  # Reemplaza con el nombre de tu archivo

df = pd.read_csv(file_path, sep='\t')  # Asume que los datos están separados por tabulaciones
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)
df = df.drop(columns=["CodigoSujeto"])

# Variable dependiente
target = "Grupo"

# Calcular la correlación de Pearson
correlations = {}
for column in df.columns:
    if column != target:
        corr, _ = pearsonr(df[column], df[target])
        correlations[column] = corr


threshold = 0.1

bestFeatures = {feature: corr for feature, corr in correlations.items() if abs(corr) > threshold}

# Convertir a DataFrame para el heatmap
corr_df = pd.DataFrame.from_dict(bestFeatures, orient='index', columns=[target])

# Graficar heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlación de Pearson con Grupo")
plt.show()

# Mostrar resultados
for feature, corr_value in correlations.items():
    print(f"Correlación de {feature} con {target}: {corr_value:.4f}")
