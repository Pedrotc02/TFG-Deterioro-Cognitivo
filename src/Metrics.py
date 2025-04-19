import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report


class Metrics:
    def __init__(self, df):
        self.df = df
        self.y_true = df["Grupo"]
        self.y_pred = df["predicted_label"]

        self.mask = self.y_pred.notna()
        self.y_true_filtered = self.y_true[self.mask]
        self.y_pred_filtered = self.y_pred[self.mask]
        


    def evaluateGeneral(self):
        """
        Métricas generales del rendimiento del modelo
        """
        accuracy = accuracy_score(self.y_true_filtered, self.y_pred_filtered)
        macro_f1 = f1_score(self.y_true_filtered, self.y_pred_filtered, average='macro')
        micro_f1 = f1_score(self.y_true_filtered, self.y_pred_filtered, average='micro')
        weighted_f1 = f1_score(self.y_true_filtered, self.y_pred_filtered, average='weighted')
        

        result = (
            f"\n# MÉTRICAS GENERALES\n"
            f"# Accuracy: {accuracy:.4f}\n"
            f"# Macro F1: {macro_f1:.4f}\n"
            f"# Micro F1: {micro_f1:.4f}\n"
            f"# Weighted F1: {weighted_f1:.4f}\n"
        )

        return result
    

    def evaluateBinary(self):
        """
        Métricas para el modelo binario
        """
        y_true_binary = (self.y_true_filtered > 0).astype(int)
        y_pred_binary = (self.y_pred_filtered > 0).astype(int)

        precision_bin = precision_score(y_true_binary, y_pred_binary)
        recall_bin = recall_score(y_true_binary, y_pred_binary)
        f1_bin = f1_score(y_true_binary, y_pred_binary)
        accuracy_bin = accuracy_score(y_true_binary, y_pred_binary)

        result = (
            f"\n# MÉTRICAS BINARIAS (Deterioro > 0 vs 0)\n"
            f"# Precisión: {precision_bin:.4f}\n"
            f"# Recall: {recall_bin:.4f}\n"
            f"# F1 Score: {f1_bin:.4f}\n"
            f"# Accuracy: {accuracy_bin:.4f}\n"
        )

        return result
    

    def evaluateMultiClass(self):
        """
        Métricas para el modelo multinivel
        """
        report = classification_report(self.y_true_filtered, self.y_pred_filtered, digits=4)
        cm = confusion_matrix(self.y_true_filtered, self.y_pred_filtered)

        result = (
            f"\n# MÉTRICAS MULTICLASE (Grado de deterioro)\n"
            f"# Reporte:\n{report}\n"
            f"# Matriz de Confusión:\n{cm}\n"
        )

        return result
    

    def saveMetrics(self, output_path):
        full_metrics = (
            self.evaluateGeneral() +
            self.evaluateBinary() +
            self.evaluateMultiClass()
        )

        if output_path:
            with open(output_path, "a") as f:
                f.write("\n" + "#" * 40 + "\n")
                f.write(full_metrics)
                print(f"Métricas añadidas al final de {output_path}")
        else:
            print("No se ha especificado una ruta de salida para guardar las métricas.")






# Prueba 
df = pd.read_csv("./predictions/output_cognitive_roberta.tsv", sep="\t")
metrics = Metrics(df)
metrics.saveMetrics("./predictions/output_cognitive_roberta.tsv")