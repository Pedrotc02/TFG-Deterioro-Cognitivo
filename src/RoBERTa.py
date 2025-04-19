from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import torch

class RoBERTa:
    def __init__(self):
        self.tokenizerBinary = AutoTokenizer.from_pretrained("PedroTC/multi_model")
        self.modelBinary = AutoModelForSequenceClassification.from_pretrained("PedroTC/multi_model")

        self.tokenizerMulti = AutoTokenizer.from_pretrained("PedroTC/multi_model")
        self.modelMulti = AutoModelForSequenceClassification.from_pretrained("PedroTC/multi_model")
    

    def _predict(self, model, tokenizer, texts):
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

        with torch.no_grad():
            outputs = model(**tokens)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

        return preds.cpu().numpy()


    def hierarchicalClassfication(self, texts):
        pred_nivel1 = self._predict(self.modelBinary, self.tokenizerBinary, texts)
        resultado = []

        indices_modelB = [i for i, clase in enumerate(pred_nivel1) if clase != 0]

        if indices_modelB:
            texts_modelB = [texts[i] for i in indices_modelB]
            pred_nivel2 = self._predict(self.modelBinary, self.tokenizerMulti, texts_modelB)

            j = 0
            for i in range(len(pred_nivel1)):
                if pred_nivel1[i] == 0:
                    resultado.append(0)
                else:
                    resultado.append(pred_nivel2[j])
                    j += 1
        else:
            resultado = [0] * len(pred_nivel1)

        return np.array(resultado)
    

    def predict(self, input_path, output_path, text_column="Sentence"):
        if self.modelBinary is None or self.modelMulti is None:
            raise ValueError("Los modelos no están cargados. Carga los modelos primero")
        
        df = pd.read_csv(input_path, sep="\t")
        if text_column not in df.columns:
            raise ValueError(f"La columna '{text_column}' no se encuentra en el DataFrame.")
        
        texts = df[text_column].astype(str).tolist()
        pred = self.hierarchicalClassfication(texts)

        df["predicted_label"] = pred

        df.to_csv(output_path, sep='\t', index=False)

    


roberta = RoBERTa()
roberta.predict("./data/only_sentences.tsv", "./predictions/output_cognitive_roberta.tsv")