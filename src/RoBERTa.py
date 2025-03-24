from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class RoBERTa:
    def __init__(self, text):
        self.text = text
        self.tokenizer = AutoTokenizer.from_pretrained("PedroTC/xlm-roberta-finetuned")
        self.model = AutoModelForSequenceClassification.from_pretrained("PedroTC/xlm-roberta-finetuned")
    

    def predict(self):
        inputs = self.tokenizer(self.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        return predicted_class
    


kdvj = RoBERTa("Pues en la zona elevada está el tamaño natural.")
print(kdvj.predict())