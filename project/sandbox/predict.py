import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

le = joblib.load("label_encoder.pkl")

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.fc(x), dim=1)
        return torch.sum(w * x, dim=1)

class MentalHealthModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(768, 64, bidirectional=True, batch_first=True)
        self.attn = Attention(128)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        lstm_out, _ = self.lstm(bert_out)
        attn_out = self.attn(lstm_out)
        return self.fc(attn_out)

model = MentalHealthModel(len(le.classes_))
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "I feel anxious and mentally exhausted"
tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)

with torch.no_grad():
    outputs = model(tokens["input_ids"], tokens["attention_mask"])
    probs = torch.softmax(outputs, dim=1)

label = le.inverse_transform([probs.argmax().item()])[0]
confidence = probs.max().item()

print("\n🧠 Mental Health Prediction")
print("Input Text:", text)
print("Predicted Disorder:", label)
print("Confidence:", round(confidence * 100, 2), "%")
