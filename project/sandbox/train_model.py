import torch
import torch.nn as nn
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("dataset.csv")

# Drop index column
if df.columns[0] not in ["statement", "text"]:
    df = df.drop(df.columns[0], axis=1)

# Rename Kaggle columns
df = df.rename(columns={"statement": "text", "status": "label"})

# Clean text
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df = df.dropna(subset=["text", "label"])

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])
joblib.dump(le, "label_encoder.pkl")

# ---------------- TOKENIZATION ----------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

encodings = tokenizer(
    df["text"].tolist(),
    padding=True,
    truncation=True,
    max_length=64,   # ⬅ REDUCED length (very important)
    return_tensors="pt"
)

input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
labels = torch.tensor(df["label"].values)

X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_train_ids, X_train_mask, y_train)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # ⬅ SMALL batch

# ---------------- MODEL ----------------
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
        for param in self.bert.parameters():   # ⬅ FREEZE BERT
            param.requires_grad = False
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

model = MentalHealthModel(len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---------------- TRAIN ----------------
for epoch in range(5):   # ⬅ fewer epochs
    total_loss = 0
    for batch in train_loader:
        ids, mask, labels = [x.to(device) for x in batch]

        optimizer.zero_grad()
        outputs = model(ids, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "model.pth")
print("\n✅ Training completed successfully")
