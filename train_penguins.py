import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path(__file__).parent / "web"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_penguins()[[
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "species",
]].dropna()

feature_names = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

class_names = sorted(df["species"].unique().tolist())
class_to_idx = {name: i for i, name in enumerate(class_names)}

X = df[feature_names].to_numpy(dtype=np.float32)
y = df["species"].map(class_to_idx).to_numpy(dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

class PenguinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        return self.net(x)

model = PenguinNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 250
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    train_pred = model(X_train_t).argmax(dim=1)
    test_logits = model(X_test_t)
    test_pred = test_logits.argmax(dim=1)
    train_acc = (train_pred == y_train_t).float().mean().item()
    test_acc = (test_pred == y_test_t).float().mean().item()

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

torch.save(model.state_dict(), OUT_DIR / "penguin_model.pt")

dummy_input = torch.randn(1, 4, dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    OUT_DIR / "penguin_model.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=17,
    external_data=False
)

metadata = {
    "feature_names": feature_names,
    "class_names": class_names,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "test_accuracy": round(test_acc, 4),
    "train_accuracy": round(train_acc, 4),
}

with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Saved files:")
print(" - web/penguin_model.pt")
print(" - web/penguin_model.onnx")
print(" - web/metadata.json")
