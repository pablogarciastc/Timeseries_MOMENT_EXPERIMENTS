import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from main import prepare_dataset
from textlets_multihead import TextEmbedderLETS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
# --- 1. Cargar los datos ---
base_path = Path('../data') / "dailysport"
x_train, y_train = prepare_dataset(base_path / 'x_train.pkl', base_path / 'state_train.pkl')
x_test, y_test = prepare_dataset(base_path / 'x_test.pkl', base_path / 'state_test.pkl')

# --- 2. Crear DataLoader ---
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=8, shuffle=False)

# --- 3. Instanciar encoder ---
encoder = TextEmbedderLETS(model_name="meta-llama/Llama-2-7b-hf").to("cuda").eval()

# --- 4. Extraer embeddings ---
all_feats = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in tqdm(train_loader, desc="Extracting embeddings"):
        x_batch = x_batch.to("cuda").float()
        feats = encoder(x_batch)  # [B, d_model]
        all_feats.append(feats.cpu().numpy())
        all_labels.append(y_batch.cpu().numpy())

X = np.concatenate(all_feats, axis=0)
y = np.concatenate(all_labels, axis=0)

print(f"✅ Embeddings shape: {X.shape}, Labels shape: {y.shape}")

# --- 5. Guardar a CSV ---
df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
df["label"] = y
csv_path = "textencoder_embeddings.csv"
df.to_csv(csv_path, index=False)
print(f"💾 Guardado CSV en {csv_path}")

# --- 6. Entrenar y evaluar Random Forest ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=270,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=1,  # Ajustar según desbalance
    tree_method='hist',
    device='cuda'
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Random Forest accuracy (solo embeddings): {acc*100:.2f}%")
print("\n--- Reporte detallado ---")
print(classification_report(y_test, y_pred))
