
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from main import prepare_dataset
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

# --- 2. Load quantized LLaMA model directly ---
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

llama = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
).eval()

d_model = llama.config.hidden_size
print(f"LLaMA loaded successfully with hidden_size={d_model}")

# --- 3. Encode data directly ---
def tensor_to_text(seq, precision=2):
    """Versión mejorada con contexto estadístico"""
    channel_texts = []
    for ch_id, channel in enumerate(seq):
        # Valores escalados
        scale = 10 ** precision
        channel_scaled = torch.round(channel * scale).long().tolist()
        values_str = " ".join([str(int(v)) for v in channel_scaled[:50]])  # Limitar longitud

        channel_text = (
            f"Channel {ch_id + 1}: "
            f"values: {values_str}"
        )
        channel_texts.append(channel_text)

    return " | ".join(channel_texts)

def format_series_as_text(seq):
    scale = 10 ** 2
    channel_texts = []
    for ch_id, channel in enumerate(seq):
        channel_scaled = torch.round(channel * scale).long().tolist()
        channel_scaled = channel_scaled[:50]
        str_values = []
        for v in channel_scaled:
            if v < 0:
                digits = list(str(abs(int(v))))
                spaced = "- " + " ".join(digits)
            else:
                digits = list(str(int(v)))
                spaced = " ".join(digits)
            str_values.append(spaced)
        joined = " , ".join(str_values)
        channel_texts.append(f"Channel {ch_id + 1} : values: {joined} ")
    return " ".join(channel_texts).strip()



all_feats = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in tqdm(train_loader, desc="Extracting embeddings from LLaMA"):
        #batch_text = [tensor_to_text(seq) for seq in x_batch]
        batch_text = [tensor_to_text(seq) for seq in x_batch]

        inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=1000).to("cuda")
        outs = llama(**inputs, output_hidden_states=True)
        embeddings = outs.hidden_states[-1].mean(dim=1)
        all_feats.append(embeddings.cpu().numpy())
        all_labels.append(y_batch.numpy())

X = np.concatenate(all_feats, axis=0)
y = np.concatenate(all_labels, axis=0)
print(f"✅ Embeddings shape: {X.shape}, Labels shape: {y.shape}")

# --- 4. Save to CSV ---
df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
df["label"] = y
df.to_csv("llama_embeddings.csv", index=False)
print("💾 Saved embeddings to llama_embeddings.csv")

# --- 5. Train Random Forest to evaluate ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

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

print(f"\n🎯 XGBoost accuracy (LLaMA embeddings only): {acc*100:.2f}%")
print("\n--- Detailed Report ---")
print(classification_report(y_test, y_pred))
