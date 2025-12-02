from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA

# --- 1. Cargar CSV ---
csv_path = "../data/TwoPatterns/TwoPatterns_all_data.csv"
df = pd.read_csv(csv_path)
print(f"✅ Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas")

# --- 2. Separar features y labels ---
X = df.drop(columns=["class"]).values.astype(np.float32)
y = df["class"].values.astype(int)

# --- 3. Crear DataLoader ---
tensor_x = torch.tensor(X)
tensor_y = torch.tensor(y)
train_data = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(train_data, batch_size=16, shuffle=False)

# --- 4. Cargar modelo LLaMA cuantizado ---
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

# --- 5. Conversión tensor → texto ---
def tensor_to_text(seq, precision=2):
    scale = 10 ** precision

    # Handle different input shapes
    if seq.dim() == 1:
        # Univariate: [timesteps] → treat as single channel
        seq = seq.unsqueeze(0)  # Now [1, timesteps]

    # Now seq is [n_channels, timesteps]
    channel_texts = []
    for ch_id in range(seq.shape[0]):
        channel = seq[ch_id]  # Get the channel [timesteps]
        channel_scaled = torch.round(channel * scale).long()

        # Convert to list of integers
        values_list = channel_scaled.tolist()
        if isinstance(values_list, int):  # Single value case
            values_list = [values_list]

        # Create text representation
        values_str = " ".join([str(int(v)) for v in values_list])
        channel_text = f"Channel {ch_id + 1}: values {values_str}"
        channel_texts.append(channel_text)

    return " | ".join(channel_texts)

# --- 6. Extraer embeddings ---
all_feats = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in tqdm(train_loader, desc="Extracting embeddings from LLaMA"):
        batch_text = [tensor_to_text(seq) for seq in x_batch]
        inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True, max_length=1000).to("cuda")
        outs = llama(**inputs, output_hidden_states=True)
        embeddings = outs.hidden_states[-1].mean(dim=1)
        all_feats.append(embeddings.cpu().numpy())
        all_labels.append(y_batch.numpy())

X = np.concatenate(all_feats, axis=0)
y = np.concatenate(all_labels, axis=0)
print(f"✅ Embeddings shape: {X.shape}, Labels shape: {y.shape}")

df_emb_original = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
df_emb_original["label"] = y
df_emb_original.to_csv("llama_embeddings_twopatterns_original.csv", index=False)
print("💾 Saved ORIGINAL embeddings to llama_embeddings_twopatterns_original.csv")

# --- 8. Aplicar PCA y guardar embeddings REDUCIDOS ---
pca = PCA(n_components=128, random_state=42)
X_reduced = pca.fit_transform(X)
print(f"✅ Reduced to {X_reduced.shape[1]} dims, variance retained: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# Guardar embeddings reducidos con PCA
df_emb_pca = pd.DataFrame(X_reduced, columns=[f"pca_{i}" for i in range(X_reduced.shape[1])])
df_emb_pca["label"] = y
df_emb_pca.to_csv("llama_embeddings_twopatterns_pca200.csv", index=False)
print("💾 Saved PCA-reduced embeddings to llama_embeddings_twopatterns_pca200.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.1, stratify=y, random_state=42
)

# Now GPU should work
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method='hist',
    device='cuda',
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 XGBoost accuracy (LLaMA embeddings on TwoPatterns): {acc*100:.2f}%")
print("\n--- Detailed Report ---")
print(classification_report(y_test, y_pred))
