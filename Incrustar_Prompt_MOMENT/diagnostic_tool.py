"""
Herramienta de Diagnóstico para Continual Learning con L2Prompt

Analiza:
1. Distribución de prompts por tarea
2. Similitudes entre task keys
3. Features por tarea (clustering)
4. Matriz de confusión de predicción de tareas
5. Análisis de gradientes y learning
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json


class ContinualLearningDiagnostic:
    """Herramienta completa de diagnóstico"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.diagnostics = {}
        
    def analyze_task_keys(self):
        """Analiza la separación entre task keys"""
        print("\n" + "="*80)
        print("ANÁLISIS DE TASK KEYS")
        print("="*80)
        
        # Obtener task keys
        task_keys = torch.stack([key.data for key in self.model.task_predictor.task_keys])
        n_tasks = task_keys.shape[0]
        
        # Normalizar
        task_keys_norm = F.normalize(task_keys, p=2, dim=1)
        
        # Matriz de similitud
        similarity_matrix = torch.matmul(task_keys_norm, task_keys_norm.T)
        similarity_matrix = similarity_matrix.cpu().numpy()
        
        print("\nMatriz de Similitud entre Task Keys (cosine):")
        print("(Valores cercanos a 1 = keys muy similares = problema!)")
        print(f"\n{' ':10}", end='')
        for i in range(n_tasks):
            print(f"Task{i+1:2d}  ", end='')
        print()
        
        for i in range(n_tasks):
            print(f"Task {i+1:2d}:  ", end='')
            for j in range(n_tasks):
                if i == j:
                    print(f"  1.00  ", end='')
                else:
                    print(f"{similarity_matrix[i,j]:6.3f}  ", end='')
            print()
        
        # Estadísticas
        off_diagonal = similarity_matrix[~np.eye(n_tasks, dtype=bool)]
        print(f"\nEstadísticas de similitud entre diferentes tareas:")
        print(f"  Media: {off_diagonal.mean():.3f}")
        print(f"  Max:   {off_diagonal.max():.3f}")
        print(f"  Min:   {off_diagonal.min():.3f}")
        print(f"  Std:   {off_diagonal.std():.3f}")
        
        if off_diagonal.max() > 0.7:
            print("\n⚠️  PROBLEMA DETECTADO: Task keys muy similares (>0.7)")
            print("    → Las tareas son difíciles de distinguir")
            print("    → Solución: aumentar task_loss_weight o usar orthogonal regularization")
        
        self.diagnostics['task_key_similarity'] = similarity_matrix
        
        return similarity_matrix
    
    def analyze_prompt_usage(self, task_data_loaders, task_id_current):
        """Analiza qué prompts usa cada tarea"""
        print("\n" + "="*80)
        print("ANÁLISIS DE USO DE PROMPTS")
        print("="*80)
        
        self.model.eval()
        prompt_usage_by_task = {}
        
        with torch.no_grad():
            for task_id, loader in task_data_loaders.items():
                if task_id > task_id_current:
                    continue
                    
                prompt_counts = torch.zeros(self.model.l2prompt.pool_size)
                
                for x_batch, _ in loader:
                    x_batch = x_batch.to(self.device)
                    
                    # Get embeddings
                    bsz, n_channels, seq_len = x_batch.shape
                    input_mask = torch.ones((bsz, seq_len), device=self.device)
                    
                    x_norm = self.model.moment.normalizer(x=x_batch, mask=input_mask, mode="norm")
                    x_norm = torch.nan_to_num(x_norm, nan=0, posinf=0, neginf=0)
                    patches = self.model.moment.tokenizer(x=x_norm)
                    enc_in = self.model.moment.patch_embedding(patches, mask=input_mask)
                    n_patches = enc_in.shape[2]
                    enc_in = enc_in.reshape(bsz * n_channels, n_patches, -1)
                    
                    # Get query
                    q_out = self.model.moment.encoder(inputs_embeds=enc_in)
                    query = q_out.last_hidden_state.mean(dim=1)
                    
                    # Select prompts
                    _, indices = self.model.l2prompt.select_prompts_from_query(query)
                    
                    # Count
                    for idx in indices.flatten():
                        prompt_counts[idx] += 1
                
                prompt_usage_by_task[task_id] = prompt_counts.cpu().numpy()
        
        # Visualización
        print(f"\nTop-{self.model.l2prompt.top_k} prompts más usados por tarea:")
        print(f"(Pool size: {self.model.l2prompt.pool_size})\n")
        
        overlap_matrix = np.zeros((task_id_current + 1, task_id_current + 1))
        
        for task_id in range(task_id_current + 1):
            counts = prompt_usage_by_task[task_id]
            top_indices = np.argsort(counts)[-10:][::-1]  # Top 10
            
            print(f"Task {task_id + 1}:")
            for idx in top_indices[:5]:
                pct = 100 * counts[idx] / counts.sum()
                print(f"  Prompt {idx:2d}: {counts[idx]:5.0f} veces ({pct:5.1f}%)")
            print()
            
        # Calcular solapamiento entre tareas
        for i in range(task_id_current + 1):
            for j in range(task_id_current + 1):
                counts_i = prompt_usage_by_task[i]
                counts_j = prompt_usage_by_task[j]
                
                # Top-K prompts de cada tarea
                top_k = 5
                top_i = set(np.argsort(counts_i)[-top_k:])
                top_j = set(np.argsort(counts_j)[-top_k:])
                
                # Jaccard similarity
                overlap = len(top_i & top_j) / len(top_i | top_j)
                overlap_matrix[i, j] = overlap
        
        print("\nMatriz de Solapamiento de Prompts (Jaccard sobre top-5):")
        print("(0 = sin solapamiento, 1 = mismo conjunto de prompts)")
        print(f"\n{' ':10}", end='')
        for i in range(task_id_current + 1):
            print(f"Task{i+1:2d}  ", end='')
        print()
        
        for i in range(task_id_current + 1):
            print(f"Task {i+1:2d}:  ", end='')
            for j in range(task_id_current + 1):
                print(f"{overlap_matrix[i,j]:6.3f}  ", end='')
            print()
        
        # Detectar problemas
        off_diag_overlap = overlap_matrix[~np.eye(task_id_current + 1, dtype=bool)]
        if len(off_diag_overlap) > 0 and off_diag_overlap.mean() > 0.5:
            print("\n⚠️  PROBLEMA DETECTADO: Alto solapamiento entre tareas (>0.5)")
            print("    → Tareas usan prompts muy similares")
            print("    → Solución: aumentar pool_size o reducir top_k")
        
        self.diagnostics['prompt_usage'] = prompt_usage_by_task
        self.diagnostics['prompt_overlap'] = overlap_matrix
        
        return prompt_usage_by_task, overlap_matrix
    
    def analyze_feature_space(self, task_data_loaders, task_id_current, max_samples=500):
        """Analiza la separación de features por tarea usando t-SNE"""
        print("\n" + "="*80)
        print("ANÁLISIS DEL ESPACIO DE FEATURES")
        print("="*80)
        
        self.model.eval()
        
        all_features = []
        all_task_ids = []
        all_predictions = []
        
        with torch.no_grad():
            for task_id, loader in task_data_loaders.items():
                if task_id > task_id_current:
                    continue
                
                task_features = []
                task_predictions = []
                
                for x_batch, _ in loader:
                    if len(task_features) * x_batch.size(0) >= max_samples:
                        break
                        
                    x_batch = x_batch.to(self.device)
                    bsz, n_channels, seq_len = x_batch.shape
                    
                    # Get features (same as in forward)
                    input_mask = torch.ones((bsz, seq_len), device=self.device)
                    x_norm = self.model.moment.normalizer(x=x_batch, mask=input_mask, mode="norm")
                    x_norm = torch.nan_to_num(x_norm, nan=0, posinf=0, neginf=0)
                    patches = self.model.moment.tokenizer(x=x_norm)
                    enc_in = self.model.moment.patch_embedding(patches, mask=input_mask)
                    n_patches = enc_in.shape[2]
                    enc_in = enc_in.reshape(bsz * n_channels, n_patches, -1)
                    
                    q_out = self.model.moment.encoder(inputs_embeds=enc_in)
                    query = q_out.last_hidden_state.mean(dim=1)
                    
                    selected_prompts, _ = self.model.l2prompt.select_prompts_from_query(query)
                    x_with_prompts = torch.cat([selected_prompts, enc_in], dim=1)
                    
                    outputs = self.model.moment.encoder(inputs_embeds=x_with_prompts)
                    hidden = outputs.last_hidden_state
                    pooled = hidden.mean(dim=1)
                    pooled = pooled.view(bsz, n_channels, -1).mean(dim=1)
                    
                    # Task prediction
                    task_info = self.model.task_predictor(pooled, training=False)
                    
                    task_features.append(pooled.cpu())
                    # FIX: Expand predicted_task to match batch size
                    predicted_task = task_info['predicted_task']
                    if isinstance(predicted_task, int):
                        # Single prediction for whole batch - repeat for each sample
                        task_predictions.extend([predicted_task] * bsz)
                    else:
                        # Per-sample predictions
                        task_predictions.extend(predicted_task.cpu().numpy().tolist())

                task_features = torch.cat(task_features, dim=0)
                all_features.append(task_features)
                all_task_ids.extend([task_id] * task_features.size(0))
                all_predictions.extend(task_predictions)

        # Concatenar todo
        all_features = torch.cat(all_features, dim=0).numpy()
        all_task_ids = np.array(all_task_ids)
        all_predictions = np.array(all_predictions)

        print(f"\nRecolectadas {all_features.shape[0]} muestras de {task_id_current + 1} tareas")

        # Matriz de confusión de predicción de tareas
        print("\nMatriz de Confusión de Predicción de Tareas:")
        print("(Filas = Ground Truth, Columnas = Predicción)")

        conf_matrix = np.zeros((task_id_current + 1, task_id_current + 1))
        for true_task in range(task_id_current + 1):
            mask = all_task_ids == true_task
            preds = all_predictions[mask]
            for pred_task in range(task_id_current + 1):
                count = np.sum(preds == pred_task)
                conf_matrix[true_task, pred_task] = count

        # Normalizar por fila
        conf_matrix_norm = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)

        print(f"\n{' ':10}", end='')
        for i in range(task_id_current + 1):
            print(f"Pred{i+1:2d}  ", end='')
        print()

        for i in range(task_id_current + 1):
            print(f"True {i+1:2d}:  ", end='')
            for j in range(task_id_current + 1):
                print(f"{conf_matrix_norm[i,j]:6.2%}  ", end='')
            print()

        # Accuracy por tarea
        print("\nTask Prediction Accuracy por tarea:")
        for i in range(task_id_current + 1):
            acc = conf_matrix_norm[i, i]
            print(f"  Task {i+1}: {acc:.2%}")

        # Detectar confusiones
        print("\nPrincipales confusiones (>10%):")
        for i in range(task_id_current + 1):
            for j in range(task_id_current + 1):
                if i != j and conf_matrix_norm[i, j] > 0.1:
                    print(f"  Task {i+1} → Task {j+1}: {conf_matrix_norm[i,j]:.2%}")

        self.diagnostics['confusion_matrix'] = conf_matrix_norm
        self.diagnostics['features'] = all_features
        self.diagnostics['task_ids'] = all_task_ids
        self.diagnostics['predictions'] = all_predictions

        return conf_matrix_norm

    def check_training_state(self):
        """Verifica el estado de congelamiento de parámetros"""
        print("\n" + "="*80)
        print("ESTADO DE PARÁMETROS")
        print("="*80)

        # Task keys
        print("\nTask Keys (requires_grad):")
        for i, key in enumerate(self.model.task_predictor.task_keys):
            status = "✓ Trainable" if key.requires_grad else "✗ Frozen"
            print(f"  Task {i+1}: {status}")

        # Classifier heads
        print("\nClassifier Heads:")
        for i, head in enumerate(self.model.classifier.heads):
            trainable_params = sum(p.requires_grad for p in head.parameters())
            total_params = sum(1 for _ in head.parameters())
            status = "✓ Trainable" if trainable_params > 0 else "✗ Frozen"
            print(f"  Task {i+1}: {status} ({trainable_params}/{total_params} params)")

        # L2Prompt
        print("\nL2Prompt Pool:")
        trainable_prompt = self.model.l2prompt.prompts.requires_grad
        trainable_keys = self.model.l2prompt.keys.requires_grad
        print(f"  Prompts: {'✓ Trainable' if trainable_prompt else '✗ Frozen'}")
        print(f"  Keys:    {'✓ Trainable' if trainable_keys else '✗ Frozen'}")

    def analyze_task_key_gradients(self, train_loader, task_id):
        """Analiza los gradientes de los task keys durante el entrenamiento"""
        print("\n" + "="*80)
        print(f"ANÁLISIS DE GRADIENTES - Task {task_id + 1}")
        print("="*80)

        self.model.train()

        # Una batch de entrenamiento
        x_batch, y_batch = next(iter(train_loader))
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # Forward
        local_labels = y_batch - (task_id * self.model.classes_per_task)
        logits, task_loss, task_acc, task_info = self.model.forward_with_task_loss(
            x_batch, y_batch, task_id
        )

        cls_loss = F.cross_entropy(logits, local_labels)
        total_loss = cls_loss + 0.5 * task_loss  # Usar task_loss_weight del entrenamiento

        # Backward
        total_loss.backward()

        # Analizar gradientes de task keys
        print("\nGradientes de Task Keys:")
        print(f"{'Task':<8} {'Grad Norm':<15} {'Grad Mean':<15} {'Grad Max':<15}")
        print("-" * 60)

        for i, key in enumerate(self.model.task_predictor.task_keys):
            if key.grad is not None:
                grad_norm = key.grad.norm().item()
                grad_mean = key.grad.mean().item()
                grad_max = key.grad.abs().max().item()
                print(f"Task {i+1:<3} {grad_norm:<15.6f} {grad_mean:<15.6f} {grad_max:<15.6f}")
            else:
                print(f"Task {i+1:<3} {'No gradient (frozen)'}")

        # Limpiar gradientes
        self.model.zero_grad()

        print("\nSimilitudes de la batch con cada Task Key:")
        task_keys_matrix = torch.stack([key.data for key in self.model.task_predictor.task_keys])
        task_keys_norm = F.normalize(task_keys_matrix, dim=-1)

        # Get features from batch
        bsz, n_channels, seq_len = x_batch.shape
        input_mask = torch.ones((bsz, seq_len), device=self.device)
        x_norm = self.model.moment.normalizer(x=x_batch, mask=input_mask, mode="norm")
        x_norm = torch.nan_to_num(x_norm, nan=0, posinf=0, neginf=0)
        patches = self.model.moment.tokenizer(x=x_norm)
        enc_in = self.model.moment.patch_embedding(patches, mask=input_mask)
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(bsz * n_channels, n_patches, -1)

        with torch.no_grad():
            q_out = self.model.moment.encoder(inputs_embeds=enc_in)
            query = q_out.last_hidden_state.mean(dim=1)

            selected_prompts, _ = self.model.l2prompt.select_prompts_from_query(query)
            x_with_prompts = torch.cat([selected_prompts, enc_in], dim=1)

            outputs = self.model.moment.encoder(inputs_embeds=x_with_prompts)
            hidden = outputs.last_hidden_state
            pooled = hidden.mean(dim=1)
            pooled = pooled.view(bsz, n_channels, -1).mean(dim=1)

            features_norm = F.normalize(pooled, dim=-1)
            similarities = torch.matmul(features_norm, task_keys_norm.t())

            print(f"\nSimilitud media con cada tarea (batch de Task {task_id + 1}):")
            for i in range(len(self.model.task_predictor.task_keys)):
                mean_sim = similarities[:, i].mean().item()
                max_sim = similarities[:, i].max().item()
                print(f"  Task {i+1}: mean={mean_sim:.3f}, max={max_sim:.3f}")

    def save_diagnostics(self, output_path):
        """Guarda los diagnósticos en archivo"""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        # Guardar matrices numpy
        np.savez(
            output_path / 'diagnostics.npz',
            task_key_similarity=self.diagnostics.get('task_key_similarity'),
            prompt_overlap=self.diagnostics.get('prompt_overlap'),
            confusion_matrix=self.diagnostics.get('confusion_matrix'),
            features=self.diagnostics.get('features'),
            task_ids=self.diagnostics.get('task_ids'),
            predictions=self.diagnostics.get('predictions')
        )

        print(f"\n✅ Diagnósticos guardados en {output_path}")


def run_diagnostics_after_task(trainer, task_id):
    """
    Función helper para ejecutar diagnósticos después de entrenar una tarea

    Args:
        trainer: ContinualLearningTrainer instance
        task_id: ID de la tarea recién entrenada
    """
    print("\n" + "="*80)
    print(f"DIAGNÓSTICO DESPUÉS DE TASK {task_id + 1}")
    print("="*80)

    diagnostic = ContinualLearningDiagnostic(trainer.model, trainer.device)

    # 1. Analizar task keys
    diagnostic.analyze_task_keys()

    # 2. Preparar data loaders para análisis
    task_loaders = {}
    for tid in range(task_id + 1):
        x_test, y_test = trainer.task_test_data[tid]
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(x_test, y_test)
        task_loaders[tid] = DataLoader(dataset, batch_size=32, shuffle=False)

    # 3. Analizar uso de prompts
    diagnostic.analyze_prompt_usage(task_loaders, task_id)

    # 4. Analizar espacio de features
    diagnostic.analyze_feature_space(task_loaders, task_id)

    # 5. Verificar estado de parámetros
    diagnostic.check_training_state()

    # 6. Analizar gradientes (si hay un siguiente task)
    if task_id < trainer.n_tasks - 1:
        next_task_id = task_id + 1
        x_train, y_train = trainer.task_train_data[next_task_id]
        from torch.utils.data import DataLoader, TensorDataset
        train_dataset = TensorDataset(x_train[:32], y_train[:32])  # Solo una batch
        train_loader = DataLoader(train_dataset, batch_size=32)
        diagnostic.analyze_task_key_gradients(train_loader, next_task_id)

    # Guardar
    diagnostic.save_diagnostics(Path(trainer.args.output_dir) / f'diagnostics_task_{task_id + 1}')

    return diagnostic