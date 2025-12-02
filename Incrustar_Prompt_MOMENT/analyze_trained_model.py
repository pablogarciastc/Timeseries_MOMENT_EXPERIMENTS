"""
Análisis Post-Hoc de Modelo Entrenado

Carga un checkpoint y ejecuta análisis detallado sin re-entrenar

Uso:
    python analyze_trained_model.py --checkpoint continual_results/final_model.pt \
        --dataset dailysport --n_tasks 6
"""

import sys
sys.path.insert(0, '/mnt/user-data/uploads')
sys.path.insert(0, '/home/claude')

import torch
import argparse
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from continual_learning import prepare_dataset
from moment_multihead import PromptedMOMENT
from diagnostic_tool import ContinualLearningDiagnostic
import torch.nn.functional as F


class ModelAnalyzer:
    """Analiza un modelo ya entrenado"""
    
    def __init__(self, checkpoint_path, dataset, n_tasks, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        
        # Cargar datos
        print("Cargando datos...")
        base_path = Path('../data') / dataset
        self.x_train, self.y_train = prepare_dataset(
            base_path / 'x_train.pkl', 
            base_path / 'state_train.pkl'
        )
        self.x_test, self.y_test = prepare_dataset(
            base_path / 'x_test.pkl', 
            base_path / 'state_test.pkl'
        )
        
        # Determinar clases por tarea
        n_classes = len(torch.unique(self.y_train))
        self.classes_per_task = n_classes // n_tasks
        
        self.task_classes = {}
        for i in range(n_tasks):
            start = i * self.classes_per_task
            end = (i + 1) * self.classes_per_task if i < n_tasks - 1 else n_classes
            self.task_classes[i] = list(range(start, end))
        
        print(f"Clases por tarea:")
        for tid, classes in self.task_classes.items():
            print(f"  Task {tid+1}: {classes}")
        
        # Crear data loaders por tarea
        self.task_test_loaders = {}
        for task_id, classes in self.task_classes.items():
            test_mask = torch.isin(self.y_test, torch.tensor(classes))
            x_test_task = self.x_test[test_mask]
            y_test_task = self.y_test[test_mask]
            dataset = TensorDataset(x_test_task, y_test_task)
            self.task_test_loaders[task_id] = DataLoader(
                dataset, batch_size=batch_size, shuffle=False
            )
        
        # Cargar checkpoint
        print(f"\nCargando checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recrear modelo con la misma arquitectura
        if 'config' in checkpoint:
            config = checkpoint['config']
            pool_size = config.get('pool_size', 20)
            prompt_length = config.get('prompt_length', 5)
            top_k = config.get('top_k', 5)
            moment_model = config.get('moment_model', 'small')
        else:
            # Valores por defecto
            pool_size = 20
            prompt_length = 5
            top_k = 5
            moment_model = 'small'
        
        print(f"Configuración del modelo:")
        print(f"  pool_size={pool_size}, prompt_length={prompt_length}")
        print(f"  top_k={top_k}, moment_model={moment_model}")
        
        self.model = PromptedMOMENT(
            n_tasks=n_tasks,
            classes_per_task=self.classes_per_task,
            pool_size=pool_size,
            prompt_length=prompt_length,
            top_k=top_k,
            moment_model=moment_model
        ).to(self.device)
        
        # Cargar pesos
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✅ Modelo cargado exitosamente\n")
    
    def quick_evaluation(self):
        """Evaluación rápida de accuracy"""
        print("="*80)
        print("EVALUACIÓN RÁPIDA")
        print("="*80)
        
        oracle_accs = {}
        soft_accs = {}
        task_pred_accs = {}
        
        for task_id in range(self.n_tasks):
            loader = self.task_test_loaders[task_id]
            
            # Oracle mode
            correct_oracle = 0
            total = 0
            
            # Soft mode
            correct_soft = 0
            correct_task_pred = 0
            
            with torch.no_grad():
                for x_batch, y_batch in loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    bsz = x_batch.size(0)
                    
                    # Oracle
                    logits_oracle = self.model(x_batch, task_id=task_id)
                    local_labels = y_batch - (task_id * self.classes_per_task)
                    _, pred_oracle = logits_oracle.max(1)
                    correct_oracle += pred_oracle.eq(local_labels).sum().item()
                    
                    # Soft
                    logits_soft, task_info = self.model(
                        x_batch, 
                        predict_task=True, 
                        return_task_info=True
                    )
                    _, pred_soft = logits_soft.max(1)
                    correct_soft += pred_soft.eq(local_labels).sum().item()
                    
                    # Task prediction accuracy
                    predicted_task = task_info['predicted_task']
                    if isinstance(predicted_task, int):
                        correct_task_pred += (predicted_task == task_id) * bsz
                    else:
                        correct_task_pred += (predicted_task == task_id).sum().item()
                    
                    total += bsz
            
            oracle_accs[task_id] = 100 * correct_oracle / total
            soft_accs[task_id] = 100 * correct_soft / total
            task_pred_accs[task_id] = 100 * correct_task_pred / total
        
        # Imprimir resultados
        print("\n--- Oracle Mode ---")
        for tid, acc in oracle_accs.items():
            print(f"Task {tid+1}: {acc:.2f}%")
        print(f"Average: {np.mean(list(oracle_accs.values())):.2f}%")
        
        print("\n--- Soft Prediction Mode ---")
        for tid, acc in soft_accs.items():
            print(f"Task {tid+1}: {acc:.2f}% (Task Pred: {task_pred_accs[tid]:.2f}%)")
        print(f"Average Acc: {np.mean(list(soft_accs.values())):.2f}%")
        print(f"Average Task Pred: {np.mean(list(task_pred_accs.values())):.2f}%")
        
        return oracle_accs, soft_accs, task_pred_accs
    
    def run_full_diagnostics(self, output_dir='analysis_output'):
        """Ejecuta diagnósticos completos"""
        print("\n" + "="*80)
        print("DIAGNÓSTICOS COMPLETOS")
        print("="*80)
        
        diagnostic = ContinualLearningDiagnostic(self.model, self.device)
        
        # 1. Task keys
        diagnostic.analyze_task_keys()
        
        # 2. Prompt usage
        diagnostic.analyze_prompt_usage(self.task_test_loaders, self.n_tasks - 1)
        
        # 3. Feature space
        diagnostic.analyze_feature_space(self.task_test_loaders, self.n_tasks - 1)
        
        # 4. Training state
        diagnostic.check_training_state()
        
        # Guardar
        diagnostic.save_diagnostics(Path(output_dir))
        
        return diagnostic
    
    def analyze_single_task_confusion(self, task_id):
        """Análisis detallado de confusión para una tarea específica"""
        print(f"\n{'='*80}")
        print(f"ANÁLISIS DETALLADO - TASK {task_id + 1}")
        print(f"{'='*80}")
        
        loader = self.task_test_loaders[task_id]
        
        # Recolectar predicciones
        all_features = []
        all_task_predictions = []
        all_task_logits = []
        all_similarities = []
        
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(self.device)
                
                # Get features
                bsz, n_channels, seq_len = x_batch.shape
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
                
                all_features.append(pooled.cpu())

                # FIX: Expand predicted_task to match batch size
                predicted_task = task_info['predicted_task']
                if isinstance(predicted_task, int):
                    all_task_predictions.extend([predicted_task] * bsz)
                else:
                    all_task_predictions.extend(predicted_task.cpu().numpy().tolist())

                all_task_logits.append(task_info['task_logits'].cpu())
                all_similarities.append(task_info['similarities'].cpu())

        all_features = torch.cat(all_features, dim=0)
        all_task_logits = torch.cat(all_task_logits, dim=0)
        all_similarities = torch.cat(all_similarities, dim=0)

        # Análisis estadístico
        print(f"\nSimilitud con cada Task Key (promedio sobre {all_similarities.size(0)} muestras):")
        for i in range(self.n_tasks):
            mean_sim = all_similarities[:, i].mean().item()
            std_sim = all_similarities[:, i].std().item()
            max_sim = all_similarities[:, i].max().item()
            min_sim = all_similarities[:, i].min().item()

            marker = "👉" if i == task_id else "  "
            print(f"{marker} Task {i+1}: mean={mean_sim:.4f} ± {std_sim:.4f}, "
                  f"range=[{min_sim:.4f}, {max_sim:.4f}]")

        # Probabilidades de task después de softmax
        task_probs = F.softmax(all_task_logits / self.model.task_predictor.temperature, dim=-1)

        print(f"\nProbabilidades de Task (después de softmax):")
        for i in range(self.n_tasks):
            mean_prob = task_probs[:, i].mean().item()
            std_prob = task_probs[:, i].std().item()
            marker = "👉" if i == task_id else "  "
            print(f"{marker} Task {i+1}: {mean_prob:.4f} ± {std_prob:.4f}")

        # Identificar muestras mal clasificadas
        task_preds = task_probs.argmax(dim=-1).numpy()
        misclassified_mask = task_preds != task_id
        n_misclassified = misclassified_mask.sum()

        print(f"\nMuestras mal clasificadas: {n_misclassified}/{len(task_preds)} "
              f"({100*n_misclassified/len(task_preds):.2f}%)")

        if n_misclassified > 0:
            print("\nDistribución de predicciones erróneas:")
            for i in range(self.n_tasks):
                if i != task_id:
                    count = (task_preds[misclassified_mask] == i).sum()
                    if count > 0:
                        pct = 100 * count / n_misclassified
                        print(f"  → Task {i+1}: {count} muestras ({pct:.1f}%)")

        return {
            'features': all_features,
            'similarities': all_similarities,
            'task_probs': task_probs,
            'task_preds': task_preds
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='dailysport')
    parser.add_argument('--n_tasks', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='analysis_output')
    parser.add_argument('--analyze_task', type=int, default=None,
                       help='ID de tarea específica para análisis detallado (0-indexed)')

    args = parser.parse_args()

    # Crear analizador
    analyzer = ModelAnalyzer(
        args.checkpoint,
        args.dataset,
        args.n_tasks,
        args.batch_size
    )

    # Evaluación rápida
    analyzer.quick_evaluation()

    # Diagnósticos completos
    analyzer.run_full_diagnostics(args.output_dir)

    # Análisis de tarea específica si se solicita
    if args.analyze_task is not None:
        if 0 <= args.analyze_task < args.n_tasks:
            analyzer.analyze_single_task_confusion(args.analyze_task)
        else:
            print(f"\n⚠️ Task ID inválido: {args.analyze_task}. "
                  f"Debe estar entre 0 y {args.n_tasks-1}")
    else:
        # Analizar todas las tareas
        print("\n" + "="*80)
        print("ANÁLISIS POR TAREA")
        print("="*80)
        for task_id in range(args.n_tasks):
            analyzer.analyze_single_task_confusion(task_id)

    print("\n✅ Análisis completado!")


if __name__ == "__main__":
    main()