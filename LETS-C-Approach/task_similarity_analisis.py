"""
Análisis Profundo de Tareas Similares

Diagnóstico especializado para entender por qué Task 2 y Task 3 tienen alta similaridad
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
import argparse

from main import prepare_dataset
from textlets_multihead import PromptedLETS
from utils import set_seed, get_device


class DeepTaskAnalyzer:
    """Análisis profundo de similaridad entre tareas específicas"""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def analyze_task_pair(self, x1, y1, task1_id, x2, y2, task2_id, batch_size=64):
        """
        Análisis detallado de un par de tareas

        Returns:
            dict con múltiples métricas
        """
        print(f"\n{'=' * 80}")
        print(f"ANÁLISIS DETALLADO: Task {task1_id} vs Task {task2_id}")
        print(f"{'=' * 80}\n")

        # 1. Extraer embeddings base (sin prompts)
        emb1_base = self._extract_embeddings(x1, task1_id, use_prompts=False, batch_size=batch_size)
        emb2_base = self._extract_embeddings(x2, task2_id, use_prompts=False, batch_size=batch_size)

        # 2. Extraer embeddings con prompts
        emb1_prompt = self._extract_embeddings(x1, task1_id, use_prompts=True, batch_size=batch_size)
        emb2_prompt = self._extract_embeddings(x2, task2_id, use_prompts=True, batch_size=batch_size)

        # 3. Extraer prompts seleccionados
        prompts1 = self._get_selected_prompts(x1, task1_id, batch_size=batch_size)
        prompts2 = self._get_selected_prompts(x2, task2_id, batch_size=batch_size)

        # 4. Calcular métricas
        metrics = {
            # Similaridad base
            'base_similarity': self._compute_cross_similarity(emb1_base, emb2_base),
            'base_intra1': self._compute_intra_similarity(emb1_base),
            'base_intra2': self._compute_intra_similarity(emb2_base),

            # Similaridad con prompts
            'prompt_similarity': self._compute_cross_similarity(emb1_prompt, emb2_prompt),
            'prompt_intra1': self._compute_intra_similarity(emb1_prompt),
            'prompt_intra2': self._compute_intra_similarity(emb2_prompt),

            # Efecto de los prompts
            'prompt_effect1': self._compute_prompt_effect(emb1_base, emb1_prompt),
            'prompt_effect2': self._compute_prompt_effect(emb2_base, emb2_prompt),

            # Análisis de prompts
            'prompt_overlap': self._compute_prompt_overlap(prompts1, prompts2),
            'g_prompt_similarity': self._compute_g_prompt_similarity(task1_id, task2_id),
        }

        # 5. Análisis de confusión
        confusion_metrics = self._analyze_confusion(
            x1, y1, task1_id, x2, y2, task2_id, batch_size=batch_size
        )
        metrics.update(confusion_metrics)

        return metrics

    def _extract_embeddings(self, x, task_id, use_prompts=True, batch_size=64):
        """Extrae embeddings de un dataset"""
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings = []

        for (x_batch,) in loader:
            x_batch = x_batch.to(self.device)

            # Base embeddings
            base_feats = self.model.text_encoder(x_batch)

            if use_prompts:
                # Con prompts
                selected_prompts = self.model.coda_prompt(base_feats, task_id=task_id)
                embeddings = base_feats + selected_prompts.mean(dim=1)
            else:
                embeddings = base_feats

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _get_selected_prompts(self, x, task_id, batch_size=64):
        """Obtiene los índices de E-Prompts seleccionados"""
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_indices = []

        for (x_batch,) in loader:
            x_batch = x_batch.to(self.device)
            base_feats = self.model.text_encoder(x_batch)

            _, selection_info = self.model.coda_prompt(
                base_feats, task_id=task_id, return_selection_info=True
            )

            if selection_info['e_prompt_indices'] is not None:
                all_indices.append(selection_info['e_prompt_indices'].cpu())

        if all_indices:
            return torch.cat(all_indices, dim=0)
        return None

    def _compute_cross_similarity(self, emb1, emb2):
        """Similaridad promedio entre dos conjuntos de embeddings"""
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)

        sim_matrix = emb1_norm @ emb2_norm.T
        return sim_matrix.mean().item()

    def _compute_intra_similarity(self, emb):
        """Similaridad intra-task"""
        emb_norm = F.normalize(emb, p=2, dim=1)
        sim_matrix = emb_norm @ emb_norm.T

        # Excluir diagonal
        mask = ~torch.eye(len(emb), dtype=bool)
        return sim_matrix[mask].mean().item()

    def _compute_prompt_effect(self, emb_base, emb_prompt):
        """Mide cuánto cambian los embeddings por los prompts"""
        diff = emb_prompt - emb_base
        return diff.norm(dim=1).mean().item()

    def _compute_prompt_overlap(self, prompts1, prompts2):
        """Calcula overlap de E-Prompts seleccionados"""
        if prompts1 is None or prompts2 is None:
            return 0.0

        # Contar frecuencia de cada prompt
        unique1, counts1 = prompts1.flatten().unique(return_counts=True)
        unique2, counts2 = prompts2.flatten().unique(return_counts=True)

        # Calcular Jaccard similarity
        set1 = set(unique1.tolist())
        set2 = set(unique2.tolist())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _compute_g_prompt_similarity(self, task1_id, task2_id):
        """Similaridad entre G-Prompts de dos tareas"""
        if not self.model.coda_prompt.use_g_prompt:
            return 0.0

        g1 = self.model.coda_prompt.g_prompts[task1_id]
        g2 = self.model.coda_prompt.g_prompts[task2_id]

        # Flatten y normalizar
        g1_flat = F.normalize(g1.flatten().unsqueeze(0), p=2, dim=1)
        g2_flat = F.normalize(g2.flatten().unsqueeze(0), p=2, dim=1)

        return (g1_flat @ g2_flat.T).item()

    def _analyze_confusion(self, x1, y1, task1_id, x2, y2, task2_id, batch_size=64):
        """Analiza confusión de predicciones entre tareas"""

        # Predecir task1 samples
        dataset1 = TensorDataset(x1, y1)
        loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False)

        task1_predictions = []
        task1_correct = 0
        task1_total = 0

        for x_batch, y_batch in loader1:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Sin oracle (soft prediction)
            logits, task_info = self.model(x_batch, task_id=None, return_task_info=True)
            predicted_task = task_info['predicted_task']

            task1_predictions.extend([predicted_task] * len(x_batch))

            # Verificar si predice correctamente
            if predicted_task == task1_id:
                task1_correct += len(x_batch)
            task1_total += len(x_batch)

        # Predecir task2 samples
        dataset2 = TensorDataset(x2, y2)
        loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False)

        task2_predictions = []
        task2_correct = 0
        task2_total = 0
        task2_confused_with_task1 = 0

        for x_batch, y_batch in loader2:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits, task_info = self.model(x_batch, task_id=None, return_task_info=True)
            predicted_task = task_info['predicted_task']

            task2_predictions.extend([predicted_task] * len(x_batch))

            if predicted_task == task2_id:
                task2_correct += len(x_batch)
            elif predicted_task == task1_id:
                task2_confused_with_task1 += len(x_batch)

            task2_total += len(x_batch)

        return {
            'task1_prediction_accuracy': task1_correct / task1_total,
            'task2_prediction_accuracy': task2_correct / task2_total,
            'task2_confused_with_task1_rate': task2_confused_with_task1 / task2_total,
        }

    def plot_comparison(self, metrics, save_path=None):
        """Visualiza métricas de comparación"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Similaridad Base vs Prompted
        ax = axes[0, 0]
        categories = ['Intra-Task 1', 'Intra-Task 2', 'Inter-Task']
        base_vals = [metrics['base_intra1'], metrics['base_intra2'], metrics['base_similarity']]
        prompt_vals = [metrics['prompt_intra1'], metrics['prompt_intra2'], metrics['prompt_similarity']]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width / 2, base_vals, width, label='Sin Prompts', alpha=0.7, color='coral')
        ax.bar(x + width / 2, prompt_vals, width, label='Con Prompts', alpha=0.7, color='steelblue')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Efecto de Prompts en Similaridad')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=metrics['prompt_similarity'], color='red', linestyle='--', alpha=0.5, label='Inter-task target')

        # 2. Efecto de Prompts (magnitud del cambio)
        ax = axes[0, 1]
        effects = [metrics['prompt_effect1'], metrics['prompt_effect2']]
        ax.bar(['Task 1', 'Task 2'], effects, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('L2 Norm of Change')
        ax.set_title('Magnitud del Efecto de Prompts')
        ax.grid(axis='y', alpha=0.3)

        # 3. Prompt Overlap
        ax = axes[0, 2]
        overlap_data = [
            metrics['prompt_overlap'],
            metrics['g_prompt_similarity']
        ]
        colors = ['green' if x < 0.5 else 'red' for x in overlap_data]
        ax.bar(['E-Prompt\nOverlap', 'G-Prompt\nSimilarity'], overlap_data,
               color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Similarity/Overlap')
        ax.set_title('Overlap de Prompts')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Threshold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 4. Task Prediction Accuracy
        ax = axes[1, 0]
        pred_accs = [
            metrics['task1_prediction_accuracy'] * 100,
            metrics['task2_prediction_accuracy'] * 100
        ]
        colors = ['green' if x > 80 else 'orange' if x > 60 else 'red' for x in pred_accs]
        ax.bar(['Task 1', 'Task 2'], pred_accs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Task Prediction Accuracy')
        ax.axhline(y=80, color='green', linestyle='--', alpha=0.3, label='Good threshold')
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 5. Confusion Rate
        ax = axes[1, 1]
        confusion_rate = metrics['task2_confused_with_task1_rate'] * 100
        color = 'red' if confusion_rate > 20 else 'orange' if confusion_rate > 10 else 'green'
        ax.bar(['Task 2 → Task 1'], [confusion_rate], color=color, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Confusion Rate (%)')
        ax.set_title('Task 2 Confused as Task 1')
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.3, label='Warning')
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.3, label='Critical')
        ax.set_ylim([0, 100])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 6. Resumen de métricas clave
        ax = axes[1, 2]
        ax.axis('off')

        # Crear tabla de resumen
        summary_text = f"""
        RESUMEN DE DIAGNÓSTICO
        {'=' * 35}

        Similaridad Inter-Task:
        • Sin prompts: {metrics['base_similarity']:.3f}
        • Con prompts: {metrics['prompt_similarity']:.3f}
        • Cambio: {metrics['prompt_similarity'] - metrics['base_similarity']:+.3f}

        Separación (Intra - Inter):
        • Task 1: {metrics['prompt_intra1'] - metrics['prompt_similarity']:.3f}
        • Task 2: {metrics['prompt_intra2'] - metrics['prompt_similarity']:.3f}

        Overlap de Prompts:
        • E-Prompts: {metrics['prompt_overlap'] * 100:.1f}%
        • G-Prompts: {metrics['g_prompt_similarity']:.3f}

        Task Prediction:
        • Task 1: {metrics['task1_prediction_accuracy'] * 100:.1f}%
        • Task 2: {metrics['task2_prediction_accuracy'] * 100:.1f}%
        • Confusión T2→T1: {metrics['task2_confused_with_task1_rate'] * 100:.1f}%

        DIAGNÓSTICO:
        """

        # Añadir diagnóstico
        if metrics['prompt_similarity'] > 0.85:
            diagnosis = "⚠️ ALTA SIMILARIDAD CRÍTICA"
        elif metrics['prompt_similarity'] > 0.75:
            diagnosis = "⚠️ Similaridad moderada-alta"
        else:
            diagnosis = "✓ Separación aceptable"

        summary_text += diagnosis

        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Análisis guardado: {save_path}")

        plt.show()

    def recommend_solutions(self, metrics):
        """Recomienda soluciones basadas en el análisis"""
        print("\n" + "=" * 80)
        print("RECOMENDACIONES")
        print("=" * 80 + "\n")

        problems = []
        solutions = []

        # 1. Alta similaridad base
        if metrics['base_similarity'] > 0.80:
            problems.append("🔴 Alta similaridad en embeddings BASE (>0.80)")
            solutions.append("""
            → Las clases son inherentemente similares en el dataset
            → Soluciones:
              • Aumentar complejidad del text encoder (modelo más grande)
              • Añadir data augmentation específico para estas clases
              • Considerar re-balanceo de clases entre tareas
              • Usar feature engineering adicional (estadísticos, frecuency domain)
            """)

        # 2. Prompts no ayudan suficiente
        if metrics['prompt_similarity'] > 0.85 and (metrics['prompt_similarity'] - metrics['base_similarity']) > -0.05:
            problems.append("🔴 Los prompts NO reducen suficientemente la similaridad")
            solutions.append("""
            → CODA-Prompt no está diferenciando las tareas
            → Soluciones:
              • AUMENTAR pool_size de E-Prompts (de 20 a 50+)
              • AUMENTAR prompt_length (de 5 a 10-15)
              • REDUCIR top_k para forzar mayor especialización
              • Añadir regularización ortogonal más fuerte en G-Prompts
              • Entrenar prompts con learning rate MÁS ALTO
            """)

        # 3. Alto overlap de E-Prompts
        if metrics['prompt_overlap'] > 0.6:
            problems.append("🟡 Alto overlap de E-Prompts entre tareas")
            solutions.append("""
            → Las tareas usan prompts muy similares
            → Soluciones:
              • Añadir diversity loss para E-Prompts
              • Implementar prompt specialization loss
              • Usar diferentes subset de E-Prompts por tarea (partición)
            """)

        # 4. G-Prompts demasiado similares
        if metrics['g_prompt_similarity'] > 0.7:
            problems.append("🟡 G-Prompts muy similares entre tareas")
            solutions.append("""
            → Los task-specific prompts no se diferencian
            → Soluciones:
              • Añadir orthogonal loss entre G-Prompts
              • Aumentar learning rate de G-Prompts
              • Inicializar G-Prompts con mayor diversidad
            """)

        # 5. Mala predicción de tareas
        if metrics['task2_prediction_accuracy'] < 0.7:
            problems.append("🔴 Task Predictor confunde las tareas")
            solutions.append("""
            → El task predictor no discrimina bien
            → Soluciones:
              • AUMENTAR task_loss_weight (a 2.0-3.0)
              • Usar contrastive loss para task keys
              • Entrenar task predictor por más epochs
              • Añadir task-adversarial training
            """)

        # 6. Alta tasa de confusión
        if metrics['task2_confused_with_task1_rate'] > 0.15:
            problems.append("🔴 Alta tasa de confusión T2→T1")
            solutions.append("""
            → Task 2 frecuentemente se confunde con Task 1
            → Soluciones:
              • Entrenar Task 2 por MÁS epochs
              • Usar hard negative mining (samples difíciles)
              • Implementar focal loss para casos difíciles
              • Considerar re-ordenar las tareas en el curriculum
            """)

        # Print resultados
        if not problems:
            print("✅ No se detectaron problemas críticos")
            print("   Las tareas están razonablemente bien separadas")
        else:
            print(f"Se detectaron {len(problems)} problemas:\n")
            for i, (prob, sol) in enumerate(zip(problems, solutions), 1):
                print(f"{i}. {prob}")
                print(sol)
                print()


def main():
    parser = argparse.ArgumentParser()

    # Data & Model
    parser.add_argument('--dataset', type=str, default='dailysport')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_tasks', type=int, default=6)
    parser.add_argument('--classes_per_task', type=int, default=3)

    # Tareas a analizar
    parser.add_argument('--task1', type=int, default=2, help='Primera tarea a comparar')
    parser.add_argument('--task2', type=int, default=3, help='Segunda tarea a comparar')

    # Output
    parser.add_argument('--output_dir', type=str, default='task_analysis')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_test', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Crear output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Cargar datos
    print("Cargando datos...")
    base_path = Path('../data') / args.dataset

    if args.use_test:
        x_data, y_data = prepare_dataset(base_path / 'x_test.pkl', base_path / 'state_test.pkl')
        split_name = 'test'
    else:
        x_data, y_data = prepare_dataset(base_path / 'x_train.pkl', base_path / 'state_train.pkl')
        split_name = 'train'

    # Filtrar por tareas
    def get_task_data(task_id):
        start_class = task_id * args.classes_per_task
        end_class = (task_id + 1) * args.classes_per_task
        task_classes = list(range(start_class, end_class))
        mask = torch.isin(y_data, torch.tensor(task_classes))
        return x_data[mask], y_data[mask]

    x1, y1 = get_task_data(args.task1)
    x2, y2 = get_task_data(args.task2)

    print(f"Task {args.task1}: {len(x1)} samples")
    print(f"Task {args.task2}: {len(x2)} samples")

    # Cargar modelo
    print("\nCargando modelo...")
    from similarity_analisis import load_model_from_checkpoint
    model = load_model_from_checkpoint(args.checkpoint, args.n_tasks, args.classes_per_task)

    # Crear analyzer
    analyzer = DeepTaskAnalyzer(model, device=device)

    # Analizar
    metrics = analyzer.analyze_task_pair(
        x1, y1, args.task1,
        x2, y2, args.task2,
        batch_size=args.batch_size
    )

    # Visualizar
    save_path = output_dir / f'task_{args.task1}_vs_{args.task2}_analysis_{split_name}.png'
    analyzer.plot_comparison(metrics, save_path=save_path)

    # Recomendaciones
    analyzer.recommend_solutions(metrics)

    print(f"\n✓ Análisis completado. Resultados en: {output_dir}")


if __name__ == "__main__":
    main()