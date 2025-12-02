"""
Visualizador de Diagnósticos

Crea visualizaciones para entender los problemas del modelo

Uso:
    python visualize_diagnostics.py --diagnostics_dir continual_results_debug/diagnostics_task_3
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def plot_task_key_similarity(similarity_matrix, output_path):
    """Visualiza la matriz de similitud entre task keys"""
    n_tasks = similarity_matrix.shape[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Crear heatmap
    im = ax.imshow(similarity_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
    
    # Labels
    ax.set_xticks(np.arange(n_tasks))
    ax.set_yticks(np.arange(n_tasks))
    ax.set_xticklabels([f'Task {i+1}' for i in range(n_tasks)])
    ax.set_yticklabels([f'Task {i+1}' for i in range(n_tasks)])
    
    # Rotar labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Agregar valores en cada celda
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i == j:
                text = ax.text(j, i, '1.00', ha="center", va="center", color="black", fontweight='bold')
            else:
                value = similarity_matrix[i, j]
                color = "white" if value > 0.7 else "black"
                text = ax.text(j, i, f'{value:.2f}', ha="center", va="center", color=color)
    
    ax.set_title('Task Key Similarity Matrix\n(Rojo = Muy Similar = Problema)', fontsize=12, pad=20)
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    
    # Línea de umbral problemático
    ax.axhline(y=-0.5, color='red', linestyle='--', linewidth=2, alpha=0.3)
    ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=2, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Task key similarity guardado: {output_path}")
    plt.close()


def plot_prompt_overlap(overlap_matrix, output_path):
    """Visualiza el solapamiento de prompts entre tareas"""
    n_tasks = overlap_matrix.shape[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(overlap_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(np.arange(n_tasks))
    ax.set_yticks(np.arange(n_tasks))
    ax.set_xticklabels([f'Task {i+1}' for i in range(n_tasks)])
    ax.set_yticklabels([f'Task {i+1}' for i in range(n_tasks)])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(n_tasks):
        for j in range(n_tasks):
            value = overlap_matrix[i, j]
            color = "white" if value > 0.5 else "black"
            text = ax.text(j, i, f'{value:.2f}', ha="center", va="center", color=color)
    
    ax.set_title('Prompt Overlap Matrix (Jaccard Similarity)\n(Alto = Tareas Usan Mismos Prompts)', 
                 fontsize=12, pad=20)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Jaccard Similarity', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Prompt overlap guardado: {output_path}")
    plt.close()


def plot_confusion_matrix(confusion_matrix, output_path):
    """Visualiza la matriz de confusión de predicción de tareas"""
    n_tasks = confusion_matrix.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(confusion_matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(np.arange(n_tasks))
    ax.set_yticks(np.arange(n_tasks))
    ax.set_xticklabels([f'Pred {i+1}' for i in range(n_tasks)])
    ax.set_yticklabels([f'True {i+1}' for i in range(n_tasks)])
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Agregar porcentajes
    for i in range(n_tasks):
        for j in range(n_tasks):
            value = confusion_matrix[i, j]
            color = "white" if value > 0.5 else "black"
            if i == j:
                text = ax.text(j, i, f'{value:.1%}', ha="center", va="center", 
                             color=color, fontweight='bold', fontsize=11)
            else:
                text = ax.text(j, i, f'{value:.1%}', ha="center", va="center", color=color)
    
    ax.set_xlabel('Predicted Task', fontsize=12)
    ax.set_ylabel('True Task', fontsize=12)
    ax.set_title('Task Prediction Confusion Matrix\n(Diagonal = Correcto, Off-Diagonal = Error)', 
                 fontsize=12, pad=20)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=20)
    
    # Resaltar diagonal
    for i in range(n_tasks):
        rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, 
                            edgecolor='red', linewidth=3)
        ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix guardada: {output_path}")
    plt.close()


def plot_feature_tsne(features, task_ids, predictions, output_path):
    """Visualiza features con t-SNE"""
    from sklearn.manifold import TSNE
    
    print("Calculando t-SNE (esto puede tomar unos minutos)...")
    
    # Limitar número de muestras para velocidad
    max_samples = 2000
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        task_ids = task_ids[indices]
        predictions = predictions[indices]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Ground truth
    n_tasks = len(np.unique(task_ids))
    colors = plt.cm.tab10(np.linspace(0, 1, n_tasks))
    
    for task_id in np.unique(task_ids):
        mask = task_ids == task_id
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[task_id]], label=f'Task {task_id+1}',
                   alpha=0.6, s=20)
    
    ax1.set_title('Feature Space - Ground Truth', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predictions (con errores destacados)
    for task_id in np.unique(task_ids):
        # Correctos
        mask_correct = (task_ids == task_id) & (predictions == task_id)
        ax2.scatter(features_2d[mask_correct, 0], features_2d[mask_correct, 1],
                   c=[colors[task_id]], label=f'Task {task_id+1} (correct)',
                   alpha=0.6, s=20, marker='o')
        
        # Incorrectos
        mask_incorrect = (task_ids == task_id) & (predictions != task_id)
        if mask_incorrect.any():
            ax2.scatter(features_2d[mask_incorrect, 0], features_2d[mask_incorrect, 1],
                       c=[colors[task_id]], label=f'Task {task_id+1} (error)',
                       alpha=0.9, s=50, marker='x', linewidths=2)
    
    ax2.set_title('Feature Space - Predictions (X = Errors)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ t-SNE plot guardado: {output_path}")
    plt.close()


def plot_prompt_usage(prompt_usage_dict, output_path):
    """Visualiza el uso de prompts por tarea"""
    n_tasks = len(prompt_usage_dict)
    pool_size = len(list(prompt_usage_dict.values())[0])
    
    fig, axes = plt.subplots(n_tasks, 1, figsize=(12, 3*n_tasks))
    if n_tasks == 1:
        axes = [axes]
    
    for task_id, ax in enumerate(axes):
        if task_id not in prompt_usage_dict:
            continue
            
        counts = prompt_usage_dict[task_id]
        
        # Bar plot
        bars = ax.bar(range(pool_size), counts, color='steelblue', alpha=0.7)
        
        # Destacar top-5 prompts
        top_indices = np.argsort(counts)[-5:][::-1]
        for idx in top_indices:
            bars[idx].set_color('darkred')
            bars[idx].set_alpha(0.8)
        
        ax.set_xlabel('Prompt ID')
        ax.set_ylabel('Usage Count')
        ax.set_title(f'Task {task_id+1}: Prompt Usage Distribution\n(Rojo = Top-5 más usados)', 
                    fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Agregar línea de promedio
        mean_usage = counts.mean()
        ax.axhline(y=mean_usage, color='green', linestyle='--', 
                  label=f'Mean: {mean_usage:.0f}', linewidth=2)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Prompt usage guardado: {output_path}")
    plt.close()


def create_summary_report(diagnostics_path, output_path):
    """Crea un resumen visual de todos los diagnósticos"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Cargar diagnósticos
    data = np.load(diagnostics_path)
    
    # 1. Task Key Similarity
    ax1 = fig.add_subplot(gs[0, 0])
    similarity = data['task_key_similarity']
    im1 = ax1.imshow(similarity, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax1.set_title('Task Key Similarity', fontweight='bold')
    n_tasks = similarity.shape[0]
    ax1.set_xticks(range(n_tasks))
    ax1.set_yticks(range(n_tasks))
    ax1.set_xticklabels([f'T{i+1}' for i in range(n_tasks)])
    ax1.set_yticklabels([f'T{i+1}' for i in range(n_tasks)])
    plt.colorbar(im1, ax=ax1)
    
    # 2. Prompt Overlap
    ax2 = fig.add_subplot(gs[0, 1])
    overlap = data['prompt_overlap']
    im2 = ax2.imshow(overlap, cmap='YlOrRd', vmin=0, vmax=1)
    ax2.set_title('Prompt Overlap', fontweight='bold')
    ax2.set_xticks(range(n_tasks))
    ax2.set_yticks(range(n_tasks))
    ax2.set_xticklabels([f'T{i+1}' for i in range(n_tasks)])
    ax2.set_yticklabels([f'T{i+1}' for i in range(n_tasks)])
    plt.colorbar(im2, ax=ax2)
    
    # 3. Confusion Matrix
    ax3 = fig.add_subplot(gs[1, :])
    confusion = data['confusion_matrix']
    im3 = ax3.imshow(confusion, cmap='Blues', vmin=0, vmax=1)
    ax3.set_title('Task Prediction Confusion Matrix', fontweight='bold', fontsize=14)
    ax3.set_xticks(range(n_tasks))
    ax3.set_yticks(range(n_tasks))
    ax3.set_xticklabels([f'Pred {i+1}' for i in range(n_tasks)])
    ax3.set_yticklabels([f'True {i+1}' for i in range(n_tasks)])
    ax3.set_xlabel('Predicted Task')
    ax3.set_ylabel('True Task')
    
    # Agregar valores
    for i in range(n_tasks):
        for j in range(n_tasks):
            value = confusion[i, j]
            color = "white" if value > 0.5 else "black"
            ax3.text(j, i, f'{value:.1%}', ha="center", va="center", color=color)
    
    plt.colorbar(im3, ax=ax3)
    
    # 4. Diagnóstico textual
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Análisis
    off_diag_sim = similarity[~np.eye(n_tasks, dtype=bool)]
    max_sim = off_diag_sim.max()
    mean_sim = off_diag_sim.mean()
    
    off_diag_overlap = overlap[~np.eye(n_tasks, dtype=bool)]
    max_overlap = off_diag_overlap.max()
    mean_overlap = off_diag_overlap.mean()
    
    diag_conf = np.diag(confusion)
    min_task_acc = diag_conf.min()
    mean_task_acc = diag_conf.mean()
    
    # Determinar estado
    issues = []
    if max_sim > 0.7:
        issues.append(f"⚠️  Task keys muy similares (max={max_sim:.3f})")
    if max_overlap > 0.5:
        issues.append(f"⚠️  Alto solapamiento de prompts (max={max_overlap:.3f})")
    if min_task_acc < 0.8:
        issues.append(f"⚠️  Baja task prediction accuracy (min={min_task_acc:.1%})")
    
    if not issues:
        status_text = "✅ MODELO SALUDABLE\n\nNo se detectaron problemas graves."
        status_color = 'green'
    else:
        status_text = "🔴 PROBLEMAS DETECTADOS:\n\n" + "\n".join(issues)
        status_color = 'red'
    
    summary_text = f"""
{status_text}

ESTADÍSTICAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Task Key Similarity (off-diagonal):
  • Media: {mean_sim:.3f}
  • Máxima: {max_sim:.3f}
  • Umbral problemático: > 0.7

Prompt Overlap (Jaccard):
  • Media: {mean_overlap:.3f}
  • Máxima: {max_overlap:.3f}
  • Umbral problemático: > 0.5

Task Prediction Accuracy:
  • Media: {mean_task_acc:.1%}
  • Mínima: {min_task_acc:.1%}
  • Umbral saludable: > 95%
"""
    
    ax4.text(0.5, 0.5, summary_text, 
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment='center',
            horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Diagnóstico Completo del Modelo', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Resumen guardado: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics_dir', type=str, required=True,
                       help='Directorio con diagnósticos (.npz file)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directorio de salida para plots (default: mismo que diagnostics_dir)')
    
    args = parser.parse_args()
    
    diagnostics_dir = Path(args.diagnostics_dir)
    output_dir = Path(args.output_dir) if args.output_dir else diagnostics_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Cargar diagnósticos
    diag_file = diagnostics_dir / 'diagnostics.npz'
    if not diag_file.exists():
        print(f"❌ No se encontró {diag_file}")
        return
    
    print(f"Cargando diagnósticos desde: {diag_file}")
    data = np.load(diag_file, allow_pickle=True)
    
    # Crear visualizaciones
    print("\nGenerando visualizaciones...")
    
    # 1. Task key similarity
    if 'task_key_similarity' in data:
        plot_task_key_similarity(
            data['task_key_similarity'],
            output_dir / 'task_key_similarity.png'
        )
    
    # 2. Prompt overlap
    if 'prompt_overlap' in data:
        plot_prompt_overlap(
            data['prompt_overlap'],
            output_dir / 'prompt_overlap.png'
        )
    
    # 3. Confusion matrix
    if 'confusion_matrix' in data:
        plot_confusion_matrix(
            data['confusion_matrix'],
            output_dir / 'confusion_matrix.png'
        )
    
    # 4. Feature t-SNE
    if all(k in data for k in ['features', 'task_ids', 'predictions']):
        plot_feature_tsne(
            data['features'],
            data['task_ids'],
            data['predictions'],
            output_dir / 'feature_tsne.png'
        )
    
    # 5. Resumen
    create_summary_report(diag_file, output_dir / 'summary_report.png')
    
    print(f"\n✅ Todas las visualizaciones guardadas en: {output_dir}")
    print("\nArchivos generados:")
    print("  • task_key_similarity.png")
    print("  • prompt_overlap.png")
    print("  • confusion_matrix.png")
    print("  • feature_tsne.png")
    print("  • summary_report.png")


if __name__ == "__main__":
    main()
