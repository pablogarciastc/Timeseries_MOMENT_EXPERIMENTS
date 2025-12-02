"""
Análisis de Similaridad de Embeddings entre Tareas

Herramientas para:
- Extraer embeddings de diferentes tareas
- Calcular matrices de similaridad intra/inter-task
- Visualizar separación de tareas en espacio latente
- Analizar drift de embeddings durante continual learning
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
from tqdm import tqdm
import os

from main import prepare_dataset
from textlets_multihead import PromptedLETS
from utils import set_seed, get_device


class EmbeddingAnalyzer:
    """
    Analiza la similaridad de embeddings entre tareas en continual learning
    """

    def __init__(self, model, device='cuda', batch_size=32):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size

        # Storage para embeddings
        self.embeddings = {}  # {task_id: embeddings}
        self.labels = {}  # {task_id: labels}
        self.prompted_embeddings = {}  # embeddings con prompts

    @torch.no_grad()
    def extract_embeddings(self, x, y, task_id, use_prompts=True):
        """
        Extrae embeddings de un conjunto de datos

        Args:
            x: datos de entrada [N, channels, seq_len]
            y: etiquetas [N]
            task_id: ID de la tarea
            use_prompts: si usar prompts o embeddings base

        Returns:
            embeddings: [N, d_model]
            labels: [N]
        """
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_embeddings = []
        all_labels = []

        for x_batch, y_batch in tqdm(loader, desc=f"Extracting Task {task_id}"):
            x_batch = x_batch.to(self.device)

            # Base embeddings (e_fused de LLaMA)
            base_feats = self.model.text_encoder(x_batch)

            # Validar dimensionalidad
            if base_feats.dim() != 2:
                raise ValueError(f"Expected 2D embeddings, got shape {base_feats.shape}")

            if use_prompts:
                # Con prompts de CODA
                selected_prompts = self.model.coda_prompt(base_feats, task_id=task_id)
                embeddings = base_feats + selected_prompts.mean(dim=1)
            else:
                embeddings = base_feats

            all_embeddings.append(embeddings.cpu())
            all_labels.append(y_batch.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)

        return embeddings, labels

    def extract_all_tasks(self, task_data_dict, use_prompts=True):
        """
        Extrae embeddings para todas las tareas

        Args:
            task_data_dict: {task_id: (x, y)}
            use_prompts: usar prompts o no
        """
        print(f"\n{'=' * 80}")
        print(f"Extrayendo embeddings ({'CON' if use_prompts else 'SIN'} prompts)")
        print(f"{'=' * 80}\n")

        storage = self.prompted_embeddings if use_prompts else self.embeddings

        for task_id, (x, y) in task_data_dict.items():
            emb, lab = self.extract_embeddings(x, y, task_id, use_prompts)
            storage[task_id] = emb
            self.labels[task_id] = lab

            print(f"Task {task_id}: {emb.shape}, Labels: {lab.unique().tolist()}")

    def compute_similarity_matrix(self, embeddings_dict, metric='cosine'):
        """
        Calcula matriz de similaridad entre todas las muestras de todas las tareas

        Args:
            embeddings_dict: {task_id: embeddings}
            metric: 'cosine' o 'euclidean'

        Returns:
            similarity_matrix: [total_samples, total_samples]
            task_boundaries: índices de límites entre tareas
        """
        # Concatenar todos los embeddings
        all_embeddings = []
        task_boundaries = [0]

        for task_id in sorted(embeddings_dict.keys()):
            emb = embeddings_dict[task_id]
            all_embeddings.append(emb)
            task_boundaries.append(task_boundaries[-1] + len(emb))

        all_embeddings = torch.cat(all_embeddings, dim=0)  # [N_total, d_model]

        if metric == 'cosine':
            # Normalizar para cosine similarity
            all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)
            similarity = all_embeddings_norm @ all_embeddings_norm.T
        elif metric == 'euclidean':
            # Distancia euclidiana
            dists = torch.cdist(all_embeddings, all_embeddings, p=2)
            # Convertir a similaridad (1 / (1 + dist))
            similarity = 1.0 / (1.0 + dists)
        else:
            raise ValueError(f"Metric {metric} no soportado")

        return similarity.numpy(), task_boundaries

    def compute_intra_inter_similarity(self, embeddings_dict):
        """
        Calcula similaridad intra-task vs inter-task

        Returns:
            intra_sim: dict {task_id: mean_similarity}
            inter_sim: dict {(task_i, task_j): mean_similarity}
        """
        print("\nCalculando similaridades intra/inter-task...")

        intra_sim = {}
        inter_sim = {}

        task_ids = sorted(embeddings_dict.keys())

        for task_id in task_ids:
            emb = embeddings_dict[task_id]
            emb_norm = F.normalize(emb, p=2, dim=1)

            # Intra-task: similaridad promedio dentro de la tarea
            sim_matrix = emb_norm @ emb_norm.T
            # Excluir diagonal (similaridad consigo mismo = 1)
            mask = ~torch.eye(len(emb), dtype=bool)
            intra_sim[task_id] = sim_matrix[mask].mean().item()

        # Inter-task: similaridad entre tareas diferentes
        for i, task_i in enumerate(task_ids):
            for task_j in task_ids[i + 1:]:
                emb_i = F.normalize(embeddings_dict[task_i], p=2, dim=1)
                emb_j = F.normalize(embeddings_dict[task_j], p=2, dim=1)

                # Similaridad promedio entre todas las muestras de task_i y task_j
                sim_matrix = emb_i @ emb_j.T
                inter_sim[(task_i, task_j)] = sim_matrix.mean().item()

        return intra_sim, inter_sim

    def plot_similarity_matrix(self, embeddings_dict, save_path=None, title=""):
        """
        Visualiza matriz de similaridad completa
        """
        sim_matrix, task_boundaries = self.compute_similarity_matrix(embeddings_dict)

        plt.figure(figsize=(12, 10))

        # Heatmap
        sns.heatmap(sim_matrix, cmap='viridis', square=True,
                    cbar_kws={'label': 'Cosine Similarity'})

        # Marcar límites entre tareas
        for boundary in task_boundaries[1:-1]:
            plt.axhline(boundary, color='red', linewidth=2, alpha=0.7)
            plt.axvline(boundary, color='red', linewidth=2, alpha=0.7)

        # Etiquetas de tareas
        n_tasks = len(embeddings_dict)
        tick_positions = [(task_boundaries[i] + task_boundaries[i + 1]) / 2
                          for i in range(n_tasks)]
        tick_labels = [f"Task {i}" for i in range(n_tasks)]

        plt.xticks(tick_positions, tick_labels, rotation=45)
        plt.yticks(tick_positions, tick_labels, rotation=0)

        plt.title(f"Similarity Matrix: {title}", fontsize=14, fontweight='bold')
        plt.xlabel("Samples (grouped by task)")
        plt.ylabel("Samples (grouped by task)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Similarity matrix guardada: {save_path}")

        plt.show()

    def plot_intra_inter_comparison(self, embeddings_dict, save_path=None):
        """
        Compara similaridad intra-task vs inter-task
        """
        intra_sim, inter_sim = self.compute_intra_inter_similarity(embeddings_dict)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot 1: Intra-task similarity
        tasks = sorted(intra_sim.keys())
        intra_values = [intra_sim[t] for t in tasks]

        ax1.bar(tasks, intra_values, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Task ID', fontsize=12)
        ax1.set_ylabel('Mean Cosine Similarity', fontsize=12)
        ax1.set_title('Intra-Task Similarity\n(Higher = más cohesión)', fontsize=13, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_xticks(tasks)

        # Plot 2: Inter-task similarity (matriz)
        n_tasks = len(tasks)
        inter_matrix = np.zeros((n_tasks, n_tasks))

        # Llenar diagonal con intra-task
        for i, t in enumerate(tasks):
            inter_matrix[i, i] = intra_sim[t]

        # Llenar off-diagonal con inter-task
        for (ti, tj), sim in inter_sim.items():
            i, j = tasks.index(ti), tasks.index(tj)
            inter_matrix[i, j] = sim
            inter_matrix[j, i] = sim

        sns.heatmap(inter_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                    square=True, ax=ax2, cbar_kws={'label': 'Similarity'},
                    xticklabels=[f"T{i}" for i in tasks],
                    yticklabels=[f"T{i}" for i in tasks])
        ax2.set_title('Task Similarity Matrix\n(Lower off-diagonal = mejor separación)',
                      fontsize=13, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Intra/Inter comparison guardada: {save_path}")

        plt.show()

        # Print estadísticas
        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DE SIMILARIDAD")
        print("=" * 60)
        print(f"\nIntra-task similarity (promedio): {np.mean(intra_values):.4f}")
        print(f"Inter-task similarity (promedio): {np.mean(list(inter_sim.values())):.4f}")
        print(f"Ratio Intra/Inter: {np.mean(intra_values) / np.mean(list(inter_sim.values())):.2f}x")
        print("(Ideal: ratio alto = tareas bien separadas)\n")

    def plot_tsne(self, embeddings_dict, save_path=None, perplexity=30):
        """
        Visualización t-SNE de embeddings
        """
        print("\nGenerando t-SNE...")

        # Concatenar embeddings y crear etiquetas de tarea
        all_embeddings = []
        task_labels = []

        for task_id in sorted(embeddings_dict.keys()):
            emb = embeddings_dict[task_id]
            all_embeddings.append(emb.numpy())
            task_labels.extend([task_id] * len(emb))

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        task_labels = np.array(task_labels)

        # Aplicar t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        embeddings_2d = tsne.fit_transform(all_embeddings)

        # Plot
        plt.figure(figsize=(12, 9))

        colors = plt.cm.tab10(np.linspace(0, 1, len(embeddings_dict)))

        for task_id, color in zip(sorted(embeddings_dict.keys()), colors):
            mask = task_labels == task_id
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=[color], label=f'Task {task_id}', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)

        plt.xlabel('t-SNE dimension 1', fontsize=12)
        plt.ylabel('t-SNE dimension 2', fontsize=12)
        plt.title('t-SNE Visualization of Task Embeddings', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ t-SNE guardado: {save_path}")

        plt.show()

    def plot_pca(self, embeddings_dict, save_path=None):
        """
        Visualización PCA de embeddings (más rápido que t-SNE)
        """
        print("\nGenerando PCA...")

        all_embeddings = []
        task_labels = []

        for task_id in sorted(embeddings_dict.keys()):
            emb = embeddings_dict[task_id]
            all_embeddings.append(emb.numpy())
            task_labels.extend([task_id] * len(emb))

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        task_labels = np.array(task_labels)

        # Aplicar PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings)

        # Plot
        plt.figure(figsize=(12, 9))

        colors = plt.cm.tab10(np.linspace(0, 1, len(embeddings_dict)))

        for task_id, color in zip(sorted(embeddings_dict.keys()), colors):
            mask = task_labels == task_id
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        c=[color], label=f'Task {task_id}', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=12)
        plt.title('PCA Visualization of Task Embeddings', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ PCA guardado: {save_path}")

        plt.show()

    def compare_with_without_prompts(self, save_dir=None):
        """
        Compara embeddings con y sin prompts
        """
        if not self.embeddings or not self.prompted_embeddings:
            print("⚠️ Debes extraer embeddings con y sin prompts primero")
            return

        print("\n" + "=" * 80)
        print("COMPARACIÓN: CON vs SIN PROMPTS")
        print("=" * 80)

        # Similaridad sin prompts
        intra_base, inter_base = self.compute_intra_inter_similarity(self.embeddings)

        # Similaridad con prompts
        intra_prompt, inter_prompt = self.compute_intra_inter_similarity(self.prompted_embeddings)

        # Comparar
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        tasks = sorted(intra_base.keys())
        x = np.arange(len(tasks))
        width = 0.35

        # Intra-task
        intra_base_vals = [intra_base[t] for t in tasks]
        intra_prompt_vals = [intra_prompt[t] for t in tasks]

        ax1.bar(x - width / 2, intra_base_vals, width, label='Sin Prompts', alpha=0.7, color='coral')
        ax1.bar(x + width / 2, intra_prompt_vals, width, label='Con Prompts', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Task ID')
        ax1.set_ylabel('Mean Intra-Task Similarity')
        ax1.set_title('Intra-Task Similarity')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tasks)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Inter-task
        inter_base_vals = list(inter_base.values())
        inter_prompt_vals = list(inter_prompt.values())

        ax2.bar([0], [np.mean(inter_base_vals)], width=0.4, label='Sin Prompts',
                alpha=0.7, color='coral')
        ax2.bar([1], [np.mean(inter_prompt_vals)], width=0.4, label='Con Prompts',
                alpha=0.7, color='steelblue')
        ax2.set_ylabel('Mean Inter-Task Similarity')
        ax2.set_title('Inter-Task Similarity (promedio)')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Sin Prompts', 'Con Prompts'])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / 'prompts_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparación guardada: {save_path}")

        plt.show()

        # Print metrics
        print(f"\nSIN PROMPTS:")
        print(f"  Intra-task: {np.mean(intra_base_vals):.4f}")
        print(f"  Inter-task: {np.mean(inter_base_vals):.4f}")
        print(f"  Ratio: {np.mean(intra_base_vals) / np.mean(inter_base_vals):.2f}x")

        print(f"\nCON PROMPTS:")
        print(f"  Intra-task: {np.mean(intra_prompt_vals):.4f}")
        print(f"  Inter-task: {np.mean(inter_prompt_vals):.4f}")
        print(f"  Ratio: {np.mean(intra_prompt_vals) / np.mean(inter_prompt_vals):.2f}x")


def load_model_from_checkpoint(checkpoint_path, n_tasks, classes_per_task,
                                pool_size=10, prompt_length=5, top_k=5,
                                use_g_prompt=True, use_e_prompt=True):
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM
    device = get_device()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    text_encoder = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PromptedLETS(
        n_tasks=n_tasks,
        classes_per_task=classes_per_task,
        pool_size=pool_size,
        prompt_length=prompt_length,
        top_k=top_k,
        use_g_prompt=use_g_prompt,
        use_e_prompt=use_e_prompt,
        text_encoder=text_encoder
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Cargar parcialmente sin petar
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Ignored {len(unexpected)} unexpected and {len(missing)} missing keys.")
    return model


def main():
    parser = argparse.ArgumentParser(description='Analizar similaridad de embeddings entre tareas')

    # Data
    parser.add_argument('--dataset', type=str, default='dailysport')
    parser.add_argument('--n_tasks', type=int, default=6)
    parser.add_argument('--classes_per_task', type=int, default=3)

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path al checkpoint del modelo entrenado')
    parser.add_argument('--pool_size', type=int, default=10)
    parser.add_argument('--prompt_length', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--use_g_prompt', action='store_true', default=True)
    parser.add_argument('--use_e_prompt', action='store_true', default=True)

    # Analysis
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (reducido para LLaMA)')
    parser.add_argument('--use_test', action='store_true',
                        help='Usar test set en lugar de train')
    parser.add_argument('--tsne_perplexity', type=int, default=30)

    # Output
    parser.add_argument('--output_dir', type=str, default='embedding_analysis')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Verificar autenticación de HuggingFace (el modelo la maneja internamente)
    if 'HF_TOKEN' in os.environ:
        print("✓ HuggingFace token detectado en variables de entorno")
    else:
        print("⚠️ No se detectó HF_TOKEN. Si tienes problemas, configura:")
        print("   export HF_TOKEN='tu_token_aqui'")
        print("   o usa: huggingface-cli login")

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Cargar datos
    print("\nCargando datos...")
    base_path = Path('../data') / args.dataset

    if args.use_test:
        x_data, y_data = prepare_dataset(base_path / 'x_test.pkl', base_path / 'state_test.pkl')
        split_name = 'test'
    else:
        x_data, y_data = prepare_dataset(base_path / 'x_train.pkl', base_path / 'state_train.pkl')
        split_name = 'train'

    # Crear subconjuntos por tarea
    task_data = {}
    for task_id in range(args.n_tasks):
        start_class = task_id * args.classes_per_task
        end_class = (task_id + 1) * args.classes_per_task
        task_classes = list(range(start_class, end_class))

        mask = torch.isin(y_data, torch.tensor(task_classes))
        task_data[task_id] = (x_data[mask], y_data[mask])

        print(f"Task {task_id}: {mask.sum()} samples, classes {task_classes}")

    # Cargar modelo
    print("\nCargando modelo...")
    model = load_model_from_checkpoint(
        args.checkpoint,
        args.n_tasks,
        args.classes_per_task,
        pool_size=args.pool_size,
        prompt_length=args.prompt_length,
        top_k=args.top_k,
        use_g_prompt=args.use_g_prompt,
        use_e_prompt=args.use_e_prompt
    )

    # Crear analyzer
    analyzer = EmbeddingAnalyzer(model, device=device, batch_size=args.batch_size)

    # Extraer embeddings (sin prompts)
    analyzer.extract_all_tasks(task_data, use_prompts=False)

    # Extraer embeddings (con prompts)
    analyzer.extract_all_tasks(task_data, use_prompts=True)

    # === ANÁLISIS ===

    print("\n" + "=" * 80)
    print("GENERANDO VISUALIZACIONES")
    print("=" * 80)

    # 1. Matriz de similaridad (sin prompts)
    analyzer.plot_similarity_matrix(
        analyzer.embeddings,
        save_path=output_dir / f'similarity_matrix_base_{split_name}.png',
        title=f"Base Embeddings ({split_name})"
    )

    # 2. Matriz de similaridad (con prompts)
    analyzer.plot_similarity_matrix(
        analyzer.prompted_embeddings,
        save_path=output_dir / f'similarity_matrix_prompted_{split_name}.png',
        title=f"Prompted Embeddings ({split_name})"
    )

    # 3. Comparación Intra/Inter (con prompts)
    analyzer.plot_intra_inter_comparison(
        analyzer.prompted_embeddings,
        save_path=output_dir / f'intra_inter_comparison_{split_name}.png'
    )

    # 4. Comparación con/sin prompts
    analyzer.compare_with_without_prompts(save_dir=output_dir)

    # 5. t-SNE (con prompts)
    analyzer.plot_tsne(
        analyzer.prompted_embeddings,
        save_path=output_dir / f'tsne_prompted_{split_name}.png',
        perplexity=args.tsne_perplexity
    )

    # 6. PCA (con prompts)
    analyzer.plot_pca(
        analyzer.prompted_embeddings,
        save_path=output_dir / f'pca_prompted_{split_name}.png'
    )

    print("\n" + "=" * 80)
    print(f"✓ Análisis completado. Resultados en: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()