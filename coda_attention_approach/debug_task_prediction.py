"""
Debug script to visualize task prediction behavior
Helps understand why task prediction is failing
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_task_keys(model):
    """Analyze the learned task keys"""
    print("\n" + "=" * 80)
    print("TASK KEY ANALYSIS")
    print("=" * 80)

    # Get all task keys
    keys = []
    for key in model.task_predictor.task_keys:
        keys.append(key.detach().cpu())

    keys = torch.stack(keys, dim=0)  # [n_tasks, d_model]
    keys_norm = F.normalize(keys, p=2, dim=1)

    # Compute pairwise similarities
    similarity = torch.matmul(keys_norm, keys_norm.t())

    print("\nPairwise Key Similarities (should be low):")
    print(similarity.numpy())

    # Compute key norms
    key_norms = torch.norm(keys, dim=1)
    print(f"\nKey norms: {key_norms.numpy()}")
    print(f"Mean: {key_norms.mean():.4f}, Std: {key_norms.std():.4f}")

    # Check if keys are too similar
    mask = torch.eye(len(keys))
    off_diag = similarity * (1 - mask)
    max_similarity = off_diag.max().item()
    mean_similarity = off_diag.abs().mean().item()

    print(f"\nOff-diagonal statistics:")
    print(f"  Max similarity: {max_similarity:.4f}")
    print(f"  Mean abs similarity: {mean_similarity:.4f}")

    if max_similarity > 0.7:
        print("  ⚠️ WARNING: Keys are too similar (>0.7)!")
    elif mean_similarity > 0.3:
        print("  ⚠️ WARNING: Keys lack diversity (mean >0.3)")
    else:
        print("  ✓ Keys are well-separated")

    return similarity.numpy()


def visualize_task_prediction(model, test_loader, device, task_id, max_batches=5):
    """Visualize what the model predicts for each task"""
    print(f"\n{'=' * 80}")
    print(f"TASK PREDICTION VISUALIZATION - Task {task_id + 1}")
    print(f"{'=' * 80}")

    model.eval()

    all_predictions = []
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            if i >= max_batches:
                break

            x_batch = x_batch.to(device)

            # Get task prediction info
            _, task_info = model(x_batch, task_id=None, return_task_info=True)

            all_predictions.append(task_info['predicted_task'])
            all_probs.append(task_info['task_probs'].cpu())
            all_logits.append(task_info['task_logits'].cpu())

    # Aggregate
    all_probs = torch.cat(all_probs, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    print(f"\nTask {task_id + 1} samples analyzed: {len(all_probs)}")
    print(f"Most common predictions: {all_predictions}")

    # Average probabilities
    mean_probs = all_probs.mean(dim=0)
    print(f"\nAverage prediction probabilities:")
    for i, prob in enumerate(mean_probs):
        marker = "✓" if i == task_id else "✗"
        print(f"  {marker} Task {i + 1}: {prob:.4f}")

    # Check if correct task is even in top-3
    top3 = mean_probs.topk(3)
    if task_id in top3.indices:
        print(f"✓ Correct task {task_id + 1} is in top-3 predictions")
    else:
        print(f"❌ Correct task {task_id + 1} is NOT in top-3 predictions!")

    return all_probs.numpy(), all_logits.numpy()


def plot_confusion_matrix(model, task_test_data, device, n_tasks):
    """Create confusion matrix for task prediction"""
    print(f"\n{'=' * 80}")
    print("TASK PREDICTION CONFUSION MATRIX")
    print(f"{'=' * 80}")

    from torch.utils.data import DataLoader, TensorDataset

    confusion = np.zeros((n_tasks, n_tasks))

    model.eval()

    with torch.no_grad():
        for true_task_id in range(n_tasks):
            if true_task_id not in task_test_data:
                continue

            x_test, y_test = task_test_data[true_task_id]
            test_dataset = TensorDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            predictions = []

            for x_batch, _ in test_loader:
                x_batch = x_batch.to(device)
                _, task_info = model(x_batch, task_id=None, return_task_info=True)

                # Get per-sample predictions
                _, pred_tasks = task_info['task_logits'].max(dim=-1)
                predictions.extend(pred_tasks.cpu().tolist())

            # Count predictions
            for pred in predictions:
                confusion[true_task_id, pred] += 1

            # Normalize row
            if confusion[true_task_id].sum() > 0:
                confusion[true_task_id] /= confusion[true_task_id].sum()

    # Print confusion matrix
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("     ", end="")
    for i in range(n_tasks):
        print(f"T{i + 1:2d}  ", end="")
    print()

    for i in range(n_tasks):
        print(f"T{i + 1:2d} | ", end="")
        for j in range(n_tasks):
            val = confusion[i, j]
            if i == j:
                color = "\033[92m" if val > 0.8 else "\033[93m" if val > 0.5 else "\033[91m"
                print(f"{color}{val:.2f}\033[0m ", end="")
            else:
                print(f"{val:.2f} ", end="")
        print()

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, cmap='Blues', aspect='auto')
    plt.colorbar(label='Prediction Probability')
    plt.xlabel('Predicted Task')
    plt.ylabel('True Task')
    plt.title('Task Prediction Confusion Matrix')

    # Add text annotations
    for i in range(n_tasks):
        for j in range(n_tasks):
            text = f'{confusion[i, j]:.2f}'
            color = 'white' if confusion[i, j] > 0.5 else 'black'
            plt.text(j, i, text, ha='center', va='center', color=color)

    plt.xticks(range(n_tasks), [f'T{i + 1}' for i in range(n_tasks)])
    plt.yticks(range(n_tasks), [f'T{i + 1}' for i in range(n_tasks)])
    plt.tight_layout()
    plt.savefig('task_prediction_confusion.png', dpi=300)
    print("\n📊 Confusion matrix saved to task_prediction_confusion.png")

    return confusion


def main():
    """
    Usage:
    Add this to your continual_learning_coda.py after training each task:

    from debug_task_prediction import analyze_task_keys, visualize_task_prediction

    # After training
    analyze_task_keys(self.model)

    # Before evaluation
    for task_id in range(task_id + 1):
        visualize_task_prediction(
            self.model,
            test_loader,
            self.device,
            task_id
        )
    """
    print("This is a utility module. Import functions in your training script.")
    print("See docstring in main() for usage examples.")


if __name__ == "__main__":
    main()