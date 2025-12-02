"""
Script de Continual Learning CON DIAGNÓSTICOS INTEGRADOS

Uso:
    python continual_learning_debug.py --dataset dailysport --pool_size 60 \
        --epochs_per_task 3 --task_loss_weight 0.5 --n_tasks 6 --top_k 1 \
        --moment_model small --batch_size 32 --run_diagnostics
"""

import sys
import os

# Agregar el directorio de uploads al path
sys.path.insert(0, '/mnt/user-data/uploads')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import torch.nn.functional as F

# Importar desde /home/claude
sys.path.insert(0, '/home/claude')
from diagnostic_tool import ContinualLearningDiagnostic, run_diagnostics_after_task

# Imports from uploads
from continual_learning import ContinualLearningTrainer


class ContinualLearningTrainerDebug(ContinualLearningTrainer):
    """Versión extendida con diagnósticos"""
    
    def run_continual_learning(self):
        print(f"Task loss weight: {self.args.task_loss_weight}")
        print(f"Epochs per task: {self.args.epochs_per_task}")
        print(f"Run diagnostics: {self.args.run_diagnostics}\n")

        initial_oracle_accs = {}
        initial_soft_accs = {}

        for task_id in range(self.n_tasks):
            # Train
            self.train_task(task_id, self.args.epochs_per_task)

            # Evaluate
            print(f"\n{'='*80}")
            print(f"Evaluación después de Task {task_id + 1}")
            print(f"{'='*80}")

            # Oracle
            print(f"\n--- Oracle Mode (Upper Bound) ---")
            oracle_accs = self.evaluate_all_tasks(up_to_task=task_id + 1, use_oracle=True)

            for tid, acc in oracle_accs.items():
                print(f"Task {tid+1}: {acc:.2f}%")
                if tid not in initial_oracle_accs:
                    initial_oracle_accs[tid] = acc

            oracle_avg = np.mean(list(oracle_accs.values()))
            print(f"Average: {oracle_avg:.2f}%")

            # Soft
            print(f"\n--- Soft Prediction Mode ---")
            soft_accs, task_pred_accs = self.evaluate_all_tasks(up_to_task=task_id + 1, use_oracle=False)

            for tid, acc in soft_accs.items():
                task_pred_acc = task_pred_accs[tid]
                print(f"Task {tid+1}: {acc:.2f}% (Task Pred: {task_pred_acc:.2f}%)")
                if tid not in initial_soft_accs:
                    initial_soft_accs[tid] = acc

            soft_avg = np.mean(list(soft_accs.values()))
            task_pred_avg = np.mean(list(task_pred_accs.values()))
            print(f"Average Acc: {soft_avg:.2f}%")
            print(f"Average Task Pred: {task_pred_avg:.2f}%")

            # Forgetting
            if task_id > 0:
                print(f"\n--- Forgetting ---")

                oracle_forg = {}
                for tid in range(task_id):
                    oracle_forg[tid] = initial_oracle_accs[tid] - oracle_accs[tid]
                oracle_avg_forg = np.mean(list(oracle_forg.values()))
                print(f"Oracle Forgetting: {oracle_avg_forg:.2f}%")
                for tid, forg in oracle_forg.items():
                    print(f"  Task {tid+1}: {forg:+.2f}%")

                soft_forg = {}
                for tid in range(task_id):
                    soft_forg[tid] = initial_soft_accs[tid] - soft_accs[tid]
                soft_avg_forg = np.mean(list(soft_forg.values()))
                print(f"Soft Forgetting: {soft_avg_forg:.2f}%")
                for tid, forg in soft_forg.items():
                    print(f"  Task {tid+1}: {forg:+.2f}%")

                self.results['forgetting_oracle'][task_id] = oracle_forg
                self.results['forgetting_soft'][task_id] = soft_forg

            self.results['oracle_accuracies'][task_id] = oracle_accs
            self.results['soft_accuracies'][task_id] = soft_accs
            self.results['task_prediction_accuracies'][task_id] = task_pred_accs

            # 🔍 EJECUTAR DIAGNÓSTICOS
            if self.args.run_diagnostics:
                run_diagnostics_after_task(self, task_id)

            print("="*80 + "\n")

            if self.args.save_checkpoints:
                from continual_learning import save_checkpoint
                checkpoint_path = Path(self.args.output_dir) / f"task_{task_id+1}_checkpoint.pt"
                save_checkpoint({
                    'task_id': task_id,
                    'model_state_dict': self.model.state_dict(),
                    'results': self.results
                }, checkpoint_path)

        self._print_final_summary()
        self._save_results()

        return self.results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='dailysport')

    # Data
    parser.add_argument('--x_train', type=str, default='x_train.pkl')
    parser.add_argument('--x_test', type=str, default='x_test.pkl')
    parser.add_argument('--state_train', type=str, default='state_train.pkl')
    parser.add_argument('--state_test', type=str, default='state_test.pkl')

    # Model
    parser.add_argument('--moment_model', type=str, default='small')
    parser.add_argument('--prompt_length', type=int, default=5)
    parser.add_argument('--pool_size', type=int, default=20)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--n_tasks', type=int, default=6)

    parser.add_argument('--task_loss_weight', type=float, default=1.0)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs_per_task', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Diagnostics
    parser.add_argument('--run_diagnostics', action='store_true', 
                       help='Ejecutar análisis diagnóstico después de cada tarea')

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='continual_results_debug')
    parser.add_argument('--save_checkpoints', action='store_true')

    args = parser.parse_args()

    trainer = ContinualLearningTrainerDebug(args)
    results = trainer.run_continual_learning()

    print("\n✅ Entrenamiento y diagnósticos completados!")


if __name__ == "__main__":
    main()
