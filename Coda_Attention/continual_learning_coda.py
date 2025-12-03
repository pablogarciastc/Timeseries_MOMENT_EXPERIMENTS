

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

from main import prepare_dataset
from moment_multihead_coda import PromptedMOMENT
from utils import set_seed, AverageMeter, save_checkpoint, get_device


class ContinualLearningTrainer:

    def __init__(self, args):
        self.args = args
        self.device = get_device()
        set_seed(args.seed)

        self.n_tasks = args.n_tasks
        self.classes_per_task = None
        self.task_classes = {}

        self._load_data()

        n_classes = len(torch.unique(self.y_train))
        self.classes_per_task = n_classes // self.n_tasks

        self.task_classes = {
            i: list(range(i * self.classes_per_task, (i + 1) * self.classes_per_task))
            for i in range(self.n_tasks)
        }

        print(f"\n{'='*80}")
        print("CONTINUAL LEARNING WITH CODA-PROMPT")
        print(f"{'='*80}")

        self.model = PromptedMOMENT(
            n_tasks=self.n_tasks,
            classes_per_task=self.classes_per_task,
            pool_size=args.pool_size,
            prompt_length=args.prompt_length,
            top_k=args.top_k,
            moment_model=args.moment_model,
            use_g_prompt=args.use_g_prompt,
            use_e_prompt=args.use_e_prompt,
            task_predictor_type=args.task_predictor_type
        ).to(self.device)

        self.results = {
            'oracle_accuracies': {},
            'soft_accuracies': {},
            'task_prediction_accuracies': {},
            'forgetting_oracle': {},
            'forgetting_soft': {},
            'config': vars(args)
        }

    def _load_data(self):
        print("Loading data...")

        base_path = Path('../data') / self.args.dataset
        self.x_train, self.y_train = prepare_dataset(base_path / 'x_train.pkl', base_path / 'state_train.pkl')
        self.x_test, self.y_test = prepare_dataset(base_path / 'x_test.pkl', base_path / 'state_test.pkl')

        print(f"Train: {self.x_train.shape}, Test: {self.x_test.shape}")

        n_classes = len(torch.unique(self.y_train))
        self.classes_per_task = n_classes // self.n_tasks
        if n_classes % self.n_tasks != 0:
            print(f"⚠️ Warning: {n_classes} classes don't divide evenly into {self.n_tasks} tasks.")
            print("Last tasks might have fewer classes.")

        self.task_classes = {}
        for i in range(self.n_tasks):
            start = i * self.classes_per_task
            end = (i + 1) * self.classes_per_task if i < self.n_tasks - 1 else n_classes
            self.task_classes[i] = list(range(start, end))

        print(f"\nTotal classes: {n_classes} | Tasks: {self.n_tasks} | Classes per task: {self.classes_per_task}")
        for tid, classes in self.task_classes.items():
            print(f"Task {tid+1}: {classes}")

        self.task_train_data = {}
        self.task_test_data = {}

        for task_id, classes in self.task_classes.items():
            train_mask = torch.isin(self.y_train, torch.tensor(classes))
            test_mask = torch.isin(self.y_test, torch.tensor(classes))
            self.task_train_data[task_id] = (self.x_train[train_mask], self.y_train[train_mask])
            self.task_test_data[task_id] = (self.x_test[test_mask], self.y_test[test_mask])
            print(f"Task {task_id+1}: Train={train_mask.sum()}, Test={test_mask.sum()}")

    def train_task(self, task_id, epochs):
        print(f"\n{'='*80}")
        print(f"Training Task {task_id + 1}/{self.n_tasks}: Classes {self.task_classes[task_id]}")
        print(f"{'='*80}\n")

        x_train, y_train = self.task_train_data[task_id]

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2
        )

        if task_id == 0:
            g_prompt_lr = self.args.lr
            e_prompt_lr = self.args.lr
        else:
            g_prompt_lr = self.args.lr
            e_prompt_lr = self.args.lr * 0.1

        optimizer_params = [
            {
                'params': self.model.classifier.heads[task_id].parameters(),
                'lr': self.args.lr,
                'name': 'classifier'
            }
        ]

        if hasattr(self.model.task_predictor, 'task_keys'):
            optimizer_params.extend([
                {
                    'params': [self.model.task_predictor.task_keys[task_id]],
                    'lr': self.args.lr * 3.0,
                    'name': 'task_key'
                },
                {
                    'params': self.model.task_predictor.projection.parameters(),
                    'lr': self.args.lr * 2.0,
                    'name': 'task_projection'
                }
            ])
        elif hasattr(self.model.task_predictor, 'task_prototypes'):
            optimizer_params.extend([
                {
                    'params': [self.model.task_predictor.task_prototypes],
                    'lr': self.args.lr * 2.0,
                    'name': 'task_prototypes'
                },
                {
                    'params': self.model.task_predictor.feature_projector.parameters(),
                    'lr': self.args.lr * 1.5,
                    'name': 'feature_projector'
                },
                {
                    'params': self.model.task_predictor.cross_attention.parameters(),
                    'lr': self.args.lr * 1.5,
                    'name': 'cross_attention'
                },
                {
                    'params': self.model.task_predictor.classifier.parameters(),
                    'lr': self.args.lr * 2.0,
                    'name': 'task_classifier'
                }
            ])
        elif hasattr(self.model.task_predictor, 'attention_predictor'):
            optimizer_params.extend([
                {
                    'params': self.model.task_predictor.attention_predictor.parameters(),
                    'lr': self.args.lr * 1.5,
                    'name': 'attention_path'
                },
                {
                    'params': self.model.task_predictor.distance_projector.parameters(),
                    'lr': self.args.lr * 1.5,
                    'name': 'distance_projector'
                },
                {
                    'params': [self.model.task_predictor.distance_keys[task_id]],
                    'lr': self.args.lr * 3.0,
                    'name': 'distance_key'
                }
            ])

        if self.args.use_g_prompt:
            optimizer_params.append({
                'params': [self.model.coda_prompt.g_prompts[task_id]],
                'lr': g_prompt_lr,
                'name': 'g_prompt'
            })

        if self.args.use_e_prompt:
            optimizer_params.extend([
                {
                    'params': [self.model.coda_prompt.e_prompts],
                    'lr': e_prompt_lr,
                    'name': 'e_prompts'
                },
                {
                    'params': [self.model.coda_prompt.e_keys],
                    'lr': e_prompt_lr,
                    'name': 'e_keys'
                }
            ])

        optimizer = optim.Adam(optimizer_params, weight_decay=self.args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            losses = AverageMeter()
            cls_losses = AverageMeter()
            task_losses = AverageMeter()
            cls_accs = AverageMeter()
            task_accs = AverageMeter()

            pbar = tqdm(train_loader, desc=f"Task {task_id+1} Epoch {epoch+1}/{epochs}")
            for x_batch, y_batch in pbar:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                local_labels = y_batch - (task_id * self.classes_per_task)

                logits, task_loss, task_acc, task_info = self.model.forward_with_task_loss(
                    x_batch, y_batch, task_id
                )

                cls_loss = criterion(logits, local_labels)

                diversity_loss = task_info.get('diversity_loss', 0)
                if isinstance(diversity_loss, torch.Tensor):
                    total_loss = cls_loss + self.args.task_loss_weight * task_loss + \
                                self.args.diversity_loss_weight * diversity_loss
                else:
                    total_loss = cls_loss + self.args.task_loss_weight * task_loss

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                _, predicted = logits.max(1)
                cls_acc = predicted.eq(local_labels).sum().item() / local_labels.size(0)

                losses.update(total_loss.item(), y_batch.size(0))
                cls_losses.update(cls_loss.item(), y_batch.size(0))
                task_losses.update(task_loss.item(), y_batch.size(0))
                cls_accs.update(cls_acc, y_batch.size(0))
                task_accs.update(task_acc.item(), y_batch.size(0))

                pbar.set_postfix({
                    'loss': f'{losses.avg:.4f}',
                    'cls_acc': f'{cls_accs.avg*100:.2f}%',
                    'task_acc': f'{task_accs.avg*100:.2f}%'
                })

            print(f"Epoch {epoch+1}: Loss={losses.avg:.4f}, "
                  f"Cls={cls_accs.avg*100:.2f}%, Task={task_accs.avg*100:.2f}%")

        self.model.freeze_task(task_id)
        print(f"\n🔒 Task {task_id} frozen\n")

        if self.args.use_e_prompt:
            stats = self.model.coda_prompt.get_prompt_statistics()
            if stats:
                print(f"\nE-Prompt Statistics:")
                print(f"  Most used: {stats['most_used']} | Least used: {stats['least_used']}")

    @torch.no_grad()
    def evaluate_task(self, task_id, use_oracle=False):
        self.model.eval()

        x_test, y_test = self.task_test_data[task_id]

        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=2
        )

        correct = 0
        total = 0
        task_correct = 0

        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            if use_oracle:
                logits = self.model(x_batch, task_id=task_id)
                predicted_task = task_id
            else:
                logits, task_info = self.model(x_batch, task_id=None, return_task_info=True)
                predicted_task = task_info['predicted_task']
                task_correct += (predicted_task == task_id) * len(y_batch)

            predicted = logits.max(1)[1]
            predicted_global = predicted + (predicted_task * self.classes_per_task)

            correct += predicted_global.eq(y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = 100.0 * correct / total

        if not use_oracle:
            task_acc = 100.0 * task_correct / total
            return accuracy, task_acc

        return accuracy

    @torch.no_grad()
    def evaluate_all_tasks(self, up_to_task=None, use_oracle=False):
        accuracies = {}
        task_accuracies = {}
        max_task = up_to_task if up_to_task is not None else self.n_tasks

        for task_id in range(max_task):
            if task_id in self.task_test_data:
                if use_oracle:
                    acc = self.evaluate_task(task_id, use_oracle=True)
                    accuracies[task_id] = acc
                else:
                    acc, task_acc = self.evaluate_task(task_id, use_oracle=False)
                    accuracies[task_id] = acc
                    task_accuracies[task_id] = task_acc

        if use_oracle:
            return accuracies
        return accuracies, task_accuracies

    def run_continual_learning(self):
        print(f"Task loss weight: {self.args.task_loss_weight}")
        print(f"Epochs per task: {self.args.epochs_per_task}")
        print(f"G-Prompt enabled: {self.args.use_g_prompt}")
        print(f"E-Prompt enabled: {self.args.use_e_prompt}\n")

        initial_oracle_accs = {}
        initial_soft_accs = {}

        for task_id in range(self.n_tasks):
            self.train_task(task_id, self.args.epochs_per_task)

            print(f"\n{'='*80}")
            print(f"Evaluation after Task {task_id + 1}")
            print(f"{'='*80}")

            print(f"\n--- Oracle Mode (Upper Bound) ---")
            oracle_accs = self.evaluate_all_tasks(up_to_task=task_id + 1, use_oracle=True)

            for tid, acc in oracle_accs.items():
                print(f"Task {tid+1}: {acc:.2f}%")
                if tid not in initial_oracle_accs:
                    initial_oracle_accs[tid] = acc

            oracle_avg = np.mean(list(oracle_accs.values()))
            print(f"Average: {oracle_avg:.2f}%")

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

            print("="*80 + "\n")

            if self.args.save_checkpoints:
                checkpoint_path = Path(self.args.output_dir) / f"task_{task_id+1}_checkpoint.pt"
                save_checkpoint({
                    'task_id': task_id,
                    'model_state_dict': self.model.state_dict(),
                    'results': self.results
                }, checkpoint_path)

        self._print_final_summary()
        self._save_results()

        return self.results

    def _print_final_summary(self):
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        final_task = self.n_tasks - 1

        print("\n--- Oracle Mode ---")
        oracle_final = self.results['oracle_accuracies'][final_task]
        for tid, acc in oracle_final.items():
            print(f"Task {tid+1}: {acc:.2f}%")
        print(f"Final Average: {np.mean(list(oracle_final.values())):.2f}%")

        print("\n--- Soft Prediction Mode ---")
        soft_final = self.results['soft_accuracies'][final_task]
        task_pred_final = self.results['task_prediction_accuracies'][final_task]
        for tid, acc in soft_final.items():
            print(f"Task {tid+1}: {acc:.2f}% (Task Pred: {task_pred_final[tid]:.2f}%)")
        print(f"Final Classification Avg: {np.mean(list(soft_final.values())):.2f}%")
        print(f"Final Task Prediction Avg: {np.mean(list(task_pred_final.values())):.2f}%")

        if self.n_tasks - 1 in self.results['forgetting_oracle']:
            print("\n--- Final Forgetting ---")
            oracle_forg = self.results['forgetting_oracle'][self.n_tasks - 1]
            print(f"Oracle Forgetting: {np.mean(list(oracle_forg.values())):.2f}%")

            soft_forg = self.results['forgetting_soft'][self.n_tasks - 1]
            print(f"Soft Forgetting: {np.mean(list(soft_forg.values())):.2f}%")

        print("="*80)

    def _save_results(self):
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        results_serializable = {
            'oracle_accuracies': {
                str(k): {str(k2): float(v2) for k2, v2 in v.items()}
                for k, v in self.results['oracle_accuracies'].items()
            },
            'soft_accuracies': {
                str(k): {str(k2): float(v2) for k2, v2 in v.items()}
                for k, v in self.results['soft_accuracies'].items()
            },
            'task_prediction_accuracies': {
                str(k): {str(k2): float(v2) for k2, v2 in v.items()}
                for k, v in self.results['task_prediction_accuracies'].items()
            },
            'forgetting_oracle': {
                str(k): {str(k2): float(v2) for k2, v2 in v.items()}
                for k, v in self.results['forgetting_oracle'].items()
            },
            'forgetting_soft': {
                str(k): {str(k2): float(v2) for k2, v2 in v.items()}
                for k, v in self.results['forgetting_soft'].items()
            },
            'config': self.results['config']
        }

        results_path = output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\n✓ Results saved: {results_path}")

        final_model_path = output_dir / 'final_model.pt'
        save_checkpoint({
            'model_state_dict': self.model.state_dict(),
            'results': self.results,
            'config': vars(self.args)
        }, final_model_path)

        print(f"✓ Final model saved: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description='Continual Learning with CODA-Prompt')

    parser.add_argument('--dataset', type=str, default='dailysport')

    parser.add_argument('--x_train', type=str, default='x_train.pkl')
    parser.add_argument('--x_test', type=str, default='x_test.pkl')
    parser.add_argument('--state_train', type=str, default='state_train.pkl')
    parser.add_argument('--state_test', type=str, default='state_test.pkl')

    parser.add_argument('--moment_model', type=str, default='small',
                        choices=['small', 'base', 'large'])
    parser.add_argument('--n_tasks', type=int, default=6)

    parser.add_argument('--prompt_length', type=int, default=5,
                        help='Length of each prompt')
    parser.add_argument('--pool_size', type=int, default=10,
                        help='Size of E-Prompt pool (shared prompts)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of E-Prompts to select')
    parser.add_argument('--use_g_prompt', action='store_true', default=True,
                        help='Use G-Prompt (task-specific prompts)')
    parser.add_argument('--use_e_prompt', action='store_true', default=True,
                        help='Use E-Prompt (shared general prompts)')

    parser.add_argument('--task_predictor_type', type=str, default='attention',
                        choices=['simple', 'attention', 'hybrid'],
                        help='Type of task predictor: simple (key matching), '
                             'attention (cross-attention), hybrid (ensemble)')
    parser.add_argument('--task_loss_weight', type=float, default=1.0,
                        help='Weight for task prediction loss')
    parser.add_argument('--diversity_loss_weight', type=float, default=0.01,
                        help='Weight for prompt diversity loss')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs_per_task', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='coda_results')
    parser.add_argument('--save_checkpoints', action='store_true')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("CONTINUAL LEARNING WITH CODA-PROMPT")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Tasks: {args.n_tasks}")
    print(f"Prompt length: {args.prompt_length}")
    print(f"E-Prompt pool size: {args.pool_size}")
    print(f"Top-K selection: {args.top_k}")
    print(f"G-Prompt: {args.use_g_prompt}")
    print(f"E-Prompt: {args.use_e_prompt}")
    print(f"Epochs per task: {args.epochs_per_task}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")

    trainer = ContinualLearningTrainer(args)
    results = trainer.run_continual_learning()

    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()