"""
Continual Learning con MOMENT + L2Prompt + Multi-Head

CON PARAMETER FREEZING CORRECTO para prevenir forgetting
"""

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
from textlets_multihead import PromptedLETS
from utils import set_seed, AverageMeter, save_checkpoint, get_device


class ContinualLearningTrainer:
    """
    Trainer con parameter freezing correcto

    Estrategia:
    1. Train Task T â†’ congela classifier head T y task key T
    2. Train Task T+1 â†’ solo head T+1 y key T+1 son trainable
    3. L2Prompt pool se entrena con LR reducido despuÃ©s del primer task
    """

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

        # Crear asignación automática de clases por tarea
        self.task_classes = {
            i: list(range(i * self.classes_per_task, (i + 1) * self.classes_per_task))
            for i in range(self.n_tasks)
        }

        print(f"\n{'='*80}")
        print("CONTINUAL LEARNING CON PARAMETER FREEZING")
        print(f"{'='*80}")

        self.model = PromptedLETS(
            n_tasks=self.n_tasks,
            classes_per_task=self.classes_per_task,
            pool_size=args.pool_size,
            prompt_length=args.prompt_length,
            top_k=args.top_k,
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
        print("Cargando datos...")

        base_path = Path('../data') / self.args.dataset
        self.x_train, self.y_train = prepare_dataset(base_path / 'x_train.pkl', base_path / 'state_train.pkl')
        self.x_test, self.y_test = prepare_dataset(base_path / 'x_test.pkl', base_path / 'state_test.pkl')

        print(f"Train: {self.x_train.shape}, Test: {self.x_test.shape}")

        # Detectar número total de clases
        n_classes = len(torch.unique(self.y_train))
        self.classes_per_task = n_classes // self.n_tasks
        if n_classes % self.n_tasks != 0:
            print(f"⚠️ Aviso: {n_classes} clases no se dividen exactamente entre {self.n_tasks} tareas.")
            print("Las últimas tareas podrían tener menos clases.")

        # Crear asignación automática de clases por tarea
        self.task_classes = {}
        for i in range(self.n_tasks):
            start = i * self.classes_per_task
            end = (i + 1) * self.classes_per_task if i < self.n_tasks - 1 else n_classes
            self.task_classes[i] = list(range(start, end))

        print(f"\nTotal classes: {n_classes} | Tasks: {self.n_tasks} | Classes per task: {self.classes_per_task}")
        for tid, classes in self.task_classes.items():
            print(f"Task {tid+1}: {classes}")

        # Crear subconjuntos de entrenamiento y test
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
        print(f"Training Task {task_id + 1}/6: Classes {self.task_classes[task_id]}")
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
            prompt_lr = self.args.lr
        else:
            # Tasks siguientes: LR reducido para prompts
            prompt_lr = self.args.lr * 0.7  # 10x mÃ¡s bajo

        optimizer_params = [
            {
                'params': self.model.classifier.heads[task_id].parameters(),
                'lr': self.args.lr,
                'name': 'classifier'
            },
            {
                # G-Prompt: prompt específico de la tarea
                'params': [self.model.coda_prompt.g_prompts[task_id]],
                'lr': self.args.lr * 0.1,
                'name': 'g_prompt'
            },
            {
                # E-Prompts: pool compartido (entrena TODO el pool)
                'params': [self.model.coda_prompt.e_prompts],
                'lr': self.args.lr * 0.01,  # más bajo porque es compartido
                'name': 'e_prompts'
            },
            {
                # E-Keys: keys para selección de E-prompts
                'params': [self.model.coda_prompt.e_keys],
                'lr': self.args.lr * 0.005,  # muy bajo para mantener estabilidad
                'name': 'e_keys'
            },
            {
                # Task predictor key
                'params': [self.model.task_predictor.task_keys[task_id]],
                'lr': self.args.lr * 0.05,
                'name': 'task_key'
            }
        ]

        optimizer = optim.Adam(optimizer_params, weight_decay=self.args.weight_decay)
        criterion = nn.CrossEntropyLoss()

        import torch
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        # Supón que ya tienes text_encoder instanciado
        text_encoder = self.model.text_encoder.eval()  # o TextEmbedderLETS()


        # --- Paso 2. Extrae features y etiquetas del loader ---
        all_feats = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to("cuda").float()
                feats = text_encoder(x_batch)  # [B, d_model]
                all_feats.append(feats.cpu().numpy())
                all_labels.append(y_batch.cpu().numpy())

        X = np.concatenate(all_feats, axis=0)
        y = np.concatenate(all_labels, axis=0)

        print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        # Training loop
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

                logits, task_loss, task_acc, _ = self.model.forward_with_task_loss(
                    x_batch, y_batch, task_id
                )

                cls_loss = criterion(logits, local_labels)
                total_loss = cls_loss + self.args.task_loss_weight * task_loss

                total_loss.backward()
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

        # ðŸ”’ CRITICAL: Congelar task despuÃ©s de entrenar
        self.model.freeze_task(task_id)
        print(f"\nðŸ”’ Task {task_id} congelado\n")

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
        print(f"Epochs per task: {self.args.epochs_per_task}\n")

        initial_oracle_accs = {}
        initial_soft_accs = {}

        for task_id in range(self.n_tasks):
            # Train
            self.train_task(task_id, self.args.epochs_per_task)


            # Evaluate
            print(f"\n{'='*80}")
            print(f"EvaluaciÃ³n despuÃ©s de Task {task_id + 1}")
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
        print("RESUMEN FINAL")
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

        print(f"\nâœ“ Resultados guardados: {results_path}")

        final_model_path = output_dir / 'final_model.pt'
        save_checkpoint({
            'model_state_dict': self.model.state_dict(),
            'results': self.results,
            'config': vars(self.args)
        }, final_model_path)

        print(f"âœ“ Modelo final guardado: {final_model_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='dailysport')

    # Data
    parser.add_argument('--x_train', type=str, default='x_train.pkl')
    parser.add_argument('--x_test', type=str, default='x_test.pkl')
    parser.add_argument('--state_train', type=str, default='state_train.pkl')
    parser.add_argument('--state_test', type=str, default='state_test.pkl')

    # Model
    parser.add_argument('--prompt_length', type=int, default=5)
    parser.add_argument('--pool_size', type=int, default=20)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--n_tasks', type=int, default=6)

    # ðŸ”§ CRITICAL: task_loss_weight mÃ¡s alto
    parser.add_argument('--task_loss_weight', type=float, default=1.0)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs_per_task', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='continual_results')
    parser.add_argument('--save_checkpoints', action='store_true')

    args = parser.parse_args()

    trainer = ContinualLearningTrainer(args)
    results = trainer.run_continual_learning()

    print("\nâœ… Entrenamiento completado!")


if __name__ == "__main__":
    main()