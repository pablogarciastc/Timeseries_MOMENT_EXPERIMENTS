import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import argparse
import pickle
import torch.nn.functional as F

from momentfm import MOMENTPipeline
from utils import set_seed, AverageMeter, save_checkpoint, get_device

class MultiHeadClassifier(nn.Module):
    def __init__(self, n_tasks, classes_per_task, d_model, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.classes_per_task = classes_per_task
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, classes_per_task)
            ) for _ in range(n_tasks)
        ])
    def forward(self, x, task_id):
        if isinstance(task_id, int):
            return self.heads[task_id](x)
        if torch.is_tensor(task_id) and task_id.dim() == 0:
            return self.heads[int(task_id.item())](x)
        b = x.size(0)
        out = torch.zeros(b, self.classes_per_task, device=x.device)
        for tid, head in enumerate(self.heads):
            mask = task_id == tid
            if mask.any():
                out[mask] = head(x[mask])
        return out

class L2PromptPool(nn.Module):
    def __init__(self, pool_size=20, prompt_length=5, d_model=512, top_k=5):
        super().__init__()
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.d_model = d_model
        self.top_k = top_k
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_length, d_model) * 0.01)
        self.keys = nn.Parameter(torch.randn(pool_size, d_model) * 0.01)
    def select_prompts(self, x, task_key=None):
        B_real = x.size(0)
        query = x.mean(dim=1)
        if task_key is not None:
            if task_key.dim() == 1:
                query = (query + task_key.unsqueeze(0)) / 2.0
            elif task_key.dim() == 2:
                query = (query + task_key) / 2.0
        query_norm = F.normalize(query, p=2, dim=1)
        keys_norm = F.normalize(self.keys, p=2, dim=1)
        similarity = torch.matmul(query_norm, keys_norm.T)
        _, top_k_indices = similarity.topk(self.top_k, dim=1)
        expanded_indices = top_k_indices[:, :, None, None].expand(
            B_real,
            self.top_k,
            self.prompt_length,
            self.d_model
        )
        prompts_expanded = self.prompts[None, :, :, :].expand(B_real, -1, -1, -1)
        selected_prompts = torch.gather(prompts_expanded, 1, expanded_indices)
        return selected_prompts, top_k_indices
    def forward(self, x, task_key=None):
        B_real, _, d_model = x.shape
        selected_prompts, topk = self.select_prompts(x, task_key)
        selected_prompts_flat = selected_prompts.reshape(B_real, -1, d_model)
        x_with_prompts = torch.cat([selected_prompts_flat, x], dim=1)
        return x_with_prompts, topk

class TaskPredictor(nn.Module):
    def __init__(self, n_tasks, d_model):
        super().__init__()
        self.task_keys = nn.ParameterList([nn.Parameter(torch.randn(d_model)) for _ in range(n_tasks)])
        for k in self.task_keys:
            nn.init.normal_(k, mean=0, std=0.02)
        self.temperature = nn.Parameter(torch.ones(1))
    def l2_normalize(self, x, dim=-1, epsilon=1e-12):
        s = torch.sum(x ** 2, dim=dim, keepdim=True)
        inv = torch.rsqrt(torch.maximum(s, torch.tensor(epsilon, device=x.device)))
        return x * inv
    def forward(self, x, training=False, task_id=None):
        b = x.size(0)
        K = torch.stack(list(self.task_keys), dim=0)
        x = self.l2_normalize(x, dim=-1)
        K = self.l2_normalize(K, dim=-1)
        sim = torch.matmul(x, K.t())
        logits = sim / self.temperature
        probs = torch.softmax(logits, dim=-1)
        if training and task_id is not None:
            pred = task_id
        else:
            _, idx = logits.max(dim=-1)
            pred = torch.mode(idx).values.item() if b > 1 else idx[0].item()
        return {"predicted_task": pred, "task_logits": logits, "task_probs": probs}

class PromptedMOMENT(nn.Module):
    def __init__(self, base_model, n_tasks, pool_size=20, prompt_length=5, top_k=5, classes_per_task=3):
        super().__init__()
        self.model = base_model
        self.l2prompt = L2PromptPool(pool_size, prompt_length, base_model.config.d_model, top_k)
        self.task_keys = nn.ParameterList([nn.Parameter(torch.randn(base_model.config.d_model) * 0.01) for _ in range(n_tasks)])
        self.task_predictor = TaskPredictor(n_tasks=n_tasks, d_model=base_model.config.d_model)
        self.classifier = MultiHeadClassifier(n_tasks=n_tasks, classes_per_task=classes_per_task, d_model=base_model.config.d_model)
        for _, p in self.model.named_parameters():
            p.requires_grad = False
        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task
    def forward(self, x_enc, task_id=None, predict_task=False):
        bsz, n_channels, seq_len = x_enc.shape
        device = x_enc.device
        input_mask = torch.ones((bsz, seq_len), device=device)
        x_norm = self.model.normalizer(x=x_enc, mask=input_mask, mode="norm")
        patches = self.model.tokenizer(x=x_norm)
        enc_in = self.model.patch_embedding(patches, mask=input_mask)
        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(bsz * n_channels, n_patches, self.model.config.d_model)
        if predict_task and task_id is None:
            avg_embed = enc_in.mean(dim=1)
            tp_out = self.task_predictor(avg_embed, training=False, task_id=None)
            predicted_task_scalar = tp_out["predicted_task"]
            predicted_task_vec = torch.full((bsz * n_channels,), predicted_task_scalar, dtype=torch.long, device=device)
            task_key = self.task_keys[predicted_task_scalar]
        else:
            if task_id is None:
                predicted_task_scalar = 0
                predicted_task_vec = torch.zeros(bsz * n_channels, dtype=torch.long, device=device)
                task_key = None
            else:
                predicted_task_scalar = int(task_id)
                predicted_task_vec = torch.full((bsz * n_channels,), predicted_task_scalar, dtype=torch.long, device=device)
                task_key = self.task_keys[predicted_task_scalar]
        prompted, topk = self.l2prompt(enc_in, task_key)
        outputs = self.model.encoder(inputs_embeds=prompted)
        hidden_states = outputs.last_hidden_state
        feats = hidden_states.mean(dim=1)
        feats = feats.view(bsz, n_channels, -1).mean(dim=1)
        logits = self.classifier(feats, predicted_task_scalar if task_id is not None or predict_task else predicted_task_vec)
        return logits, predicted_task_scalar, topk

def isolate_gradients(model, selected_indices):
    if model.l2prompt.prompts.grad is None:
        return
    mask = torch.zeros_like(model.l2prompt.prompts.grad)
    for batch_topk in selected_indices:
        for idx in batch_topk:
            mask[idx] = 1.0
    model.l2prompt.prompts.grad *= mask

class ContinualLearningTrainer:
    def __init__(self, args):
        self.args = args
        self.device = get_device()
        set_seed(args.seed)
        base_model = MOMENTPipeline.from_pretrained(args.model_ckpt)
        self.classes_per_task = 3
        self.n_tasks = 18 // self.classes_per_task
        self.model = PromptedMOMENT(
            base_model,
            n_tasks=self.n_tasks,
            pool_size=args.pool_size,
            prompt_length=args.prompt_length,
            top_k=args.top_k,
            classes_per_task=self.classes_per_task
        ).to(self.device)
        for _, p in self.model.model.named_parameters():
            p.requires_grad = False
        for name, p in self.model.named_parameters():
            if any(k in name for k in ["l2prompt", "classifier", "task_predictor", "task_keys"]):
                p.requires_grad = True
        self.results = {
            "oracle_accuracies": {},
            "soft_accuracies": {},
            "task_prediction_accuracies": {},
            "forgetting_oracle": {},
            "forgetting_soft": {}
        }
        self._prepare_data()
    def _prepare_data(self):
        base_path = Path("../data/dailysport")
        with open(base_path / "x_train.pkl", "rb") as f:
            x_train = pickle.load(f)
        with open(base_path / "state_train.pkl", "rb") as f:
            y_train = pickle.load(f)
        with open(base_path / "x_test.pkl", "rb") as f:
            x_test = pickle.load(f)
        with open(base_path / "state_test.pkl", "rb") as f:
            y_test = pickle.load(f)
        self.x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)
        self.x_test = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)
        self.task_train_data = self._split_tasks(self.x_train, self.y_train)
        self.task_test_data = self._split_tasks(self.x_test, self.y_test)
    def _split_tasks(self, x, y):
        tasks = {}
        for task_start in range(0, 18, self.classes_per_task):
            class_range = torch.arange(task_start, task_start + self.classes_per_task, device=y.device)
            idx = torch.isin(y, class_range)
            tasks[task_start // self.classes_per_task] = (x[idx], y[idx])
        return tasks
    def _remap_local(self, y, task_id):
        return y - task_id * self.classes_per_task
    def train_task(self, task_id, epochs):
        x_task, y_task = self.task_train_data[task_id]
        y_local = self._remap_local(y_task, task_id)
        loader = DataLoader(TensorDataset(x_task, y_local), batch_size=self.args.batch_size, shuffle=True)
        params = [
            {"params": self.model.classifier.parameters(), "lr": self.args.lr},
            {"params": self.model.l2prompt.parameters(), "lr": self.args.lr * 0.1},
            {"params": [self.model.task_keys[task_id]], "lr": self.args.lr * 5.0},
            {"params": [self.model.task_predictor.task_keys[task_id], self.model.task_predictor.temperature], "lr": self.args.lr * 2.0},
        ]
        optimizer = optim.AdamW(params)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            meter = AverageMeter()
            for xb, yb in DataLoader(loader.dataset, batch_size=self.args.batch_size, shuffle=True):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits, _, topk = self.model(xb, task_id=task_id, predict_task=False)
                loss = criterion(logits, yb)
                loss.backward()
                isolate_gradients(self.model, topk)
                optimizer.step()
                meter.update(loss.item(), yb.size(0))
    @torch.no_grad()
    def evaluate_all_tasks(self, up_to_task, use_oracle=True, verbose=False):
        soft_accs = {}
        task_pred_accs = {}
        oracle_accs = {}
        self.model.eval()
        for tid in range(up_to_task):
            x_eval, y_eval = self.task_test_data[tid]
            y_local = self._remap_local(y_eval, tid)
            loader = DataLoader(TensorDataset(x_eval, y_local), batch_size=self.args.batch_size, shuffle=False)
            correct = 0
            total = 0
            task_correct = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                if use_oracle:
                    logits, _, _ = self.model(xb, task_id=tid, predict_task=False)
                    preds = logits.argmax(dim=1)
                    correct += preds.eq(yb).sum().item()
                    total += yb.size(0)
                else:
                    logits, pred_tid_scalar, _ = self.model(xb, task_id=None, predict_task=True)
                    preds = logits.argmax(dim=1)
                    correct += preds.eq(yb).sum().item()
                    total += yb.size(0)
                    task_correct += (pred_tid_scalar == tid) * yb.size(0)
            acc = 100.0 * correct / max(1, total)
            if use_oracle:
                oracle_accs[tid] = acc
            else:
                soft_accs[tid] = acc
                task_pred_accs[tid] = 100.0 * task_correct / max(1, total)
        if use_oracle:
            return oracle_accs
        return soft_accs, task_pred_accs
    def _print_final_summary(self):
        pass
    def _save_results(self):
        pass
    def run(self):
        initial_oracle_accs = {}
        initial_soft_accs = {}
        for task_id in range(self.n_tasks):
            self.train_task(task_id, self.args.epochs_per_task)
            print(f"\n{'='*80}")
            print(f"EVALUATION AFTER TASK {task_id + 1}")
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
            soft_accs, task_pred_accs = self.evaluate_all_tasks(up_to_task=task_id + 1, use_oracle=False, verbose=True)
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
                print(f"\n--- Forgetting Analysis ---")
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
                save_checkpoint({'task_id': task_id,'model_state_dict': self.model.state_dict(),'results': self.results}, checkpoint_path)
        self._print_final_summary()
        self._save_results()
        return self.results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str, required=True)
    parser.add_argument('--prompt_length', type=int, default=5)
    parser.add_argument('--epochs_per_task', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pool_size', type=int, default=20)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--save_checkpoints', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    args = parser.parse_args()
    trainer = ContinualLearningTrainer(args)
    print(f"Train shape: {trainer.x_train.shape}, Test shape: {trainer.x_test.shape}")
    trainer.run()

if __name__ == "__main__":
    main()
