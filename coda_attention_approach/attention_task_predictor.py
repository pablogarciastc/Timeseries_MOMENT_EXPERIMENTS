"""
Attention-Based Task Predictor for MOMENT's Collapsed Feature Space

Solves the query-key matching collapse problem by:
1. Using cross-attention instead of simple cosine similarity
2. Learning rich task prototypes instead of simple key vectors
3. Multi-head attention to capture different discriminative aspects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionTaskPredictor(nn.Module):

    def __init__(self, n_tasks, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_tasks = n_tasks
        self.d_model = d_model
        self.n_heads = n_heads

        self.task_prototypes = nn.Parameter(
            torch.randn(n_tasks, d_model)
        )
        nn.init.orthogonal_(self.task_prototypes)

        self.feature_projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_tasks)
        )

        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        print(f"AttentionTaskPredictor: {n_tasks} tasks, {n_heads} heads, {d_model} dims")

    def freeze_task_prototype(self, task_id):
        pass

    def forward(self, x, training=False, task_id=None):
        batch_size = x.size(0)

        x_proj = self.feature_projector(x)

        query = x_proj.unsqueeze(1)

        prototypes = self.task_prototypes.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        attended_features, attn_weights = self.cross_attention(
            query=query,
            key=prototypes,
            value=prototypes
        )

        attended_features = attended_features.squeeze(1)
        attn_weights = attn_weights.squeeze(1)

        task_logits = self.classifier(attended_features)

        task_logits = task_logits / self.temperature.clamp(min=0.1, max=2.0)
        task_probs = F.softmax(task_logits, dim=-1)

        if training and task_id is not None:
            predicted_task = task_id
        else:
            _, pred_indices = task_logits.max(dim=-1)
            if batch_size > 1:
                predicted_task = torch.mode(pred_indices).values.item()
            else:
                predicted_task = pred_indices[0].item()

        return {
            'predicted_task': predicted_task,
            'task_logits': task_logits,
            'task_probs': task_probs,
            'attn_weights': attn_weights,
            'attended_features': attended_features
        }

    def compute_prototype_separation_loss(self):
        prototypes_norm = F.normalize(self.task_prototypes, p=2, dim=1)
        similarity = torch.matmul(prototypes_norm, prototypes_norm.t())

        mask = torch.eye(self.n_tasks, device=similarity.device)
        off_diagonal = similarity * (1 - mask)

        separation_loss = off_diagonal.abs().mean()

        return separation_loss


class HybridTaskPredictor(nn.Module):

    def __init__(self, n_tasks, d_model, n_heads=4):
        super().__init__()
        self.n_tasks = n_tasks
        self.d_model = d_model

        self.attention_predictor = AttentionTaskPredictor(
            n_tasks, d_model, n_heads=n_heads, dropout=0.1
        )

        self.distance_projector = nn.Sequential(
            nn.Linear(d_model, d_model * 3),
            nn.LayerNorm(d_model * 3),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(d_model * 3, d_model)
        )

        self.distance_keys = nn.ParameterList([
            nn.Parameter(torch.randn(d_model)) for _ in range(n_tasks)
        ])

        if n_tasks <= d_model:
            keys_init = torch.nn.init.orthogonal_(torch.empty(d_model, n_tasks))
            for i in range(n_tasks):
                self.distance_keys[i].data = keys_init[:, i].clone()

        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        print(f"HybridTaskPredictor: attention + distance ensemble")

    def forward(self, x, training=False, task_id=None):
        batch_size = x.size(0)

        attn_output = self.attention_predictor(x, training, task_id)
        attn_logits = attn_output['task_logits']

        x_dist = self.distance_projector(x)
        keys_matrix = torch.stack([k for k in self.distance_keys], dim=0)

        diff = x_dist.unsqueeze(1) - keys_matrix.unsqueeze(0)
        distances = (diff ** 2).sum(dim=-1)
        dist_logits = -distances

        alpha = torch.sigmoid(self.fusion_weight)
        task_logits = alpha * attn_logits + (1 - alpha) * dist_logits
        task_probs = F.softmax(task_logits, dim=-1)

        if training and task_id is not None:
            predicted_task = task_id
        else:
            _, pred_indices = task_logits.max(dim=-1)
            if batch_size > 1:
                predicted_task = torch.mode(pred_indices).values.item()
            else:
                predicted_task = pred_indices[0].item()

        return {
            'predicted_task': predicted_task,
            'task_logits': task_logits,
            'task_probs': task_probs,
            'attn_weights': attn_output['attn_weights'],
            'fusion_weight': alpha.item()
        }

    def compute_separation_loss(self):
        attn_loss = self.attention_predictor.compute_prototype_separation_loss()

        keys = torch.stack([k for k in self.distance_keys], dim=0)
        keys_norm = F.normalize(keys, dim=1)
        similarity = keys_norm @ keys_norm.t()
        mask = torch.eye(self.n_tasks, device=similarity.device)
        dist_loss = (similarity * (1 - mask)).abs().mean()

        return 0.5 * (attn_loss + dist_loss)