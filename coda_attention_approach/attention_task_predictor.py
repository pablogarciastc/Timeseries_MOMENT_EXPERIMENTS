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
    """
    Task predictor using cross-attention mechanism

    Instead of: similarity = normalize(features) @ normalize(keys)
    We use: attended_features = MultiHeadAttention(features, task_prototypes)

    This is more robust to MOMENT's collapsed feature geometry.
    """

    def __init__(self, n_tasks, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_tasks = n_tasks
        self.d_model = d_model
        self.n_heads = n_heads

        # Task prototypes (learnable, richer than simple keys)
        # Each task gets a full embedding that can capture complex patterns
        self.task_prototypes = nn.Parameter(
            torch.randn(n_tasks, d_model)
        )
        nn.init.orthogonal_(self.task_prototypes)

        # Feature projection to expand collapsed MOMENT space
        self.feature_projector = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Multi-head cross-attention
        # Query: input features
        # Key/Value: task prototypes
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_tasks)
        )

        # Temperature for final logits
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

        print(f"AttentionTaskPredictor: {n_tasks} tasks, {n_heads} heads, {d_model} dims")

    def freeze_task_prototype(self, task_id):
        """Freeze prototype for a specific task"""
        # Note: Can't freeze individual parameters in a tensor
        # Instead, we'll mask gradients in training
        pass

    def forward(self, x, training=False, task_id=None):
        """
        Args:
            x: features [batch, d_model]
            training: if True and task_id provided, uses ground truth
            task_id: ground truth task_id (only during training)

        Returns:
            dict with predicted_task, task_logits, task_probs, attention_weights
        """
        batch_size = x.size(0)

        # Project features to untangle MOMENT's collapsed space
        x_proj = self.feature_projector(x)  # [batch, d_model]

        # Prepare for attention
        # Query: projected features
        # Key/Value: task prototypes
        query = x_proj.unsqueeze(1)  # [batch, 1, d_model]

        # Expand prototypes for batch
        prototypes = self.task_prototypes.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, n_tasks, d_model]

        # Cross-attention: each sample attends to all task prototypes
        attended_features, attn_weights = self.cross_attention(
            query=query,
            key=prototypes,
            value=prototypes
        )  # [batch, 1, d_model], [batch, 1, n_tasks]

        attended_features = attended_features.squeeze(1)  # [batch, d_model]
        attn_weights = attn_weights.squeeze(1)  # [batch, n_tasks]

        # Classify based on attended features
        task_logits = self.classifier(attended_features)  # [batch, n_tasks]

        # Apply temperature
        task_logits = task_logits / self.temperature.clamp(min=0.1, max=2.0)
        task_probs = F.softmax(task_logits, dim=-1)

        # Prediction
        if training and task_id is not None:
            predicted_task = task_id
        else:
            _, pred_indices = task_logits.max(dim=-1)
            if batch_size > 1:
                # Use mode (most common prediction in batch)
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
        """
        Encourage task prototypes to be well-separated
        Similar to contrastive learning
        """
        prototypes_norm = F.normalize(self.task_prototypes, p=2, dim=1)
        similarity = torch.matmul(prototypes_norm, prototypes_norm.t())

        # Mask diagonal
        mask = torch.eye(self.n_tasks, device=similarity.device)
        off_diagonal = similarity * (1 - mask)

        # Penalize high similarities
        separation_loss = off_diagonal.abs().mean()

        return separation_loss


class HybridTaskPredictor(nn.Module):
    """
    Hybrid approach: Attention + Euclidean distance

    Combines:
    1. Attention-based prototype matching
    2. Euclidean distance in expanded space
    3. Ensemble prediction
    """

    def __init__(self, n_tasks, d_model, n_heads=4):
        super().__init__()
        self.n_tasks = n_tasks
        self.d_model = d_model

        # Path 1: Attention-based
        self.attention_predictor = AttentionTaskPredictor(
            n_tasks, d_model, n_heads=n_heads, dropout=0.1
        )

        # Path 2: Distance-based with expansion
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

        # Initialize keys orthogonally
        if n_tasks <= d_model:
            keys_init = torch.nn.init.orthogonal_(torch.empty(d_model, n_tasks))
            for i in range(n_tasks):
                self.distance_keys[i].data = keys_init[:, i].clone()

        # Fusion weights (learnable)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        print(f"HybridTaskPredictor: attention + distance ensemble")

    def forward(self, x, training=False, task_id=None):
        batch_size = x.size(0)

        # Path 1: Attention-based prediction
        attn_output = self.attention_predictor(x, training, task_id)
        attn_logits = attn_output['task_logits']

        # Path 2: Distance-based prediction
        x_dist = self.distance_projector(x)
        keys_matrix = torch.stack([k for k in self.distance_keys], dim=0)

        # Euclidean distances
        diff = x_dist.unsqueeze(1) - keys_matrix.unsqueeze(0)
        distances = (diff ** 2).sum(dim=-1)
        dist_logits = -distances  # Negative distance as logits

        # Ensemble (learnable fusion)
        alpha = torch.sigmoid(self.fusion_weight)
        task_logits = alpha * attn_logits + (1 - alpha) * dist_logits
        task_probs = F.softmax(task_logits, dim=-1)

        # Prediction
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
        """Combined separation loss for both paths"""
        # Attention prototypes
        attn_loss = self.attention_predictor.compute_prototype_separation_loss()

        # Distance keys
        keys = torch.stack([k for k in self.distance_keys], dim=0)
        keys_norm = F.normalize(keys, dim=1)
        similarity = keys_norm @ keys_norm.t()
        mask = torch.eye(self.n_tasks, device=similarity.device)
        dist_loss = (similarity * (1 - mask)).abs().mean()

        return 0.5 * (attn_loss + dist_loss)