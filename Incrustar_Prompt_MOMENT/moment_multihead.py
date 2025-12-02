"""
MOMENT + L2Prompt + Multi-Head Classifier + Soft Task-ID Prediction

Combines:
- ZERO forgetting (isolated multi-head classifier)
- Task-agnostic inference (soft task-ID prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskPredictor(nn.Module):
    """
    Predicts task-ID from features using learnable task keys.
    Each task has an independent nn.Parameter that can be frozen.
    """

    def __init__(self, n_tasks, d_model):
        super().__init__()
        self.n_tasks = n_tasks
        self.d_model = d_model

        # Use ParameterList so we can freeze individual task keys
        self.task_keys = nn.ParameterList([
            nn.Parameter(torch.randn(d_model)) for _ in range(n_tasks)
        ])

        # Initialize orthogonally for better separation
        for i, key in enumerate(self.task_keys):
            nn.init.normal_(key, mean=0, std=0.02)

        # Temperature for softmax
        self.temperature = nn.Parameter(torch.ones(1))

        print(f"TaskPredictor: {n_tasks} tasks, {d_model} dims")

    def l2_normalize(self, x, dim=-1, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def freeze_task_key(self, task_id):
        """Freeze the key for a specific task"""
        self.task_keys[task_id].requires_grad = False

    def unfreeze_task_key(self, task_id):
        """Unfreeze the key for a specific task"""
        self.task_keys[task_id].requires_grad = True

    def forward(self, x, training=False, task_id=None):
        """
        Args:
            x: features [batch, d_model]
            training: if True and task_id provided, returns task_id (supervised)
            task_id: ground truth task_id (only used during training)

        Returns:
            dict with predicted_task, task_logits, task_probs
        """
        batch_size = x.size(0)

        # Stack all task keys into a matrix
        task_keys_matrix = torch.stack([key for key in self.task_keys], dim=0)  # [n_tasks, d_model]

        # Normalize
        x_norm = self.l2_normalize(x, dim=-1)
        task_keys_norm = self.l2_normalize(task_keys_matrix, dim=-1)

        # Compute similarities
        similarity = torch.matmul(x_norm, task_keys_norm.t())  # [batch, n_tasks]
        task_logits = similarity / self.temperature
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
            'similarities': similarity
        }


class MultiHeadClassifier(nn.Module):
    """Multiple classifier heads - one per task"""

    def __init__(self, n_tasks, classes_per_task, d_model, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, classes_per_task)
            ) for _ in range(n_tasks)
        ])

        print(f"MultiHeadClassifier: {n_tasks} tasks, {classes_per_task} classes each")

    def forward(self, x, task_id):
        if isinstance(task_id, int):
            return self.heads[task_id](x)
        else:
            batch_size = x.size(0)
            logits = torch.zeros(batch_size, self.classes_per_task, device=x.device)
            for tid in range(self.n_tasks):
                mask = task_id == tid
                if mask.any():
                    logits[mask] = self.heads[tid](x[mask])
            return logits



class PromptedMOMENT(nn.Module):
    def __init__(self, n_tasks, pool_size=20, prompt_length=5, top_k=5, classes_per_task=3, moment_model='small'):
        super().__init__()
        try:
            from momentfm import MOMENTPipeline

            model_name = f"AutonLab/MOMENT-1-{moment_model}"
            print(f"Loading {model_name}...")

            self.moment = MOMENTPipeline.from_pretrained(
                model_name,
                model_kwargs={
                    'task_name': 'reconstruction',
                    'enable_gradient_checkpointing': False,
                }
            )
            self.moment.init()

            if hasattr(self.moment, 'config'):
                moment_d_model = self.moment.config.d_model
            else:
                moment_d_model = 512 if moment_model == 'small' else 1024

            d_model = moment_d_model

            for param in self.moment.parameters():
                param.requires_grad = False
            print(f"MOMENT frozen, d_model={d_model}")

        except ImportError:
            print("WARNING: MOMENT not available")
            self.moment = nn.Identity()
            d_model = d_model or 512

        from l2prompt_faithful import L2PromptPool
        self.l2prompt = L2PromptPool(
            pool_size=pool_size,
            prompt_length=prompt_length,
            d_model=d_model,
            top_k=top_k
        )

        self.task_predictor = TaskPredictor(n_tasks=n_tasks, d_model=d_model)
        self.classifier = MultiHeadClassifier(
            n_tasks=n_tasks,
            classes_per_task=classes_per_task,
            d_model=d_model
        )
        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task


    def freeze_task(self, task_id):
        """Freeze all parameters for a specific task"""
        # Freeze classifier head
        for param in self.classifier.heads[task_id].parameters():
            param.requires_grad = False

        # Freeze task key
        self.task_predictor.freeze_task_key(task_id)

        print(f"Task {task_id} frozen")


    def forward(self, x_enc, task_id=None, predict_task=False, return_task_info=False):


        bsz, n_channels, seq_len = x_enc.shape
        device = x_enc.device
        input_mask = torch.ones((bsz, seq_len), device=device).to(x_enc.device)

        x_norm = self.moment.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_norm = torch.nan_to_num(x_norm, nan=0, posinf=0, neginf=0)
        patches = self.moment.tokenizer(x=x_norm)
        enc_in = self.moment.patch_embedding(patches, mask=input_mask)


        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(bsz * n_channels, n_patches, self.moment.config.d_model)

        with torch.no_grad():
            self.moment.eval()
            q_out = self.moment.encoder(inputs_embeds=enc_in)
            q_x = q_out.last_hidden_state.mean(dim=1)

        selected_prompts, _ = self.l2prompt.select_prompts_from_query(q_x)
        x_with_prompts = torch.cat([selected_prompts, enc_in], dim=1)


        # Encode
        with torch.no_grad():
            self.moment.eval()
            outputs = self.moment.encoder(inputs_embeds=x_with_prompts)

        hidden_states = outputs.last_hidden_state

        # Pool features
        pooled = hidden_states.mean(dim=1)
        pooled = pooled.view(bsz, n_channels, -1).mean(dim=1)  # [bsz, d_model]


        training_mode = task_id is not None
        task_info = self.task_predictor(pooled, training=training_mode, task_id=task_id)
        predicted_task_id = task_info['predicted_task']

        logits = self.classifier(pooled, predicted_task_id)

        if return_task_info:
            return logits, task_info
        return logits

    def forward_with_task_loss(self, x_enc, labels, task_id):
        """
        Forward with task prediction loss

        Args:
            x_enc: input [batch, channels, seq_len]
            labels: ground truth labels [batch]
            task_id: task identifier (int)

        Returns:
            logits, task_loss, task_acc, task_info
        """

        bsz, n_channels, seq_len = x_enc.shape
        device = x_enc.device

        # Normalize and tokenize
        input_mask = torch.ones((bsz, seq_len), device=device)
        x_norm = self.moment.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_norm = torch.nan_to_num(x_norm, nan=0, posinf=0, neginf=0)
        patches = self.moment.tokenizer(x=x_norm)
        enc_in = self.moment.patch_embedding(patches, mask=input_mask)
        n_patches = enc_in.shape[2]

        # Reshape for processing
        enc_in = enc_in.reshape(bsz * n_channels, n_patches, self.moment.config.d_model)

        with torch.no_grad():
            self.moment.eval()
            q_out = self.moment.encoder(inputs_embeds=enc_in)
            q_x = q_out.last_hidden_state.mean(dim=1)

        selected_prompts, _ = self.l2prompt.select_prompts_from_query(q_x)
        x_with_prompts = torch.cat([selected_prompts, enc_in.detach()], dim=1)


        # Encode
        self.moment.eval()
        outputs = self.moment.encoder(inputs_embeds=x_with_prompts)

        hidden_states = outputs.last_hidden_state

        # Pool features
        pooled = hidden_states.mean(dim=1)
        pooled = pooled.view(bsz, n_channels, -1).mean(dim=1)  # [bsz, d_model]
        # Task prediction
        task_info = self.task_predictor(pooled, training=True, task_id=task_id)

        # Classification with oracle task_id
        logits = self.classifier(pooled, task_id)

        # Task loss - USE CORRECT BATCH SIZE
        if isinstance(task_id, int):
            task_targets = torch.full((bsz,), task_id, dtype=torch.long, device=device)
        else:
            task_targets = task_id

        task_loss = F.cross_entropy(task_info['task_logits'], task_targets)

        # Task accuracy
        _, task_preds = task_info['task_logits'].max(dim=-1)
        task_acc = task_preds.eq(task_targets).float().mean()

        return logits, task_loss, task_acc, task_info
