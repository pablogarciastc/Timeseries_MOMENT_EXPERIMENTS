
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_task_predictor import AttentionTaskPredictor, HybridTaskPredictor


class MultiHeadClassifier(nn.Module):

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

    def __init__(
        self,
        n_tasks,
        classes_per_task=3,
        pool_size=10,
        prompt_length=5,
        top_k=5,
        moment_model='small',
        use_g_prompt=True,
        use_e_prompt=True,
        task_predictor_type='attention'
    ):
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
            d_model = 512

        from coda_prompt import CODAPromptPool
        self.coda_prompt = CODAPromptPool(
            n_tasks=n_tasks,
            pool_size=pool_size,
            prompt_length=prompt_length,
            d_model=d_model,
            top_k=top_k,
            use_g_prompt=use_g_prompt,
            use_e_prompt=use_e_prompt
        )

        print(f"\nUsing {task_predictor_type} task predictor...")
        if task_predictor_type == 'attention':
            self.task_predictor = AttentionTaskPredictor(
                n_tasks=n_tasks,
                d_model=d_model,
                n_heads=8,
                dropout=0.1
            )
        elif task_predictor_type == 'hybrid':
            self.task_predictor = HybridTaskPredictor(
                n_tasks=n_tasks,
                d_model=d_model,
                n_heads=4
            )
        else:
            raise ValueError(f"Unknown task_predictor_type: {task_predictor_type}")

        self.classifier = MultiHeadClassifier(
            n_tasks=n_tasks,
            classes_per_task=classes_per_task,
            d_model=d_model
        )

        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task
        self.d_model = d_model

    def freeze_task(self, task_id):
        for param in self.classifier.heads[task_id].parameters():
            param.requires_grad = False

        self.coda_prompt.freeze_g_prompt(task_id)

        print(f"✓ Task {task_id} frozen (classifier + G-Prompt)")

    def forward(self, x_enc, task_id=None, predict_task=False, return_task_info=False):
        bsz, n_channels, seq_len = x_enc.shape
        device = x_enc.device

        input_mask = torch.ones((bsz, seq_len), device=device)
        x_norm = self.moment.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_norm = torch.nan_to_num(x_norm, nan=0, posinf=0, neginf=0)
        patches = self.moment.tokenizer(x=x_norm)
        enc_in = self.moment.patch_embedding(patches, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(bsz * n_channels, n_patches, self.d_model)

        with torch.no_grad():
            self.moment.eval()
            q_out = self.moment.encoder(inputs_embeds=enc_in)
            q_x = q_out.last_hidden_state.mean(dim=1)

        selected_prompts = self.coda_prompt(q_x, task_id=task_id)

        x_with_prompts = torch.cat([selected_prompts, enc_in], dim=1)

        with torch.no_grad():
            self.moment.eval()
            outputs = self.moment.encoder(inputs_embeds=x_with_prompts)

        hidden_states = outputs.last_hidden_state

        pooled = hidden_states.mean(dim=1)
        pooled = pooled.view(bsz, n_channels, -1).mean(dim=1)

        training_mode = task_id is not None
        task_info = self.task_predictor(pooled, training=training_mode, task_id=task_id)
        predicted_task_id = task_info['predicted_task']

        logits = self.classifier(pooled, predicted_task_id)

        if return_task_info:
            return logits, task_info
        return logits

    def forward_with_task_loss(self, x_enc, labels, task_id):
        bsz, n_channels, seq_len = x_enc.shape
        device = x_enc.device

        input_mask = torch.ones((bsz, seq_len), device=device)
        x_norm = self.moment.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_norm = torch.nan_to_num(x_norm, nan=0, posinf=0, neginf=0)
        patches = self.moment.tokenizer(x=x_norm)
        enc_in = self.moment.patch_embedding(patches, mask=input_mask)
        n_patches = enc_in.shape[2]

        enc_in = enc_in.reshape(bsz * n_channels, n_patches, self.d_model)

        with torch.no_grad():
            self.moment.eval()
            q_out = self.moment.encoder(inputs_embeds=enc_in)
            q_x = q_out.last_hidden_state.mean(dim=1)

        selected_prompts, selection_info = self.coda_prompt(
            q_x, task_id=task_id, return_selection_info=True
        )
        x_with_prompts = torch.cat([selected_prompts, enc_in.detach()], dim=1)

        self.moment.eval()
        outputs = self.moment.encoder(inputs_embeds=x_with_prompts)
        hidden_states = outputs.last_hidden_state

        pooled = hidden_states.mean(dim=1)
        pooled = pooled.view(bsz, n_channels, -1).mean(dim=1)

        task_info = self.task_predictor(pooled, training=True, task_id=task_id)

        if selection_info and 'diversity_loss' in selection_info:
            task_info['diversity_loss'] = selection_info['diversity_loss']

        logits = self.classifier(pooled, task_id)

        if isinstance(task_id, int):
            task_targets = torch.full((bsz,), task_id, dtype=torch.long, device=device)
        else:
            task_targets = task_id

        task_loss = F.cross_entropy(task_info['task_logits'], task_targets)

        if hasattr(self.task_predictor, 'compute_separation_loss'):
            sep_loss = self.task_predictor.compute_separation_loss()
            task_loss = task_loss + 0.1 * sep_loss
        elif hasattr(self.task_predictor, 'compute_prototype_separation_loss'):
            sep_loss = self.task_predictor.compute_prototype_separation_loss()
            task_loss = task_loss + 0.1 * sep_loss

        _, task_preds = task_info['task_logits'].max(dim=-1)
        task_acc = task_preds.eq(task_targets).float().mean()

        return logits, task_loss, task_acc, task_info