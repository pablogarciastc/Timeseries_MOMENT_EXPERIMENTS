import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from coda_prompt import CODAPromptPool


class StaticClassStatistics(nn.Module):
    def __init__(self, bottleneck_dim=512):
        super().__init__()

        self.class_stats = {
            0: [0.325, 0.001, 0.0, 1.0, 1.0, 5.0, -0.866, 0.345, 0.823],
            1: [0.246, 0.000, 0.0, 0.5, 1.0, 2.0, -0.106, 0.315, 0.601],
            2: [0.202, 0.001, 0.0, 1.0, 1.0, 2.0, 1.025, 0.448, 0.859],
            3: [-0.109, 0.000, 0.0, 0.5, 1.0, 3.0, -0.667, 0.447, 0.672],
            4: [0.219, 0.031, 0.5, 1.0, 1.0, 25.0, -1.228, 0.039, 0.959],
            5: [0.233, 0.034, 0.5, 1.0, 1.0, 25.0, -1.364, 0.238, 0.941],
            6: [0.248, 0.001, 0.0, 1.0, 1.0, 1.0, 0.262, -0.198, 0.820],
            7: [0.156, 0.007, 0.0, 1.0, 1.0, 1.0, 0.371, -0.186, 0.735],
            8: [0.195, 0.012, 0.25, 0.5, 1.0, 0.0, 0.431, -0.179, 0.371],
            9: [0.188, 0.009, 0.0, 0.5, 28.0, 0.0, 0.050, -0.564, 0.190],
            10: [0.129, 0.013, 0.5, 0.5, 31.0, 0.0, -0.184, -0.609, 0.562],
            11: [-0.205, 0.025, 0.5, 0.5, 24.0, 0.0, -0.160, -0.252, 0.315],
            12: [0.130, 0.009, 0.25, 0.5, 1.0, 0.0, 0.031, -0.675, 0.471],
            13: [-0.175, 0.012, 0.25, 0.5, 1.0, 0.0, -0.031, -0.777, 0.621],
            14: [0.607, 0.007, 0.0, 1.0, 1.0, 1.0, 0.298, -0.170, 0.773],
            15: [0.587, 0.011, 0.0, 1.0, 1.0, 0.0, -0.116, -0.436, 0.841],
            16: [0.143, 0.029, 1.0, 0.5, 11.0, 5.0, 0.023, 0.247, 0.169],
            17: [0.040, 0.028, 1.0, 0.5, 1.0, 0.0, -0.166, -0.547, 0.466],
        }

        stats_list = [self.class_stats[i] for i in range(len(self.class_stats))]
        self.stats_tensor = nn.Parameter(
            torch.tensor(stats_list, dtype=torch.float32),
            requires_grad=False
        )

        n_stats = 9

        self.stats_proj = nn.Sequential(
            nn.Linear(n_stats, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        print(f"✅ Static class statistics loaded: 18 classes × {n_stats} features")

    def forward(self, labels):
        batch_stats = self.stats_tensor[labels]
        stats_emb = self.stats_proj(batch_stats)
        return stats_emb


class TextEmbedderLETS(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", max_tokens=1000, precision=2):
        super().__init__()
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"✅ Set pad_token to eos_token: {self.tokenizer.pad_token}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True
        )

        self.d_model = self.model.config.hidden_size
        self.max_tokens = max_tokens
        self.precision = precision
        self.series_proj = nn.Linear(45, self.d_model)


    def format_series_as_text(self, seq):
        channel_texts = []
        for ch_id, channel in enumerate(seq):
            scale = 10 ** self.precision
            channel_scaled = torch.round(channel * scale).long().tolist()
            values_str = " ".join([str(int(v)) for v in channel_scaled[:50]])

            channel_text = (
                f"Channel {ch_id + 1}: "
                f"values: {values_str}"
            )
            channel_texts.append(channel_text)

        return " | ".join(channel_texts)

    def forward(self, x, labels=None):

        x_normalized = torch.zeros_like(x)
        for b in range(x.size(0)):
            for c in range(x.size(1)):
                channel = x[b, c, :]
                min_val = channel.min()
                max_val = channel.max()
                if max_val > min_val:
                    x_normalized[b, c, :] = (channel - min_val) / (max_val - min_val)
                else:
                    x_normalized[b, c, :] = channel

        batch_text = [self.format_series_as_text(seq) for seq in x_normalized]
        inputs = self.tokenizer(
            batch_text, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_tokens
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            outs = self.model(**inputs, output_hidden_states=True)
            e_text = outs.hidden_states[-1].mean(dim=1)

        b, c, t = x.shape
        series_mean = x.mean(dim=1)
        series_emb = self.series_proj(series_mean)
        series_emb = F.normalize(series_emb, dim=-1)

        e_fused = e_text + series_emb
        return e_fused


class TaskPredictor(nn.Module):
    def __init__(self, n_tasks, d_model):
        super().__init__()
        self.n_tasks = n_tasks
        self.d_model = d_model

        self.task_keys = nn.ParameterList([
            nn.Parameter(torch.randn(d_model)) for _ in range(n_tasks)
        ])

        for i, key in enumerate(self.task_keys):
            nn.init.normal_(key, mean=0, std=0.02)

        self.temperature = nn.Parameter(torch.ones(1))

        print(f"TaskPredictor: {n_tasks} tasks, {d_model} dims")

    def l2_normalize(self, x, dim=-1, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def freeze_task_key(self, task_id):
        self.task_keys[task_id].requires_grad = False

    def unfreeze_task_key(self, task_id):
        self.task_keys[task_id].requires_grad = True

    def forward(self, x, training=False, task_id=None):
        batch_size = x.size(0)
        task_keys_matrix = torch.stack([key for key in self.task_keys], dim=0)
        x_norm = self.l2_normalize(x, dim=-1)
        task_keys_norm = self.l2_normalize(task_keys_matrix, dim=-1)
        similarity = torch.matmul(x_norm, task_keys_norm.t())
        task_logits = similarity / self.temperature
        task_probs = F.softmax(task_logits, dim=-1)

        if training and task_id is not None:
            predicted_task = task_logits.argmax(dim=-1)
        else:
            _, pred_indices = task_logits.max(dim=-1)
            predicted_task = torch.mode(pred_indices).values.item() if batch_size > 1 else pred_indices[0].item()

        return {
            'predicted_task': predicted_task,
            'task_logits': task_logits,
            'task_probs': task_probs,
            'similarities': similarity
        }


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

    def forward(self, x, task_id):
        if isinstance(task_id, int):
            return self.heads[task_id](x)
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, self.classes_per_task, device=x.device)
        for tid in range(self.n_tasks):
            mask = task_id == tid
            if mask.any():
                logits[mask] = self.heads[tid](x[mask])
        return logits


class PromptedLETS(nn.Module):
    def __init__(self, n_tasks, classes_per_task=3, pool_size=20, prompt_length=5, top_k=5,
                 use_pca=False, pca_dim=256):
        super().__init__()

        self.text_encoder = TextEmbedderLETS(model_name="meta-llama/Llama-2-7b-hf")
        d_model_original = self.text_encoder.d_model

        for p in self.text_encoder.model.parameters():
            p.requires_grad = False
        for p in self.text_encoder.series_proj.parameters():
            p.requires_grad = False
        print("🧊 Text encoder frozen (LLaMA + series_proj)")

        self.use_pca = use_pca
        if use_pca:
            self.pca_projection = nn.Linear(d_model_original, pca_dim)
            d_model = pca_dim
        else:
            d_model = d_model_original

        self.coda_prompt = CODAPromptPool(
            n_tasks=n_tasks,
            pool_size=pool_size,
            prompt_length=prompt_length,
            d_model=d_model,
            top_k=top_k,
            use_g_prompt=True,
            use_e_prompt=True
        )
        print(f"L2Prompt initialized: pool_size={pool_size}, top_k={top_k}")

        self.task_predictor = TaskPredictor(n_tasks, d_model)
        self.classifier = MultiHeadClassifier(n_tasks, classes_per_task, d_model)

        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task
        self.d_model = d_model

    def freeze_task(self, task_id):
        for p in self.classifier.heads[task_id].parameters():
            p.requires_grad = False
        self.task_predictor.freeze_task_key(task_id)
        print(f"✅ Task {task_id} frozen (classifier + task key)")
        self.coda_prompt.freeze_g_prompt(task_id)


    def forward(self, x_enc, task_id=None, return_task_info=False):
        base_feats = self.text_encoder(x_enc)

        if self.use_pca:
            base_feats = self.pca_projection(base_feats)

        base_feats = base_feats.unsqueeze(1)

        if task_id is None:
            task_info = self.task_predictor(base_feats.mean(dim=1), training=False)
            predicted_task = task_info["predicted_task"]
            task_id = predicted_task

        prompted = self.coda_prompt(base_feats, task_id=task_id)
        prompted_flat = prompted.mean(dim=1)

        if return_task_info:
            task_info = self.task_predictor(prompted_flat, training=False, task_id=task_id)

        logits = self.classifier(prompted_flat, task_id)

        if return_task_info:
            return logits, task_info
        return logits

    def forward_with_task_loss(self, x_enc, labels, task_id):
        base_feats = self.text_encoder(x_enc)

        if self.use_pca:
            base_feats = self.pca_projection(base_feats)

        base_feats = base_feats.unsqueeze(1)

        prompted = self.coda_prompt(base_feats, task_id=task_id)

        prompted_flat = prompted.mean(dim=1)

        task_info = self.task_predictor(prompted_flat, training=True, task_id=task_id)
        logits = self.classifier(prompted_flat, task_id)

        task_targets = torch.full((x_enc.size(0),), task_id, dtype=torch.long, device=base_feats.device)
        task_loss = F.cross_entropy(task_info["task_logits"], task_targets)
        _, task_preds = task_info["task_logits"].max(dim=-1)
        task_acc = task_preds.eq(task_targets).float().mean()

        return logits, task_loss, task_acc, task_info