import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from l2prompt_faithful import L2PromptPool


class TextEmbedderLETS(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-13b-hf", max_tokens=4096, precision=2):
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
        scale = 10 ** self.precision
        channel_texts = []
        for ch_id, channel in enumerate(seq):
            channel_scaled = torch.round(channel * scale).abs().long().tolist()
            str_values = []
            for v in channel_scaled:
                digits = list(str(int(v)))
                spaced = " ".join(digits)
                str_values.append(spaced)
            joined = " , ".join(str_values)
            channel_texts.append(f"{joined} | Channel {ch_id + 1} :")
        return " ".join(channel_texts).strip()

    def forward(self, x):
        # Normalizar a [0,1] por canal
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
        batch_size = x.size(0)
        task_keys_matrix = torch.stack([key for key in self.task_keys], dim=0)
        x_norm = self.l2_normalize(x, dim=-1)
        task_keys_norm = self.l2_normalize(task_keys_matrix, dim=-1)
        similarity = torch.matmul(x_norm, task_keys_norm.t())
        task_logits = similarity / self.temperature
        task_probs = F.softmax(task_logits, dim=-1)

        if training and task_id is not None:
            # supervision handled externally, no override here
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
        print(f"MultiHeadClassifier: {n_tasks} tasks, {classes_per_task} classes each")

    def forward(self, x, task_id):
        if isinstance(task_id, int):
            return self.heads[task_id](x)
        batch_size = x.size(0)
        logits = torch.zeros(batch_size, self.classes_per_task, device=x.device)
        for tid in range(self.n_tasks):
            mask = task_id == tid
            if mask.any():
                logits[mask] = self.heads[tid](x[mask])
        return logits  # ✓ CORREGIDO


class PromptedMOMENT(nn.Module):
    def __init__(self, n_tasks, pool_size=20, prompt_length=5, top_k=5, classes_per_task=3, moment_model='small'):
        super().__init__()

        # 1. Text Encoder (CONGELADO)
        self.text_encoder = TextEmbedderLETS(model_name="meta-llama/Llama-2-7b-hf")

        for p in self.text_encoder.model.parameters():
            p.requires_grad = False


        print("✓ Text encoder frozen (LLaMA + series_proj)")

        # 2. MOMENT (CONGELADO)
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

            for param in self.moment.parameters():
                param.requires_grad = False

            moment_d_model = self.moment.config.d_model
            print(f"✓ MOMENT frozen, d_model={moment_d_model}")

        except ImportError:
            print("WARNING: MOMENT not available")
            self.moment = nn.Identity()
            moment_d_model = 512

        # 3. Projection (TRAINABLE - necesario para adaptar dimensiones)
        self.proj_to_moment = nn.Linear(self.text_encoder.d_model, moment_d_model)


        print(f"✓ proj_to_moment: {self.text_encoder.d_model} → {moment_d_model} (trainable)")

        # 4. L2Prompt (TRAINABLE)
        self.l2prompt = L2PromptPool(
            pool_size=pool_size,
            prompt_length=prompt_length,
            d_model=moment_d_model,
            top_k=top_k
        )
        print(f"✓ L2Prompt: pool_size={pool_size}, prompt_length={prompt_length}, top_k={top_k}")

        # 5. Task Predictor (TRAINABLE)
        self.task_predictor = TaskPredictor(n_tasks=n_tasks, d_model=moment_d_model)
        self.pooled_norm = nn.LayerNorm(moment_d_model)
        # 6. Classifier (TRAINABLE)
        self.classifier = MultiHeadClassifier(
            n_tasks=n_tasks,
            classes_per_task=classes_per_task,
            d_model=moment_d_model
        )

        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task
        self.d_model = moment_d_model

    def freeze_task(self, task_id):
        """Freeze classifier head and task key for a specific task"""
        for param in self.classifier.heads[task_id].parameters():
            param.requires_grad = False
        self.task_predictor.freeze_task_key(task_id)
        print(f"🔒 Task {task_id} frozen (classifier + task_key)")

    def forward(self, x_enc, labels=None, task_id=None, compute_task_loss=False):
        """
        Unificado: entrenamiento + inferencia

        Args:
            x_enc: tensor de entrada [B, seq_len, feats]
            labels: etiquetas de clase (solo en entrenamiento)
            task_id: id de tarea (entrenamiento supervisado o inferencia)
            compute_task_loss: si True, calcula task_loss y task_acc
        Returns:
            - Si compute_task_loss=False → (logits, task_info)
            - Si compute_task_loss=True → (logits, total_loss, task_acc, task_info)
        """
        bsz = x_enc.size(0)
        device = x_enc.device

        # === 1. Text encoder (congelado) ===
        with torch.no_grad():
            base_feats = self.text_encoder(x_enc)  # [B, text_d_model]

        # === 2. Proyección a espacio MOMENT ===
        base_feats_proj = self.proj_to_moment(base_feats)
        base_feats_moment = base_feats_proj.unsqueeze(1)

        # === 3. MOMENT encoder (congelado) ===
        with torch.no_grad():
            self.moment.eval()
            outputs = self.moment.encoder(inputs_embeds=base_feats_moment)
            hidden_states = outputs.last_hidden_state
            pooled_frozen = hidden_states.mean(dim=1)

        # === 4. Prompts ===
        query = pooled_frozen
        selected_prompts, _ = self.l2prompt.select_prompts_from_query(query.unsqueeze(1))
        prompted = torch.cat([selected_prompts, base_feats_proj.detach().unsqueeze(1)], dim=1)
        
        with torch.no_grad():
            self.moment.eval()
            outputs = self.moment.encoder(inputs_embeds=prompted)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1)

        # === 5. Task prediction ===
        task_info = self.task_predictor(pooled, training=self.training, task_id=task_id)

        # === 6. Clasificación ===
        logits = self.classifier(pooled, task_id if task_id is not None else task_info['predicted_task'])

        # === 7. Task loss opcional ===
        if compute_task_loss and task_id is not None:
            task_targets = torch.full((bsz,), task_id, dtype=torch.long, device=device)
            task_loss = F.cross_entropy(task_info['task_logits'], task_targets)
            _, task_preds = task_info['task_logits'].max(dim=-1)
            task_acc = task_preds.eq(task_targets).float().mean()
            return logits, task_loss, task_acc, task_info

        return logits, task_info
