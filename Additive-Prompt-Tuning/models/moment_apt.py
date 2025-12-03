"""
MOMENT with APT - Following zoo.py structure exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math


class APT_MOMENT(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, ema_coeff, num_layers=8):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.n_tasks = n_tasks
        self.num_layers = num_layers
        self._init_smart(prompt_param)

        self.merge_flag = True
        self.ema_coeff = ema_coeff

        self.prompt_tokens = nn.Parameter(
            torch.zeros(num_layers * 2, emb_d),
            requires_grad=True
        )

        self.register_buffer(
            'global_merged_prompt',
            torch.zeros(num_layers * 2, emb_d)
        )

        trunc_normal_(self.prompt_tokens, std=0.02)

        print(f"✓ APT Prompts initialized")
        print(f"  Shape: {list(self.prompt_tokens.shape)}")
        print(f"  Total parameters: {self.prompt_tokens.numel():,}")
        print(f"  EMA coefficient: {ema_coeff}")

    def merge_prompt(self, prompt1, prompt2):
        print("Merging prompt ... ")
        return prompt1 * self.ema_coeff + prompt2 * (1 - self.ema_coeff)

    def _init_smart(self, prompt_param):

        if isinstance(prompt_param, (list, tuple)) and len(prompt_param) > 0:
            dropout = prompt_param[0]
            if isinstance(dropout, (int, float)):
                self.prompt_dropout_ratio = float(dropout)
            else:
                self.prompt_dropout_ratio = 0.0
        else:
            self.prompt_dropout_ratio = 0.0

        if not (0.0 <= self.prompt_dropout_ratio <= 1.0):
            print(f"Warning: Invalid dropout {self.prompt_dropout_ratio}, setting to 0.0")
            self.prompt_dropout_ratio = 0.0

        self.prompt_dropout = nn.Dropout(self.prompt_dropout_ratio)

    def process_task_count(self):
        self.task_count += 1

    def forward(self, l, x_block, train=False):
        B, _, _ = x_block.shape

        prompt_groups = self.prompt_tokens

        num_heads = 8
        head_dim = self.emb_d // num_heads

        if train or not self.merge_flag:
            P_root_k = prompt_groups[l*2:l*2+1].reshape(num_heads, 1, head_dim).expand(B, num_heads, 1, head_dim)
            P_root_v = prompt_groups[l*2+1:l*2+2].reshape(num_heads, 1, head_dim).expand(B, num_heads, 1, head_dim)
        elif not train and self.merge_flag:
            P_root_k = self.global_merged_prompt[l*2:l*2+1].reshape(num_heads, 1, head_dim).expand(B, num_heads, 1, head_dim)
            P_root_v = self.global_merged_prompt[l*2+1:l*2+2].reshape(num_heads, 1, head_dim).expand(B, num_heads, 1, head_dim)
        else:
            raise ValueError("merge flag and mode err")

        P = [P_root_k, P_root_v]

        return P


def create_apt_attention_forward(original_attn, apt_prompt, layer_idx):
    def apt_attention_forward(hidden_states, *args, **kwargs):
        batch_size, seq_len, d_model = hidden_states.shape

        if hasattr(original_attn, 'q') and hasattr(original_attn, 'k') and hasattr(original_attn, 'v'):
            q = original_attn.q(hidden_states)
            k = original_attn.k(hidden_states)
            v = original_attn.v(hidden_states)

            q_dim = q.shape[-1]
            k_dim = k.shape[-1]
            v_dim = v.shape[-1]

            num_heads = 8
            head_dim_q = q_dim // num_heads
            head_dim_k = k_dim // num_heads
            head_dim_v = v_dim // num_heads

            q = q.view(batch_size, seq_len, num_heads, head_dim_q).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim_k).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim_v).transpose(1, 2)

            train_mode = getattr(apt_prompt, '_current_train_mode', False)
            prompt_list = apt_prompt.forward(layer_idx, hidden_states, train=train_mode)

            if prompt_list is not None and len(prompt_list) == 2:
                pk, pv = prompt_list

                if pk.shape[-1] != head_dim_k:
                    if pk.shape[-1] > head_dim_k:
                        pk = pk[..., :head_dim_k]
                    else:
                        pk = F.pad(pk, (0, head_dim_k - pk.shape[-1]))

                if pv.shape[-1] != head_dim_v:
                    if pv.shape[-1] > head_dim_v:
                        pv = pv[..., :head_dim_v]
                    else:
                        pv = F.pad(pv, (0, head_dim_v - pv.shape[-1]))

                k[:, :, 0:1, :] = k[:, :, 0:1, :] + pk[:, :, 0:1, :]
                v[:, :, 0:1, :] = v[:, :, 0:1, :] + pv[:, :, 0:1, :]

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim_k)
            attn_weights = F.softmax(scores, dim=-1)

            if hasattr(original_attn, 'attn_drop'):
                attn_weights = original_attn.attn_drop(attn_weights)

            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, v_dim)

            if hasattr(original_attn, 'o'):
                attn_output = original_attn.o(attn_output)

            return (attn_output, None, None)
        else:
            return original_attn.original_forward(hidden_states, *args, **kwargs)

    return apt_attention_forward


def moment_1_small_apt(tuning_config=None, **kwargs):
    from momentfm import MOMENTPipeline

    print("=" * 70)
    print("Loading MOMENT-1-small with APT")
    print("=" * 70)

    moment_model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small",
        model_kwargs={
            'task_name': 'reconstruction',
            "n_channels": 45,
            "num_class": 18,
            'enable_gradient_checkpointing': True,
        }
    )

    class MOMENTWithAPT(nn.Module):
        def __init__(self, moment_base, config):
            super().__init__()

            self.moment = moment_base
            self.config = config
            self.d_model = moment_base.config.d_model
            self._device = config._device

            print(f"✓ MOMENT loaded: d_model={self.d_model}")

            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            trunc_normal_(self.cls_token, std=0.02)
            self.proj = nn.Linear(45, self.d_model).to(self._device)

            for p in self.moment.parameters():
                p.requires_grad = False

            if hasattr(self.moment, 'model') and hasattr(self.moment.model, 'encoder'):
                encoder = self.moment.model.encoder
            elif hasattr(self.moment, 'encoder'):
                encoder = self.moment.encoder
            else:
                encoder = self.moment

            self._moment_encoder = encoder

            if hasattr(encoder, 'block'):
                encoder_blocks = list(encoder.block)
            elif hasattr(encoder, 'layers'):
                encoder_blocks = list(encoder.layers)
            else:
                encoder_blocks = []

            self.num_blocks = len(encoder_blocks) if encoder_blocks else 8

            self._attention_blocks = []
            for block in encoder_blocks:
                attn_module = None
                if hasattr(block, 'layer') and len(block.layer) > 0:
                    layer0 = block.layer[0]
                    if hasattr(layer0, 'SelfAttention'):
                        attn_module = layer0.SelfAttention
                elif hasattr(block, 'SelfAttention'):
                    attn_module = block.SelfAttention

                self._attention_blocks.append(attn_module)

            print(f"✓ Encoder: {self.num_blocks} blocks")

            n_tasks = getattr(config, 'n_tasks', 10)
            prompt_param = getattr(config, 'prompt_param', [n_tasks, 0.0])
            ema_coeff = getattr(config, 'ema_coeff', 0.5)

            self.prompt = APT_MOMENT(
                emb_d=self.d_model,
                n_tasks=n_tasks,
                prompt_param=prompt_param,
                ema_coeff=ema_coeff,
                num_layers=self.num_blocks
            )

            self._install_apt()
            print("=" * 70)

        def _install_apt(self):
            for layer_idx, attn_module in enumerate(self._attention_blocks):
                if attn_module is not None:
                    attn_module.forward = create_apt_attention_forward(
                        attn_module, self.prompt, layer_idx
                    )
            print(f"✓ APT installed in {len(self._attention_blocks)} layers")

        def _prepare_inputs(self, x):
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            x = x.to(device=self._device, dtype=torch.float32)

            x = self.proj(x)

            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            return x

        def forward(self, x, prompt=None, train=False):
            self.prompt._current_train_mode = train

            x = self._prepare_inputs(x)

            with torch.set_grad_enabled(train):
                if hasattr(self.moment, 'model') and hasattr(self.moment.model, 'encoder'):
                    moment_output = self.moment.model.encoder(inputs_embeds=x)
                elif hasattr(self.moment, 'encoder'):
                    moment_output = self.moment.encoder(inputs_embeds=x)
                else:
                    moment_output = self.moment(inputs_embeds=x)

                hidden_states = getattr(moment_output, 'last_hidden_state', moment_output)

            return hidden_states

    model = MOMENTWithAPT(moment_model, tuning_config)
    device = tuning_config._device if hasattr(tuning_config, '_device') else torch.device('cuda:0')
    model.moment = model.moment.to(device)

    return model


def vit_apt_moment(out_dim, ema_coeff=0.5, prompt_flag='apt',
                   prompt_param=None, tasks=None):

    class Config:
        def __init__(self):
            self.n_tasks = len(tasks) if tasks else 10

            if prompt_param is None:
                dropout = 0.0
            elif isinstance(prompt_param, list):
                if len(prompt_param) >= 1:
                    dropout = float(prompt_param[0])
                else:
                    dropout = 0.0
            else:
                dropout = 0.0

            self.prompt_param = [dropout, self.n_tasks]
            self.ema_coeff = ema_coeff
            self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config = Config()
    feat_model = moment_1_small_apt(tuning_config=config)

    class MOMENT_APT_Classifier(nn.Module):
        def __init__(self, feat, out_dim):
            super().__init__()
            self.num_classes = out_dim
            self.prompt_flag = 'apt'
            self.task_id = None

            self.feat = feat

            self.last = nn.Linear(feat.d_model, out_dim)
            self.clf_norm = nn.LayerNorm(feat.d_model)

            self.prompt = feat.prompt

            tuned_params = [
                "clf_norm.weight", "clf_norm.bias",
                "prompt.prompt_tokens",
                "last.weight", "last.bias",
            ]

            for name, param in self.named_parameters():
                if name in tuned_params:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            print(f"✓ Classifier: {feat.d_model} -> {out_dim}")
            print(f"✓ Weight normalization: ENABLED")
            print("=" * 70)

        def forward(self, x, train=False):
            if self.prompt is not None:
                if self.prompt_flag == 'apt':
                    out = self.feat(x, prompt=self.prompt, train=train)
                    out = out[:, 0, :]
                else:
                    raise ValueError("prompt flag not supported")
            else:
                out = self.feat(x, train=train)
                out = out[:, 0, :]

            out = self.clf_norm(out)

            wt_norm = F.normalize(self.last.weight, p=2, dim=1)
            out = torch.matmul(out, wt_norm.t())

            return out

    return MOMENT_APT_Classifier(feat_model, out_dim)