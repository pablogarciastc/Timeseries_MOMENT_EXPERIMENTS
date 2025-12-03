import math
import weakref
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
import copy
import random
import numpy as np
import logging


import math
import weakref
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
import copy
import random
import numpy as np
import logging

class Adapter_lora(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in",
                 input_dim=None,
                 output_dim=None):
        super().__init__()
        self.random_orth = True
        default_dim = config.d_model if d_model is None else d_model
        self.input_dim = default_dim if input_dim is None else input_dim
        self.output_dim = default_dim if output_dim is None else output_dim
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        self.lora_A = nn.Linear(self.down_size, self.output_dim, bias=False)
        self.lora_B = nn.Linear(self.input_dim, self.down_size, bias=False)
        if self.random_orth:
            random_matrix = torch.rand(self.input_dim, self.down_size)
            q, _ = torch.linalg.qr(random_matrix)
            q = q[:, :self.down_size]
            with torch.no_grad():
                self.lora_B.weight.copy_(q.T)
            self.lora_B.weight.data *= 0.1
        else:
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))
        if init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                self.lora_A.weight.data *= 0.1
            print(f"[ADAPTER INIT] lora_A norm: {self.lora_A.weight.norm().item():.6f}, lora_B norm: {self.lora_B.weight.norm().item():.6f}")

    def forward(self, x):
        inter_x = self.lora_B(x)
        return self.lora_A(inter_x)


def moment_1_small_cllora(tuning_config=None, **kwargs):
    from momentfm import MOMENTPipeline
    print("Loading MOMENT-1-small with CL-LoRA adapters and CLS token...")

    moment_model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small",
        model_kwargs={
            'task_name': 'reconstruction',
            "n_channels": 45,
            "num_class": 18,
            'enable_gradient_checkpointing': False,
        }
    )

    class MOMENTWithCLLoRA_CLS(nn.Module):
        def __init__(self, moment_base, config):
            super().__init__()
            self.moment = moment_base
            self.config = config
            self.d_model = moment_base.config.d_model
            self._device = config._device

            self.msa_adapt = getattr(config, 'msa_adapt', False)
            self.use_block_weight = getattr(config, 'use_block_weight', False)
            self.use_distillation = getattr(config, 'use_distillation', False)
            self.msa = getattr(config, 'msa', [1, 1, 1])

            self.general_pos = getattr(config, 'general_pos', [])
            self.specfic_pos = getattr(config, 'specfic_pos', [])
            self.adapt_pos = sorted(self.general_pos + self.specfic_pos)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
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
            elif hasattr(encoder, 'model') and hasattr(encoder.model, 'block'):
                encoder_blocks = list(encoder.model.block)
            else:
                encoder_blocks = []

            self.num_blocks = len(encoder_blocks) if len(encoder_blocks) > 0 else 8

            self._attention_blocks = []
            self._attention_qkv = []
            self._qkv_dim_lookup = []

            for block in encoder_blocks:
                attn_module = None
                if hasattr(block, 'layer') and len(block.layer) > 0:
                    layer0 = block.layer[0]
                    if hasattr(layer0, 'SelfAttention'):
                        attn_module = layer0.SelfAttention
                elif hasattr(block, 'SelfAttention'):
                    attn_module = block.SelfAttention

                self._attention_blocks.append(attn_module)
                if attn_module is None:
                    self._attention_qkv.append({})
                    self._qkv_dim_lookup.append({})
                    continue

                q_linear = getattr(attn_module, 'q', None)
                k_linear = getattr(attn_module, 'k', None)
                v_linear = getattr(attn_module, 'v', None)

                qkv_dict = {'q': q_linear, 'k': k_linear, 'v': v_linear}
                dim_lookup = {}
                for key, linear_module in qkv_dict.items():
                    if linear_module is None:
                        continue
                    weight = getattr(linear_module, 'weight', None)
                    if weight is not None:
                        dim_lookup[key] = (weight.shape[1], weight.shape[0])
                    else:
                        in_features = getattr(linear_module, 'in_features', self.d_model)
                        out_features = getattr(linear_module, 'out_features', self.d_model)
                        dim_lookup[key] = (in_features, out_features)
                self._attention_qkv.append(qkv_dict)
                self._qkv_dim_lookup.append(dim_lookup)

            self._active_adapter_mapping = {}
            self._active_block_weight_mapping = {}
            self.adapter_list = []
            self.adapter_pos_list = []
            self.cur_adapter = nn.ModuleList()

            if self.use_distillation:
                self.old_adapter_list = nn.ModuleList()
            if self.use_block_weight:
                self.block_weight_list = []
                self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
                nn.init.uniform_(self.block_weight, .5, 1.5)

            if self.msa_adapt:
                self.get_new_adapter_initial_msa()

        def _prepare_inputs(self, x, allow_new_proj=False):
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            x = x.to(device=self._device, dtype=torch.float32)
            x = self.proj(x)
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
            return x

        def _encode_with_adapters(self, x, adapter_mapping=None, block_weight_mapping=None, grad_enabled=True):
            prev_mapping = self._active_adapter_mapping
            prev_weights = self._active_block_weight_mapping
            self._active_adapter_mapping = adapter_mapping or {}
            self._active_block_weight_mapping = block_weight_mapping or {}

            try:
                with torch.set_grad_enabled(grad_enabled):
                    if hasattr(self.moment, 'model') and hasattr(self.moment.model, 'encoder'):
                        output = self.moment.model.encoder(inputs_embeds=x)
                    elif hasattr(self.moment, 'encoder'):
                        output = self.moment.encoder(inputs_embeds=x)
                    else:
                        output = self.moment(x)
            finally:
                self._active_adapter_mapping = prev_mapping
                self._active_block_weight_mapping = prev_weights

            hidden = getattr(output, 'last_hidden_state', output)
            return hidden

        def get_new_adapter_initial_msa(self):
            config = self.config
            for i in range(len(self.adapt_pos)):
                temp_adapter = nn.ModuleList()
                for j in self.msa:
                    if j == 1:
                        adapter = Adapter_lora(
                            self.config,
                            dropout=0.0,
                            bottleneck=config.ffn_num,
                            init_option=config.ffn_adapter_init_option,
                            adapter_scalar=config.ffn_adapter_scalar,
                            adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                            input_dim=self.d_model,
                            output_dim=self.d_model,
                        ).to(self._device)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)
                self.cur_adapter.append(temp_adapter)

        def add_adapter_to_list(self):
            temp_adapter = []
            for i in range(len(self.specfic_pos)):
                temp_pos = self.adapt_pos.index(self.specfic_pos[i])
                temp_adapter.append(copy.deepcopy(self.cur_adapter[temp_pos].requires_grad_(False)))
            self.adapter_list.append(temp_adapter)

            if self.use_block_weight:
                self.block_weight_old = copy.deepcopy(self.block_weight)
                self.block_weight_list.append(self.block_weight_old.requires_grad_(False))
                self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))

            self.adapter_pos_list.append(self.adapt_pos)

            if self.use_distillation:
                self.old_adapter_list.append(copy.deepcopy(self.cur_adapter).requires_grad_(False))

            if self.msa_adapt:
                self.get_new_adapter_initial_msa()

        def _build_adapter_mapping(self, saved_index=None):
            if not getattr(self, 'msa_adapt', False):
                return {}, {}

            mapping = {}
            block_weights = {}

            for pos in self.general_pos:
                if pos >= len(self._attention_qkv):
                    continue
                pos_idx = self.adapt_pos.index(pos)
                mapping[pos] = self.cur_adapter[pos_idx]

            if saved_index is None:
                for pos in self.specfic_pos:
                    if pos >= len(self._attention_qkv):
                        continue
                    pos_idx = self.adapt_pos.index(pos)
                    mapping[pos] = self.cur_adapter[pos_idx]
                    if self.use_block_weight and len(self.specfic_pos) > 0:
                        spec_idx = self.specfic_pos.index(pos)
                        block_weights[pos] = self.block_weight[:, spec_idx].clone()
            elif 0 <= saved_index < len(self.adapter_list):
                saved_adapters = self.adapter_list[saved_index]
                for spec_idx, pos in enumerate(self.specfic_pos):
                    if pos >= len(self._attention_qkv):
                        continue
                    if spec_idx < len(saved_adapters):
                        mapping[pos] = saved_adapters[spec_idx]
                        if self.use_block_weight and saved_index < len(self.block_weight_list):
                            block_weights[pos] = self.block_weight_list[saved_index][:, spec_idx].clone()
            return mapping, block_weights

        def forward_train(self, x):
            x = self._prepare_inputs(x)
            mapping, weights = self._build_adapter_mapping(saved_index=None)
            hidden = self._encode_with_adapters(x, mapping, weights, grad_enabled=True)
            return hidden[:, 0, :]  # CLS token

        def forward_test(self, x):
            x = self._prepare_inputs(x)
            features = []

            # Base CLS
            hidden_base = self._encode_with_adapters(x, {}, {}, grad_enabled=False)
            features.append(hidden_base[:, 0, :])

            # Adapters
            for idx in range(len(self.adapter_list)):
                mapping, weights = self._build_adapter_mapping(saved_index=idx)
                hidden = self._encode_with_adapters(x, mapping, weights, grad_enabled=False)
                features.append(hidden[:, 0, :])

            return torch.cat(features, dim=1)

        def forward_proto(self, x, adapt_index):
            x = self._prepare_inputs(x, allow_new_proj=True)
            if adapt_index == -1:
                hidden = self._encode_with_adapters(x, {}, {}, grad_enabled=False)
            elif 0 <= adapt_index < len(self.adapter_list):
                mapping, weights = self._build_adapter_mapping(saved_index=adapt_index)
                hidden = self._encode_with_adapters(x, mapping, weights, grad_enabled=False)
            else:
                mapping, weights = self._build_adapter_mapping(saved_index=None)
                hidden = self._encode_with_adapters(x, mapping, weights, grad_enabled=False)
            return hidden[:, 0, :]

        def _encode(self, x):
            """
            Runs the MOMENT encoder and returns the hidden states.
            Compatible with both moment.model.encoder and moment.encoder.
            """
            with torch.set_grad_enabled(True):
                if hasattr(self.moment, 'model') and hasattr(self.moment.model, 'encoder'):
                    output = self.moment.model.encoder(inputs_embeds=x)
                elif hasattr(self.moment, 'encoder'):
                    output = self.moment.encoder(inputs_embeds=x)
                else:
                    output = self.moment(x)
            hidden_states = getattr(output, 'last_hidden_state', output)
            return hidden_states

        def forward_general_cls(self, x, t_idx):
            x = self._prepare_inputs(x)

            # Student (current adapters)
            hidden_student = self._encode(x)
            student_cls = hidden_student[:, 0, :]  # CLS token output

            # Teacher (previous adapters)
            if t_idx > 0 and hasattr(self, 'old_adapter_list') and len(self.old_adapter_list) >= t_idx:
                teacher_adapter = self.old_adapter_list[t_idx - 1]
                adapted_teacher = student_cls.clone()
                for sublist in teacher_adapter:
                    for sub in sublist:
                        if not isinstance(sub, nn.Identity):
                            adapted_teacher = adapted_teacher + sub(adapted_teacher)
                teacher_cls = adapted_teacher
            else:
                teacher_cls = student_cls.detach()

            return student_cls, teacher_cls

        def forward(self, x, test=False, use_init_ptm=None):
            if use_init_ptm is None:
                use_init_ptm = getattr(self.config, "use_init_ptm", False)
            if not test:
                return self.forward_train(x)
            else:
                return self.forward_test(x)

    model = MOMENTWithCLLoRA_CLS(moment_model, tuning_config)
    device = tuning_config._device if hasattr(tuning_config, '_device') else torch.device('cuda:0')
    model = model.to(device)
    print(f"MOMENT-1-small + CLS + CL-LoRA loaded successfully!\n  d_model: {model.d_model}")
    return model

def moment_1_large_cllora(tuning_config=None, **kwargs):
    """
    Load MOMENT-1-small model and wrap it with CL-LoRA adapters
    """
    from momentfm import MOMENTPipeline

    print("Loading MOMENT-1-small with CL-LoRA adapters...")

    moment_model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            'task_name': 'classification',
            "n_channels": 45,
            "num_class": 18,
            'enable_gradient_checkpointing': False,
        }
    )

    class MOMENTWithCLLoRA(nn.Module):
        class _AdapterAwareLinear(nn.Module):
            def __init__(self, base_linear, parent, block_idx, msa_idx):
                super().__init__()
                self.base_linear = base_linear
                object.__setattr__(self, "_parent_ref", weakref.ref(parent))
                self.block_idx = block_idx
                self.msa_idx = msa_idx

            def forward(self, x):
                out = self.base_linear(x)
                parent = self._parent_ref()
                if parent is None:
                    return out

                mapping = parent._active_adapter_mapping

                if not hasattr(self, '_debug_printed'):
                    if mapping:
                        has_adapter = self.block_idx in mapping
                        print(f"[DEBUG] Block {self.block_idx}, MSA {self.msa_idx}: "
                              f"mapping={'Yes' if has_adapter else 'No'}")
                    self._debug_printed = True

                if mapping:
                    adapters = mapping.get(self.block_idx)
                    if adapters is not None and self.msa_idx < len(adapters):
                        adapter_module = adapters[self.msa_idx]
                        if adapter_module is not None and not isinstance(adapter_module, nn.Identity):
                            delta = adapter_module(x)

                            if not hasattr(adapter_module, '_debug_used'):
                                print(f"[DEBUG] ✅ Adapter used: Block {self.block_idx}, MSA {self.msa_idx}, "
                                      f"delta norm={delta.norm().item():.4f}")
                                adapter_module._debug_used = True

                            if isinstance(delta, tuple):
                                delta = delta[0]
                            weight = parent._resolve_block_weight(self.block_idx, self.msa_idx, delta)
                            if weight is not None:
                                delta = delta * weight
                            out = out + delta.to(dtype=out.dtype, device=out.device)
                return out

        def __init__(self, moment_base, config):
            super().__init__()
            self.moment = moment_base
            self.config = config
            self._attention_blocks = []
            self._attention_qkv = []
            self._qkv_dim_lookup = []

            self._active_adapter_mapping = {}
            self._active_block_weight_mapping = {}

            self.adapter_list = []
            self.adapter_pos_list = []
            self.cur_adapter = nn.ModuleList()

            if self.use_distillation:
                self.old_adapter_list = nn.ModuleList()

            if self.use_block_weight:
                self.block_weight_list = []
                self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
                nn.init.uniform_(self.block_weight, .5, 1.5)


            self.out_dim = moment_base.config.d_model
            self.d_model = moment_base.config.d_model

            self.msa_adapt = config.msa_adapt
            self.use_distillation = config.use_distillation
            self.use_block_weight = config.use_block_weight
            self.use_init_ptm = getattr(config, "use_init_ptm", False)

            if self.msa_adapt:
                self.msa = config.msa

            self.general_pos = config.general_pos
            self.specfic_pos = config.specfic_pos
            self.adapt_pos = sorted(self.general_pos + self.specfic_pos)

            self._device = config._device
            self.proj = nn.Linear(45, self.d_model).to(self._device)

            if self.use_distillation:
                self.old_adapter_list = nn.ModuleList()

            if self.use_block_weight:
                self.block_weight_list = []
                self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
                nn.init.uniform_(self.block_weight, .5, 1.5)
            self.adapter_list = []
            self.adapter_pos_list = []
            self.cur_adapter = nn.ModuleList()

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
            elif hasattr(encoder, 'model') and hasattr(encoder.model, 'block'):
                encoder_blocks = list(encoder.model.block)
            else:
                encoder_blocks = []

            self.num_blocks = len(encoder_blocks) if len(encoder_blocks) > 0 else 8

            self._attention_blocks = []
            self._attention_qkv = []
            self._qkv_dim_lookup = []
            for block in encoder_blocks:
                attn_module = None
                if hasattr(block, 'layer') and len(block.layer) > 0:
                    layer0 = block.layer[0]
                    if hasattr(layer0, 'SelfAttention'):
                        attn_module = layer0.SelfAttention
                elif hasattr(block, 'SelfAttention'):
                    attn_module = block.SelfAttention

                self._attention_blocks.append(attn_module)
                if attn_module is None:
                    self._attention_qkv.append({})
                    self._qkv_dim_lookup.append({})
                    continue

                q_linear = getattr(attn_module, 'q', None)
                k_linear = getattr(attn_module, 'k', None)
                v_linear = getattr(attn_module, 'v', None)

                qkv_dict = {'q': q_linear, 'k': k_linear, 'v': v_linear}
                dim_lookup = {}
                for key, linear_module in qkv_dict.items():
                    if linear_module is None:
                        continue
                    weight = getattr(linear_module, 'weight', None)
                    if weight is not None:
                        dim_lookup[key] = (weight.shape[1], weight.shape[0])
                    else:
                        in_features = getattr(linear_module, 'in_features', self.d_model)
                        out_features = getattr(linear_module, 'out_features', self.d_model)
                        dim_lookup[key] = (in_features, out_features)
                self._attention_qkv.append(qkv_dict)
                self._qkv_dim_lookup.append(dim_lookup)

            self._active_adapter_mapping = {}
            self._active_block_weight_mapping = {}
            self._wrap_attention_linears()

            print("\n[DEBUG] Checking Q/K/V wrapping:")
            for i in range(min(3, len(self._attention_blocks))):
                attn = self._attention_blocks[i]
                if attn:
                    q_wrapped = isinstance(getattr(attn, 'q', None), self._AdapterAwareLinear)
                    k_wrapped = isinstance(getattr(attn, 'k', None), self._AdapterAwareLinear)
                    v_wrapped = isinstance(getattr(attn, 'v', None), self._AdapterAwareLinear)
                    print(f"  Block {i}: Q={q_wrapped}, K={k_wrapped}, V={v_wrapped}")

            if self.msa_adapt:
                self.get_new_adapter_initial_msa()

        def _wrap_attention_linears(self):
            for block_idx, attn_module in enumerate(self._attention_blocks):
                if attn_module is None:
                    continue
                original_qkv = self._attention_qkv[block_idx]
                wrapped = {}
                for msa_idx, key in enumerate(['q', 'k', 'v']):
                    base_linear = original_qkv.get(key)
                    if base_linear is None:
                        continue
                    wrapper = self._AdapterAwareLinear(base_linear, self, block_idx, msa_idx)
                    setattr(attn_module, key, wrapper)
                    wrapped[key] = wrapper
                self._attention_qkv[block_idx] = wrapped

        def _resolve_block_weight(self, block_idx, msa_idx, reference_tensor):
            if not self.use_block_weight:
                return None
            weights = self._active_block_weight_mapping.get(block_idx)
            if weights is None:
                return None
            if isinstance(weights, torch.Tensor):
                weight_tensor = weights
            else:
                weight_tensor = torch.as_tensor(weights, device=reference_tensor.device, dtype=reference_tensor.dtype)
            if weight_tensor.dim() == 0:
                selected = weight_tensor
            else:
                if msa_idx >= weight_tensor.shape[0]:
                    return None
                selected = weight_tensor[msa_idx]
            if selected.device != reference_tensor.device or selected.dtype != reference_tensor.dtype:
                selected = selected.to(reference_tensor.device, reference_tensor.dtype)

            selected = selected.clone()
            if selected.dim() == 0:
                selected = selected.reshape(1, 1, 1)
            return selected

        def _get_adapter_dims(self, block_idx, msa_idx):
            default_in = self.d_model
            default_out = self.d_model
            if block_idx >= len(self._qkv_dim_lookup):
                return default_in, default_out
            dims = self._qkv_dim_lookup[block_idx]
            key = ['q', 'k', 'v'][msa_idx]
            if key not in dims:
                return default_in, default_out
            return dims[key]

        def get_new_adapter_initial_msa(self):
            config = self.config

            import traceback
            print(f"[DEBUG] get_new_adapter_initial_msa CALLED (len before={len(self.cur_adapter)})")
            traceback.print_stack(limit=4)

            if config.ffn_adapt:
                for i in range(len(self.adapt_pos)):
                    temp_adapter = nn.ModuleList()
                    for j in self.msa:
                        if j == 1:
                            in_dim, out_dim = self._get_adapter_dims(self.adapt_pos[i], len(temp_adapter))
                            adapter = Adapter_lora(
                                self.config,
                                dropout=0.0,
                                bottleneck=config.ffn_num,
                                init_option=config.ffn_adapter_init_option,
                                adapter_scalar=config.ffn_adapter_scalar,
                                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                input_dim=in_dim,
                                output_dim=out_dim,
                            ).to(self._device)
                        else:
                            adapter = nn.Identity()
                        temp_adapter.append(adapter)
                    self.cur_adapter.append(temp_adapter)
                self.cur_adapter = self.cur_adapter.to(self._device)
                self.cur_adapter.requires_grad_(True)

        def get_new_adapter_msa(self):
            config = self.config

            if config.ffn_adapt:
                for i in range(len(self.specfic_pos)):
                    pos = self.adapt_pos.index(self.specfic_pos[i])
                    temp_adapter = nn.ModuleList()
                    for j in self.msa:
                        if j == 1:
                            in_dim, out_dim = self._get_adapter_dims(self.specfic_pos[i], len(temp_adapter))
                            adapter = Adapter_lora(
                                self.config,
                                dropout=0.0,
                                bottleneck=config.ffn_num,
                                init_option=config.ffn_adapter_init_option,
                                adapter_scalar=config.ffn_adapter_scalar,
                                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                input_dim=in_dim,
                                output_dim=out_dim,
                            ).to(self._device)
                            adapter.requires_grad_(True)
                        else:
                            adapter = nn.Identity()
                        temp_adapter.append(adapter)
                    self.cur_adapter[pos] = temp_adapter

                if len(self.specfic_pos) < len(self.adapt_pos):
                    self.cur_adapter = self.cur_adapter.to(self._device)
                    self.cur_adapter.requires_grad_(True)
                    for i in self.adapt_pos:
                        if i in self.general_pos:
                            pos = self.adapt_pos.index(i)
                            for j in range(len(self.msa)):
                                if self.msa[j] == 1:
                                    self.cur_adapter[pos][j] = self.cur_adapter[pos][j].to(self._device)
                                    self.cur_adapter[pos][j].lora_B.requires_grad_(False)

        def add_adapter_to_list(self):
            temp_adapter = []
            for i in range(len(self.specfic_pos)):
                temp_pos = self.adapt_pos.index(self.specfic_pos[i])
                temp_adapter.append(copy.deepcopy(self.cur_adapter[temp_pos].requires_grad_(False)))
            self.adapter_list.append(temp_adapter)

            if self.use_block_weight:
                self.block_weight_old = copy.deepcopy(self.block_weight)
                self.block_weight_list.append(self.block_weight_old.requires_grad_(False))
                self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
                nn.init.uniform_(self.block_weight, .5, 1.5)

            self.adapter_pos_list.append(self.adapt_pos)

            if self.use_distillation:
                self.old_adapter_list.append(copy.deepcopy(self.cur_adapter).requires_grad_(False))

            if self.msa_adapt:
                self.get_new_adapter_msa()

        def freeze(self):
            for param in self.parameters():
                param.requires_grad = False

            for i in range(len(self.cur_adapter)):
                for adapter in self.cur_adapter[i]:
                    if hasattr(adapter, 'parameters'):
                        for param in adapter.parameters():
                            param.requires_grad = True

        def _prepare_inputs(self, x, allow_new_proj=False):
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            param = next(self.moment.parameters())
            target_dtype = param.dtype
            target_device = param.device
            x = x.to(device=target_device, dtype=target_dtype)

            if x.shape[-1] != self.d_model:
                if allow_new_proj and x.shape[-1] != self.proj.in_features:
                    proj = nn.Linear(x.shape[-1], self.d_model).to(target_device, dtype=target_dtype)
                    x = proj(x)
                else:
                    self.proj = self.proj.to(target_device, dtype=target_dtype)
                    x = self.proj(x)

            return x

        def _set_active_mappings(self, adapter_mapping=None, block_weight_mapping=None):
            self._active_adapter_mapping = adapter_mapping or {}
            self._active_block_weight_mapping = block_weight_mapping or {}

        def _encode_with_adapters(self, x, adapter_mapping=None, block_weight_mapping=None, grad_enabled=True):
            prev_mapping = self._active_adapter_mapping
            prev_weights = self._active_block_weight_mapping
            self._set_active_mappings(adapter_mapping, block_weight_mapping)

            try:
                with torch.set_grad_enabled(grad_enabled):
                    if hasattr(self.moment, 'model') and hasattr(self.moment.model, 'encoder'):
                        moment_output = self.moment.model.encoder(inputs_embeds=x)
                    elif hasattr(self.moment, 'encoder'):
                        moment_output = self.moment.encoder(inputs_embeds=x)
                    else:
                        moment_output = self.moment(x)
            finally:
                self._active_adapter_mapping = prev_mapping
                self._active_block_weight_mapping = prev_weights

            hidden_states = getattr(moment_output, 'last_hidden_state', moment_output)
            return hidden_states

        def _build_adapter_mapping(self, saved_index=None):
            if not getattr(self, 'msa_adapt', False):
                return {}, {}

            mapping = {}
            block_weights = {}

            for pos in self.general_pos:
                if pos >= len(self._attention_qkv):
                    continue
                pos_idx = self.adapt_pos.index(pos)
                mapping[pos] = self.cur_adapter[pos_idx]

            if saved_index is None:
                for pos in self.specfic_pos:
                    if pos >= len(self._attention_qkv):
                        continue
                    pos_idx = self.adapt_pos.index(pos)
                    mapping[pos] = self.cur_adapter[pos_idx]
                    if self.use_block_weight and pos in self.specfic_pos and len(self.specfic_pos) > 0:
                        spec_idx = self.specfic_pos.index(pos)
                        block_weights[pos] = self.block_weight[:, spec_idx].clone()
            elif 0 <= saved_index < len(self.adapter_list):
                saved_adapters = self.adapter_list[saved_index]
                for spec_idx, pos in enumerate(self.specfic_pos):
                    if pos >= len(self._attention_qkv):
                        continue
                    if spec_idx < len(saved_adapters):
                        mapping[pos] = saved_adapters[spec_idx]
                        if self.use_block_weight and saved_index < len(self.block_weight_list):
                            block_weights[pos] = self.block_weight_list[saved_index][:, spec_idx].clone()
            else:
                for pos in self.specfic_pos:
                    if pos >= len(self._attention_qkv):
                        continue
                    pos_idx = self.adapt_pos.index(pos)
                    mapping[pos] = self.cur_adapter[pos_idx]
                    if self.use_block_weight and len(self.specfic_pos) > 0:
                        spec_idx = self.specfic_pos.index(pos)
                        block_weights[pos] = self.block_weight[:, spec_idx].clone()

            return mapping, block_weights

        def _build_general_mapping_from_source(self, source_adapters):
            if not getattr(self, 'msa_adapt', False) or source_adapters is None:
                return {}, {}

            mapping = {}
            block_weights = {}
            for pos in self.general_pos:
                if pos >= len(self._attention_qkv):
                    continue
                pos_idx = self.adapt_pos.index(pos)
                if pos_idx < len(source_adapters):
                    mapping[pos] = source_adapters[pos_idx]
            return mapping, block_weights

        def forward(self, x, test=False, use_init_ptm=None):
            if use_init_ptm is None:
                use_init_ptm = self.use_init_ptm
            x = self._prepare_inputs(x)

            if not test:
                return self.forward_train(x, use_init_ptm=use_init_ptm)

            return self.forward_test(x, use_init_ptm)

        def forward_train(self, x, use_init_ptm=False):
            """
            Training mode: Similar to original ViT CL-LoRA
            Only uses current task's adapter (cur_adapter)
            """
            x = self._prepare_inputs(x)

            current_mapping, current_weights = self._build_adapter_mapping(saved_index=None)

            if not hasattr(self, '_debug_train_printed'):
                print(f"\n[DEBUG] forward_train:")
                print(f"  Mapping has {len(current_mapping)} blocks")
                print(f"  Mapped blocks: {list(current_mapping.keys())}")
                if current_weights:
                    print(f"  Block weights for: {list(current_weights.keys())}")
                self._debug_train_printed = True

            current_hidden = self._encode_with_adapters(
                x,
                current_mapping,
                current_weights,
                grad_enabled=True
            )

            outcome = current_hidden.mean(dim=1)
            return outcome

        def forward_test(self, x, use_init_ptm=False):
            features = []

            if use_init_ptm:
                hidden_states = self._encode_with_adapters(x, adapter_mapping={}, block_weight_mapping={},
                                                           grad_enabled=False)
                features.append(hidden_states.mean(dim=1))

            if getattr(self, 'msa_adapt', False):
                for adapter_idx in range(len(self.adapter_list)):
                    mapping, block_weights = self._build_adapter_mapping(saved_index=adapter_idx)
                    hidden_states = self._encode_with_adapters(x, mapping, block_weights, grad_enabled=False)
                    features.append(hidden_states.mean(dim=1))

                mapping, block_weights = self._build_adapter_mapping(saved_index=None)
                hidden_states = self._encode_with_adapters(x, mapping, block_weights, grad_enabled=False)
                features.append(hidden_states.mean(dim=1))
            else:
                hidden_states = self._encode_with_adapters(x, adapter_mapping={}, block_weight_mapping={},
                                                           grad_enabled=False)
                features.append(hidden_states.mean(dim=1))

            if not features:
                return torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype)

            return torch.cat(features, dim=1)

        def forward_proto(self, x, adapt_index):
            x = self._prepare_inputs(x, allow_new_proj=True)

            if adapt_index == -1:
                hidden_states = self._encode_with_adapters(x, adapter_mapping={}, block_weight_mapping={},
                                                           grad_enabled=False)
            elif 0 <= adapt_index < len(self.adapter_list):
                mapping, block_weights = self._build_adapter_mapping(saved_index=adapt_index)
                hidden_states = self._encode_with_adapters(x, mapping, block_weights, grad_enabled=False)
            else:
                mapping, block_weights = self._build_adapter_mapping(saved_index=None)
                hidden_states = self._encode_with_adapters(x, mapping, block_weights, grad_enabled=False)

            return hidden_states.mean(dim=1)

        def forward_general_cls(self, x, t_idx):
            x = self._prepare_inputs(x, allow_new_proj=True)

            student_mapping, student_weights = self._build_general_mapping_from_source(self.cur_adapter)
            hidden_student = self._encode_with_adapters(x, student_mapping, student_weights, grad_enabled=True)
            output_new = hidden_student.mean(dim=1)

            if t_idx > 0 and t_idx - 1 < len(self.old_adapter_list):
                teacher_source = self.old_adapter_list[t_idx - 1]
                teacher_mapping, teacher_weights = self._build_general_mapping_from_source(teacher_source)
                hidden_teacher = self._encode_with_adapters(x, teacher_mapping, teacher_weights, grad_enabled=False)
                output_teacher = hidden_teacher.mean(dim=1)
            else:
                output_teacher = output_new.detach()

            return output_new, output_teacher

    model = MOMENTWithCLLoRA(moment_model, tuning_config)

    for param in model.moment.parameters():
        param.requires_grad = False

    device = tuning_config._device if hasattr(tuning_config, '_device') else torch.device('cuda:0')
    model.moment = model.moment.to(device)

    print(f"MOMENT-1-small with CL-LoRA loaded successfully!")
    print(f"  - d_model: {model.d_model}")
    print(f"  - num_blocks: {model.num_blocks}")
    print(f"  - adapt_pos: {model.adapt_pos}")
    print(f"  - MSA config: {model.msa if hasattr(model, 'msa') else 'N/A'}")

    return model


def compute_column_importance(matrix):
    U, S, Vt = torch.linalg.svd(matrix.T, full_matrices=False)
    importance_scores = torch.sum(torch.abs(U * S), dim=1)
    scaled_scores = (importance_scores - torch.min(importance_scores)) / \
                    (torch.max(importance_scores) - torch.min(importance_scores))
    epsilon = 1e-10
    scaled_scores = torch.maximum(scaled_scores, torch.tensor(epsilon))
    return scaled_scores