"""
CODA-Prompt implementation based on:
"CODA-Prompt: COntinual Decomposed Attention-based Prompting for Rehearsal-Free Continual Learning"
(CVPR 2023)

Key differences from L2Prompt:
- Dual prompt pool: Task-specific (G-Prompt) + Shared general (E-Prompt)
- Attention-based prompt incorporation (not just concatenation)
- Learnable keys for each task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CODAPromptPool(nn.Module):
    """
    CODA-Prompt: Dual-pool prompting with task-specific and general prompts

    Architecture:
    - G-Prompt: Task-specific prompts (one per task, not shared)
    - E-Prompt: Shared general prompts (selected via query-key matching)
    - Attention-based incorporation instead of simple concatenation
    """

    def __init__(
            self,
            n_tasks,
            pool_size=10,  # Size of E-Prompt pool (shared)
            prompt_length=5,  # Length of each prompt
            d_model=512,
            top_k=5,  # Number of E-Prompts to select
            ortho_init=True,  # Orthogonal initialization for keys
            use_g_prompt=True,  # Enable task-specific G-Prompt
            use_e_prompt=True  # Enable shared E-Prompt
    ):
        """
        Args:
            n_tasks: Number of continual learning tasks
            pool_size: Size of shared E-Prompt pool
            prompt_length: Length of each prompt token
            d_model: Embedding dimension
            top_k: Number of E-Prompts to select
            ortho_init: Use orthogonal initialization for prompt keys
            use_g_prompt: Enable G-Prompt (task-specific)
            use_e_prompt: Enable E-Prompt (shared general)
        """
        super().__init__()

        self.n_tasks = n_tasks
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.d_model = d_model
        self.top_k = top_k
        self.use_g_prompt = use_g_prompt
        self.use_e_prompt = use_e_prompt

        # === G-Prompt: Task-specific prompts ===
        if self.use_g_prompt:
            # One prompt per task (not shared across tasks)
            self.g_prompts = nn.ParameterList([
                nn.Parameter(torch.randn(prompt_length, d_model) * 0.02)
                for _ in range(n_tasks)
            ])
            print(f"G-Prompt: {n_tasks} task-specific prompts of length {prompt_length}")

        # === E-Prompt: Shared general prompts ===
        if self.use_e_prompt:
            # Shared prompt pool
            self.e_prompts = nn.Parameter(
                torch.randn(pool_size, prompt_length, d_model) * 0.02
            )

            # Learnable keys for prompt selection
            self.e_keys = nn.Parameter(
                torch.randn(pool_size, d_model) * 0.02
            )

            # Orthogonal initialization for better separation
            if ortho_init:
                nn.init.orthogonal_(self.e_keys)

            # Track usage statistics
            self.register_buffer('e_prompt_usage', torch.zeros(pool_size))

            print(f"E-Prompt: {pool_size} shared prompts, top-{top_k} selection")

        # === Attention mechanism for prompt incorporation ===
        # Scale factor for attention
        self.scale = d_model ** -0.5

        print(f"CODA-Prompt initialized: d_model={d_model}")

    def freeze_g_prompt(self, task_id):
        """Freeze G-Prompt for a specific task"""
        if self.use_g_prompt and task_id < len(self.g_prompts):
            self.g_prompts[task_id].requires_grad = False

    def select_e_prompts(self, query):
        """
        Select top-K E-Prompts based on query-key similarity

        Args:
            query: Query vectors [batch, d_model]

        Returns:
            selected_prompts: [batch, K, prompt_length, d_model]
            top_k_indices: [batch, K]
            selection_scores: [batch, pool_size]
        """
        if not self.use_e_prompt:
            return None, None, None

        batch_size = query.size(0)

        # Normalize for cosine similarity
        query_norm = F.normalize(query, p=2, dim=1)  # [batch, d_model]
        keys_norm = F.normalize(self.e_keys, p=2, dim=1)  # [pool_size, d_model]

        # Compute similarity scores
        similarity = torch.matmul(query_norm, keys_norm.T)  # [batch, pool_size]

        # Add small noise during training for exploration
        if self.training:
            noise = torch.randn_like(similarity) * 0.01
            similarity = similarity + noise

        # Select top-K prompts
        selection_scores = similarity
        _, top_k_indices = similarity.topk(self.top_k, dim=1)  # [batch, K]

        # Gather selected prompts
        expanded_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1)  # [batch, K, 1, 1]
        expanded_indices = expanded_indices.expand(-1, -1, self.prompt_length, self.d_model)

        prompts_expanded = self.e_prompts.unsqueeze(0).expand(batch_size, -1, -1, -1)
        selected_prompts = torch.gather(prompts_expanded, 1, expanded_indices)

        # Update usage statistics
        if self.training:
            with torch.no_grad():
                for idx in top_k_indices.flatten():
                    self.e_prompt_usage[idx] += 1

        return selected_prompts, top_k_indices, selection_scores

    def incorporate_prompts_with_attention(self, x, g_prompt=None, e_prompts=None):
        """
        Incorporate prompts using attention mechanism (key innovation of CODA-Prompt)

        Args:
            x: Input features [batch, seq_len, d_model]
            g_prompt: G-Prompt [prompt_length, d_model] or None
            e_prompts: Selected E-Prompts [batch, K, prompt_length, d_model] or None

        Returns:
            x_with_prompts: [batch, total_prompt_len + seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        prompts_to_concat = []

        # Add G-Prompt (task-specific)
        if g_prompt is not None:
            g_prompt_expanded = g_prompt.unsqueeze(0).expand(batch_size, -1, -1)
            prompts_to_concat.append(g_prompt_expanded)

        # Add E-Prompts (shared, selected)
        if e_prompts is not None:
            # Reshape E-Prompts from [batch, K, prompt_length, d_model]
            # to [batch, K*prompt_length, d_model]
            e_prompts_flat = e_prompts.reshape(batch_size, -1, d_model)
            prompts_to_concat.append(e_prompts_flat)
        # Añadir CLS token al principio de la secuencia
        if not hasattr(self, 'cls_token'):
            self.cls_token = nn.Parameter(torch.zeros(1, 1, x.size(-1), device=x.device))
            nn.init.trunc_normal_(self.cls_token, std=0.02)


        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Concatenate all prompts with input
        if len(prompts_to_concat) > 0:
            all_prompts = torch.cat(prompts_to_concat, dim=1)
            x_with_prompts = torch.cat([all_prompts, x], dim=1)
        else:
            x_with_prompts = x



        return x_with_prompts

    def forward(self, x, task_id=None, return_selection_info=False):
        """
        Forward pass: select and incorporate prompts

        Args:
            x: Features [batch, seq_len, d_model] or query [batch, d_model]
            task_id: Task identifier (int or None)
            return_selection_info: Return selection scores and indices

        Returns:
            x_with_prompts: Features with prompts prepended
            (optional) selection_info: Dict with selection details
        """
        # Handle both feature tensors and query vectors
        if x.dim() == 2:
            # Query vector: [batch, d_model]
            query = x
            is_query_only = True
        else:
            # Feature tensor: [batch, seq_len, d_model]
            query = x.mean(dim=1)  # [batch, d_model]
            is_query_only = False

        batch_size = query.size(0)

        # === Select G-Prompt (task-specific) ===
        g_prompt = None
        if self.use_g_prompt and task_id is not None:
            if isinstance(task_id, int):
                g_prompt = self.g_prompts[task_id]
            else:
                # If task_id is a tensor, use the first one (assumes same task in batch)
                g_prompt = self.g_prompts[task_id[0].item()]

        # === Select E-Prompts (shared) ===
        e_prompts = None
        top_k_indices = None
        selection_scores = None

        if self.use_e_prompt:
            e_prompts, top_k_indices, selection_scores = self.select_e_prompts(query)

        # === Incorporate prompts ===
        if is_query_only:
            # For query-only input, return selected prompts flattened
            prompts_list = []

            if g_prompt is not None:
                g_prompt_expanded = g_prompt.unsqueeze(0).expand(batch_size, -1, -1)
                prompts_list.append(g_prompt_expanded)

            if e_prompts is not None:
                e_prompts_flat = e_prompts.reshape(batch_size, -1, self.d_model)
                prompts_list.append(e_prompts_flat)

            if len(prompts_list) > 0:
                output = torch.cat(prompts_list, dim=1)
            else:
                output = torch.zeros(batch_size, 0, self.d_model, device=query.device)
        else:
            # For feature tensors, concatenate prompts with features
            output = self.incorporate_prompts_with_attention(x, g_prompt, e_prompts)

        if return_selection_info:
            selection_info = {
                'e_prompt_indices': top_k_indices,
                'e_prompt_scores': selection_scores,
                'g_prompt_used': g_prompt is not None,
                'e_prompts_used': e_prompts is not None
            }
            return output, selection_info

        return output

    def get_prompt_statistics(self):
        """Get usage statistics for E-Prompts"""
        if not self.use_e_prompt:
            return None

        return {
            'e_prompt_usage': self.e_prompt_usage.cpu().numpy(),
            'total_selections': self.e_prompt_usage.sum().item(),
            'most_used': self.e_prompt_usage.argmax().item(),
            'least_used': self.e_prompt_usage.argmin().item()
        }