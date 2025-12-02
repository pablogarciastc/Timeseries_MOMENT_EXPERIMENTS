"""
L2Prompt implementation faithful to the original paper:
"Learning to Prompt for Continual Learning" (CVPR 2022)

Fully integrated with MOMENT backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2PromptPool(nn.Module):
    """
    L2Prompt: Learnable Prompt Pool for Continual Learning

    Faithful to the original paper:
    - Shared prompt pool across all tasks
    - Top-K selection by query-key matching
    - Gradient isolation for unselected prompts
    """

    def __init__(self, pool_size=20, prompt_length=5, d_model=512, top_k=5):
        """
        Args:
            pool_size: Total number of prompts in shared pool (default: 20)
            prompt_length: Length of each prompt (default: 5)
            d_model: Embedding dimension from MOMENT
            top_k: Number of prompts to select (default: 5)
        """
        super().__init__()

        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.d_model = d_model
        self.top_k = top_k

        # Shared prompt pool (main learnable component)
        self.prompts = nn.Parameter(
            torch.randn(pool_size, prompt_length, d_model) * 0.1
        )

        # Keys for prompt selection
        self.keys = nn.Parameter(
            torch.randn(pool_size, d_model) * 0.1
        )

        # Statistics tracking
        self.register_buffer('prompt_usage', torch.zeros(pool_size))

    def select_prompts(self, x):
        """
        Select top-K prompts based on input similarity

        Args:
            x: Input embeddings [batch, seq_len, d_model]

        Returns:
            selected_prompts: [batch, K, prompt_length, d_model]
            selected_indices: [batch, K]
        """
        batch_size = x.size(0)

        # Generate query from input (mean over sequence)
        query = x.mean(dim=1)  # [batch, d_model]

        # Normalize for cosine similarity
        query_norm = F.normalize(query, p=2, dim=1)
        keys_norm = F.normalize(self.keys, p=2, dim=1)

        # Compute similarity: [batch, pool_size]
        similarity = torch.matmul(query_norm, keys_norm.T)
        similarity += (0.02* torch.randn_like(similarity))


        # Select top-K prompts
        _, top_k_indices = similarity.topk(self.top_k, dim=1)  # [batch, K]

        # Gather selected prompts
        # Expand indices to match prompt dimensions
        expanded_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1)  # [batch, K, 1, 1]
        expanded_indices = expanded_indices.expand(-1, -1, self.prompt_length, self.d_model)

        # Select prompts: [batch, K, prompt_length, d_model]
        prompts_expanded = self.prompts.unsqueeze(0).expand(batch_size, -1, -1, -1)
        selected_prompts = torch.gather(prompts_expanded, 1, expanded_indices)

        # Update usage statistics
        if self.training:
            for idx in top_k_indices.flatten():
                self.prompt_usage[idx] += 1
        if self.training:
            print(
                f"Task ?: {torch.bincount(top_k_indices.flatten(), minlength=self.pool_size).cpu().numpy()}")

        return selected_prompts, top_k_indices

    def forward(self, x):
        """
        Forward pass: select and prepend prompts

        Args:
            x: MOMENT embeddings [batch, seq_len, d_model]

        Returns:
            x_with_prompts: [batch, K*prompt_length + seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape

        # Select top-K prompts
        selected_prompts, _ = self.select_prompts(x)  # [batch, K, prompt_length, d_model]

        # Reshape to [batch, K*prompt_length, d_model]
        selected_prompts_flat = selected_prompts.reshape(batch_size, -1, d_model)

        # Concatenate prompts with input
        x_with_prompts = torch.cat([selected_prompts_flat, x], dim=1)


        return x_with_prompts

    def select_prompts_from_query(self, query):
        """
        Select prompts using pre-computed query from transformer output

        Args:
            query: Query vectors from transformer output [batch, d_model]

        Returns:
            selected_prompts: [batch, K*prompt_length, d_model]
            top_k_indices: [batch, K]
        """
        batch_size = query.size(0)

        # Normalize for cosine similarity
        query_norm = F.normalize(query, p=2, dim=1)
        keys_norm = F.normalize(self.keys, p=2, dim=1)

        # Compute similarity
        similarity = torch.matmul(query_norm, keys_norm.T)

        # Optional: Add noise during training for exploration
        if self.training:
            similarity += (0.02 * torch.randn_like(similarity))

        # Select top-K prompts
        _, top_k_indices = similarity.topk(self.top_k, dim=1)  # [batch, K]

        # Gather selected prompts (YOUR CODE)
        # Expand indices to match prompt dimensions
        expanded_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1)  # [batch, K, 1, 1]
        expanded_indices = expanded_indices.expand(-1, -1, self.prompt_length, self.d_model)

        # Select prompts: [batch, K, prompt_length, d_model]
        prompts_expanded = self.prompts.unsqueeze(0).expand(batch_size, -1, -1, -1)
        selected_prompts = torch.gather(prompts_expanded, 1, expanded_indices)

        # Reshape to [batch, K*prompt_length, d_model] for concatenation
        selected_prompts_flat = selected_prompts.reshape(batch_size, -1, self.d_model)

        # Update usage statistics if training
        if self.training:
            for idx in top_k_indices.flatten():
                self.prompt_usage[idx] += 1
            print(f"Task ?: {torch.bincount(top_k_indices.flatten(), minlength=self.pool_size).cpu().numpy()}")

        return selected_prompts_flat, top_k_indices