import torch
import torch.nn as nn
import torch.nn.functional as F


class L2PromptPool(nn.Module):

    def __init__(self, pool_size=20, prompt_length=5, d_model=512, top_k=5):
        super().__init__()

        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.d_model = d_model
        self.top_k = top_k

        self.prompts = nn.Parameter(
            torch.randn(pool_size, prompt_length, d_model) * 0.1
        )

        self.keys = nn.Parameter(
            torch.randn(pool_size, d_model) * 0.1
        )

        self.register_buffer('prompt_usage', torch.zeros(pool_size))

    def select_prompts(self, x):
        batch_size = x.size(0)

        query = x.mean(dim=1)

        query_norm = F.normalize(query, p=2, dim=1)
        keys_norm = F.normalize(self.keys, p=2, dim=1)

        similarity = torch.matmul(query_norm, keys_norm.T)
        similarity += (0.02* torch.randn_like(similarity))

        _, top_k_indices = similarity.topk(self.top_k, dim=1)

        expanded_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1)
        expanded_indices = expanded_indices.expand(-1, -1, self.prompt_length, self.d_model)

        prompts_expanded = self.prompts.unsqueeze(0).expand(batch_size, -1, -1, -1)
        selected_prompts = torch.gather(prompts_expanded, 1, expanded_indices)

        if self.training:
            for idx in top_k_indices.flatten():
                self.prompt_usage[idx] += 1
        if self.training:
            print(
                f"Task ?: {torch.bincount(top_k_indices.flatten(), minlength=self.pool_size).cpu().numpy()}")

        return selected_prompts, top_k_indices

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        selected_prompts, _ = self.select_prompts(x)

        selected_prompts_flat = selected_prompts.reshape(batch_size, -1, d_model)

        x_with_prompts = torch.cat([selected_prompts_flat, x], dim=1)

        return x_with_prompts

    def select_prompts_from_query(self, query):
        if query.dim() == 3:
            query = query.mean(dim=1)
        batch_size = query.size(0)
        query_norm = F.normalize(query, p=2, dim=1)
        keys_norm = F.normalize(self.keys, p=2, dim=1)
        similarity = torch.matmul(query_norm, keys_norm.T)
        if self.training:
            similarity = similarity + (0.02 * torch.randn_like(similarity))
        k = min(self.top_k, similarity.size(1))
        _, top_k_indices = similarity.topk(k, dim=1)
        expanded_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1)
        expanded_indices = expanded_indices.expand(-1, -1, self.prompt_length, self.d_model)
        prompts_expanded = self.prompts.unsqueeze(0).expand(batch_size, -1, -1, -1)
        selected_prompts = torch.gather(prompts_expanded, 1, expanded_indices)
        selected_prompts_flat = selected_prompts.reshape(batch_size, -1, self.d_model)
        if self.training:
            for idx in top_k_indices.flatten():
                self.prompt_usage[idx] += 1
        return selected_prompts_flat, top_k_indices