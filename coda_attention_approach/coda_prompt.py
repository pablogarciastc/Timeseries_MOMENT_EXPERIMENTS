import torch
import torch.nn as nn
import torch.nn.functional as F


class CODAPromptPool(nn.Module):

    def __init__(
            self,
            n_tasks,
            pool_size=10,
            prompt_length=5,
            d_model=512,
            top_k=5,
            ortho_init=True,
            use_g_prompt=True,
            use_e_prompt=True
    ):
        super().__init__()

        self.n_tasks = n_tasks
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.d_model = d_model
        self.top_k = top_k
        self.use_g_prompt = use_g_prompt
        self.use_e_prompt = use_e_prompt

        if self.use_g_prompt:
            self.g_prompts = nn.ParameterList([
                nn.Parameter(torch.randn(prompt_length, d_model) * 0.02)
                for _ in range(n_tasks)
            ])
            print(f"G-Prompt: {n_tasks} task-specific prompts of length {prompt_length}")

        if self.use_e_prompt:
            self.e_prompts = nn.Parameter(
                torch.randn(pool_size, prompt_length, d_model) * 0.02
            )

            self.e_keys = nn.Parameter(
                torch.randn(pool_size, d_model) * 0.02
            )

            if ortho_init:
                nn.init.orthogonal_(self.e_keys)

            self.register_buffer('e_prompt_usage', torch.zeros(pool_size))

            print(f"E-Prompt: {pool_size} shared prompts, top-{top_k} selection")

        self.scale = d_model ** -0.5

        print(f"CODA-Prompt initialized: d_model={d_model}")

    def freeze_g_prompt(self, task_id):
        if self.use_g_prompt and task_id < len(self.g_prompts):
            self.g_prompts[task_id].requires_grad = False

    def select_e_prompts(self, query):
        if not self.use_e_prompt:
            return None, None, None

        batch_size = query.size(0)

        query_norm = F.normalize(query, p=2, dim=1)
        keys_norm = F.normalize(self.e_keys, p=2, dim=1)

        similarity = torch.matmul(query_norm, keys_norm.T)

        if self.training:
            noise = torch.randn_like(similarity) * 0.01
            similarity = similarity + noise

        selection_scores = similarity
        _, top_k_indices = similarity.topk(self.top_k, dim=1)

        expanded_indices = top_k_indices.unsqueeze(-1).unsqueeze(-1)
        expanded_indices = expanded_indices.expand(-1, -1, self.prompt_length, self.d_model)

        prompts_expanded = self.e_prompts.unsqueeze(0).expand(batch_size, -1, -1, -1)
        selected_prompts = torch.gather(prompts_expanded, 1, expanded_indices)

        if self.training:
            with torch.no_grad():
                for idx in top_k_indices.flatten():
                    self.e_prompt_usage[idx] += 1

        return selected_prompts, top_k_indices, selection_scores

    def incorporate_prompts_with_attention(self, x, g_prompt=None, e_prompts=None):
        batch_size, seq_len, d_model = x.shape
        prompts_to_concat = []

        if g_prompt is not None:
            g_prompt_expanded = g_prompt.unsqueeze(0).expand(batch_size, -1, -1)
            prompts_to_concat.append(g_prompt_expanded)

        if e_prompts is not None:
            e_prompts_flat = e_prompts.reshape(batch_size, -1, d_model)
            prompts_to_concat.append(e_prompts_flat)

        if len(prompts_to_concat) > 0:
            all_prompts = torch.cat(prompts_to_concat, dim=1)
            x_with_prompts = torch.cat([all_prompts, x], dim=1)
        else:
            x_with_prompts = x

        return x_with_prompts

    def forward(self, x, task_id=None, return_selection_info=False):
        if x.dim() == 2:
            query = x
            is_query_only = True
        else:
            query = x.mean(dim=1)
            is_query_only = False

        batch_size = query.size(0)

        g_prompt = None
        if self.use_g_prompt and task_id is not None:
            if isinstance(task_id, int):
                g_prompt = self.g_prompts[task_id]
            else:
                g_prompt = self.g_prompts[task_id[0].item()]

        e_prompts = None
        top_k_indices = None
        selection_scores = None

        if self.use_e_prompt:
            e_prompts, top_k_indices, selection_scores = self.select_e_prompts(query)

        if is_query_only:
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
        if not self.use_e_prompt:
            return None

        return {
            'e_prompt_usage': self.e_prompt_usage.cpu().numpy(),
            'total_selections': self.e_prompt_usage.sum().item(),
            'most_used': self.e_prompt_usage.argmax().item(),
            'least_used': self.e_prompt_usage.argmin().item()
        }