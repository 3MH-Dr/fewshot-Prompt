import torch
import torch.nn as nn


class CODA_Prompt(nn.Module):
    def __init__(self, length, embed_dim, embedding_key, pool_size, top_k, prompt_init='uniform', prompt_pool=True,
                 prompt_key=True, batchwise_prompt=True, prompt_key_init='uniform', ):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        prompt_pool_shape = (pool_size, length, embed_dim)
        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
        nn.init.uniform_(self.prompt, -1, 1)

        attention_shape = (pool_size, embed_dim)
        self.attention = nn.Parameter(torch.randn(attention_shape))
        nn.init.uniform_(self.attention, -1, 1)
        
        key_shape = (pool_size, embed_dim)
        self.prompt_key = nn.Parameter(torch.randn(key_shape))
        nn.init.uniform_(self.prompt_key, -1, 1)



    def forward(self, x_embed, iseval, task_count):
        out = dict()
        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        else:
            raise NotImplementedError("Not supported way of calculating embedding keys!")

        s = int(task_count * self.top_k)
        f = int((task_count + 1) * self.top_k)

        # freeze/control past tasks
        if not iseval:
            if task_count > 0:
                prompt_key = torch.cat((self.prompt_key[:s].detach().clone(), self.prompt_key[s:f]), dim=0)
                attention = torch.cat((self.attention[:s].detach().clone(), self.attention[s:f]), dim=0)
                prompt = torch.cat((self.prompt[:s].detach().clone(), self.prompt[s:f]), dim=0)
            else:
                prompt_key = self.prompt_key[s:f]
                attention = self.attention[s:f]
                prompt = self.prompt[s:f]
        else:
            prompt_key = self.prompt_key[0:f]
            attention = self.attention[0:f]
            prompt = self.prompt[0:f]

        # with attention and cosine sim
        # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
        a_querry = torch.einsum('bd,kd->bkd', x_embed_mean, attention)
        # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
        n_K = nn.functional.normalize(prompt_key, dim=1)
        q = nn.functional.normalize(a_querry, dim=2)
        aq_k = torch.einsum('bkd,kd->bk', q, n_K)
        # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
        P_ = torch.einsum('bk,kld->bld', aq_k, prompt)


        # The input with the prompt concatenated to the front. [B, prompt+support, C]
        out['total_prompt_len'] = P_.shape[1]
        out['prompted_embedding'] = torch.cat([P_, x_embed], dim=1)

        return out['prompted_embedding'], None
