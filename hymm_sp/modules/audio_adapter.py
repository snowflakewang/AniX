import torch
from diffusers import ModelMixin
from einops import rearrange
from torch import nn
import math


class AudioProjNet2(ModelMixin):
    def __init__(
        self,
        seq_len=5,
        blocks=12, 
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=4,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = (
            seq_len * blocks * channels
        )  
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_embeds):
          
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )
        context_tokens = self.norm(context_tokens)
        out_all = rearrange(
            context_tokens, "(bz f) m c -> bz f m c", f=video_length
        )

        return out_all


class PerceiverAttentionCA(nn.Module):
    def __init__(self, *, dim=3072, dim_head=64, dim_hidden=3072):
        super().__init__()
        self.dim_head = dim_head
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, dim_hidden * 2, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim, bias=False)
        self.head = dim_hidden // dim_head
        import torch.nn.init as init
        init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            init.zeros_(self.to_out.bias)

    def head_to_batch_dim(self, tensor):
        if len(tensor.shape) == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
            head_size = self.head
            tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, self.dim_head)
            tensor = tensor.permute(0, 2, 1, 3)
        else:
            batch_size, head_size, seq_len, dim = tensor.shape
            tensor = tensor.permute(0, 2, 1, 3)
            tensor = tensor.reshape(batch_size, seq_len, dim * head_size)
        return tensor

    def forward(self, x, latents):
        bsz = x.shape[0]
        x = rearrange(x, "b t aa c -> (b t) aa c")
        latents = rearrange(latents, "b t hw c -> (b t) hw c")
        x = self.norm1(x)
        latents = self.norm2(latents)

        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = self.head_to_batch_dim(q)
        v = self.head_to_batch_dim(v)
        k = self.head_to_batch_dim(k)
        
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = self.head_to_batch_dim(out)
        out = self.to_out(out)
        out = rearrange(out, "(b t) hw c -> b t hw c", b=bsz)
        return out