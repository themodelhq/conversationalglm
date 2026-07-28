from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file, save_file
from models.configuration_glm import GLMConfig

@dataclass
class GLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    hidden_states: Tensor | None = None
    emotion_logits: Tensor | None = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_positions).float()
        freqs = torch.outer(pos, inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        cos = self.cos[positions].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
        sin = self.sin[positions].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
        a, b = x[..., ::2], x[..., 1::2]
        out = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
        return out.flatten(-2)

class Attention(nn.Module):
    def __init__(self, config: GLMConfig):
        super().__init__()
        self.heads = config.num_attention_heads; self.dim = config.hidden_size // self.heads
        if config.hidden_size % self.heads: raise ValueError("hidden_size must divide num_attention_heads")
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.dim, config.max_position_embeddings)
        self.dropout = config.dropout
    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        b, t, h = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b,t,self.heads,self.dim).transpose(1,2); k = k.view(b,t,self.heads,self.dim).transpose(1,2); v = v.view(b,t,self.heads,self.dim).transpose(1,2)
        positions = torch.arange(t, device=x.device)
        q, k = self.rope(q, positions), self.rope(k, positions)
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        if attention_mask is not None:
            padding = ~attention_mask.to(torch.bool)[:, None, None, :]
            causal = causal[None, None, :, :] | padding
        else: causal = causal[None, None, :, :]
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.dim)
        scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
        probs = F.dropout(F.softmax(scores, dim=-1), self.dropout, self.training)
        return self.out((probs @ v).transpose(1,2).contiguous().view(b,t,h))

class MLP(nn.Module):
    def __init__(self, config: GLMConfig):
        super().__init__(); self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False); self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False); self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    def forward(self, x: Tensor) -> Tensor: return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, config: GLMConfig):
        super().__init__(); self.n1 = RMSNorm(config.hidden_size); self.attn = Attention(config); self.n2 = RMSNorm(config.hidden_size); self.mlp = MLP(config)
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.n1(x), mask); return x + self.mlp(self.n2(x))

class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim))
    def forward(self, x: Tensor) -> Tensor: return self.net(x)

class ConversationalGLM(nn.Module):
    def __init__(self, config: GLMConfig):
        super().__init__(); self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.vision_projector = ModalityProjector(config.vision_hidden_size, config.hidden_size)
        self.audio_projector = ModalityProjector(config.audio_hidden_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size); self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.emotion_head = nn.Linear(config.hidden_size, config.num_emotions)
        if config.tie_word_embeddings: self.lm_head.weight = self.embed_tokens.weight
        self.gradient_checkpointing = config.gradient_checkpointing
        self.apply(self._init)
    def _init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear): nn.init.normal_(module.weight, std=0.02); module.bias is not None and nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): nn.init.normal_(module.weight, std=0.02)
    def _inject(self, embeds: Tensor, ids: Tensor, features: Tensor | None, token_id: int, projector: nn.Module) -> Tensor:
        if features is None: return embeds
        projected = projector(features)
        for batch in range(ids.shape[0]):
            slots = (ids[batch] == token_id).nonzero(as_tuple=False).flatten()
            count = min(len(slots), projected.shape[1])
            if count: embeds[batch, slots[:count]] = projected[batch, :count]
        return embeds
    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None, labels: Tensor | None = None, vision_features: Tensor | None = None, audio_features: Tensor | None = None) -> GLMOutput:
        x = self.embed_tokens(input_ids)
        x = self._inject(x, input_ids, vision_features, self.config.image_token_id, self.vision_projector)
        x = self._inject(x, input_ids, audio_features, self.config.audio_token_id, self.audio_projector)
        for layer in self.layers:
            x = checkpoint(layer, x, attention_mask, use_reentrant=False) if self.gradient_checkpointing and self.training else layer(x, attention_mask)
        x = self.norm(x); logits = self.lm_head(x); loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
        return GLMOutput(logits=logits, loss=loss, hidden_states=x, emotion_logits=self.emotion_head(x[:, -1]))
    @torch.inference_mode()
    def generate(self, input_ids: Tensor, attention_mask: Tensor | None = None, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9, eos_token_id: int | None = None) -> Tensor:
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.config.max_position_embeddings:], None).logits[:, -1]
            if temperature <= 0: next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_idx = probs.sort(descending=True)
                cutoff = sorted_probs.cumsum(-1) - sorted_probs > top_p
                sorted_probs[cutoff] = 0; sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                next_id = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            ids = torch.cat([ids, next_id], dim=1)
            if bool((next_id == eos).all()): break
        return ids
    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True); self.config.save_pretrained(directory); save_file(self.state_dict(), str(directory / "model.safetensors"))
    @classmethod
    def from_pretrained(cls, directory: str | Path, device: str | torch.device = "cpu") -> "ConversationalGLM":
        directory = Path(directory); model = cls(GLMConfig.from_file(directory / "config.json")); model.load_state_dict(load_file(str(directory / "model.safetensors"), device=str(device))); return model.to(device).eval()
