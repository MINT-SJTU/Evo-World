from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldModelOutput:
    z_t: torch.Tensor
    z_h: torch.Tensor
    z_pred: torch.Tensor
    text_emb: torch.Tensor


def build_film_condition(z_t: torch.Tensor, z_pred: torch.Tensor, detach: bool = True) -> torch.Tensor:
    if detach:
        z_t = z_t.detach()
        z_pred = z_pred.detach()
    return torch.cat([z_t, z_pred], dim=-1)


def apply_film(fused_tokens: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return fused_tokens * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


def to_module_dtype(x: torch.Tensor, module: nn.Module) -> torch.Tensor:
    param = next(module.parameters(), None)
    if param is None:
        return x
    return x.to(dtype=param.dtype)


class FiLMGenerator(nn.Module):
    def __init__(self, condition_dim: int, target_dim: int, hidden_dim: int, scale: float = 0.1):
        super().__init__()
        self.scale = float(scale)
        self.proj = nn.Linear(condition_dim, target_dim)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(target_dim)
        self.proj_c = nn.Linear(target_dim, target_dim * 2)
        nn.init.zeros_(self.proj_c.weight)
        nn.init.zeros_(self.proj_c.bias)

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = to_module_dtype(condition, self.proj)
        x = self.norm(self.relu(self.proj(condition)))
        gamma, beta = self.proj_c(x).chunk(2, dim=-1)
        if self.scale > 0:
            gamma = self.scale * torch.tanh(gamma)
            beta = self.scale * torch.tanh(beta)
        return gamma, beta


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        # proj: (T, B, D)
        proj = proj.float()
        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0).clamp_min(1e-8))
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


class DinoTextWorldModel(nn.Module):
    def __init__(
        self,
        image_encoder_name: str,
        text_encoder_name: str,
        embed_dim: int,
        hidden_dim: int,
        freeze_image_encoder: bool = True,
        freeze_text_encoder: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel, CLIPTextModel, CLIPTokenizer

        self.image_encoder = AutoModel.from_pretrained(image_encoder_name)
        self.text_tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
        self.freeze_image_encoder = bool(freeze_image_encoder)
        self.freeze_text_encoder = bool(freeze_text_encoder)

        image_dim = self.image_encoder.config.hidden_size
        text_dim = self.text_encoder.config.hidden_size
        self.image_projector = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.set_encoder_trainable()

    def set_encoder_trainable(self) -> None:
        for param in self.image_encoder.parameters():
            param.requires_grad = not self.freeze_image_encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = not self.freeze_text_encoder
        if self.freeze_image_encoder:
            self.image_encoder.eval()
        if self.freeze_text_encoder:
            self.text_encoder.eval()

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_image_encoder else torch.enable_grad()
        with context:
            outputs = self.image_encoder(pixel_values=images)
            cls = outputs.last_hidden_state[:, 0]
        return self.image_projector(to_module_dtype(cls, self.image_projector))

    def _encode_text(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        tokens = self.text_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        context = torch.no_grad() if self.freeze_text_encoder else torch.enable_grad()
        with context:
            pooled = self.text_encoder(**tokens).pooler_output
        return self.text_projector(to_module_dtype(pooled, self.text_projector))

    def forward(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
        prompts: List[str],
    ) -> WorldModelOutput:
        device = current_images.device
        z_t = self._encode_images(current_images)
        z_h = self._encode_images(future_images)
        text_emb = self._encode_text(prompts, device=device)
        z_pred = self.predictor(torch.cat([z_t, text_emb], dim=-1))
        return WorldModelOutput(z_t=z_t, z_h=z_h, z_pred=z_pred, text_emb=text_emb)

    @staticmethod
    def pred_loss(output: WorldModelOutput) -> torch.Tensor:
        return F.mse_loss(output.z_pred.float(), output.z_h.float())
