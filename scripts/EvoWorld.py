import os
import sys

import torch
import torch.nn as nn
from typing import Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Evo1 import EVO1
from config import EvoConfig
from model.world_model import (
    DinoTextWorldModel,
    FiLMGenerator,
    SIGReg,
    WorldModelOutput,
    apply_film,
    build_film_condition,
)


class EVOWorld(EVO1):
    def __init__(self, config: EvoConfig):
        super().__init__(config)
        self.world_model = DinoTextWorldModel(
            image_encoder_name=config.world_image_encoder_name,
            text_encoder_name=config.world_text_encoder_name,
            embed_dim=config.world_embed_dim,
            hidden_dim=config.world_hidden_dim,
            freeze_image_encoder=config.freeze_world_image_encoder,
            freeze_text_encoder=config.freeze_world_text_encoder,
        ).to(self._device)
        self.film_generator = FiLMGenerator(
            condition_dim=config.world_embed_dim * 2,
            target_dim=config.embed_dim,
            hidden_dim=config.world_hidden_dim,
            scale=config.world_film_scale,
        ).to(self._device)
        self.sigreg = SIGReg().to(self._device)

    def compute_world_output(
        self,
        wm_current_images: torch.Tensor,
        wm_future_images: torch.Tensor,
        prompts: list[str],
    ) -> WorldModelOutput:
        return self.world_model(wm_current_images, wm_future_images, prompts)

    def compute_world_output_for_inference(
        self,
        wm_current_images: torch.Tensor,
        prompts: list[str],
    ) -> WorldModelOutput:
        z_t = self.world_model._encode_images(wm_current_images)
        text_emb = self.world_model._encode_text(prompts, device=wm_current_images.device)
        z_pred = self.world_model.predictor(torch.cat([z_t, text_emb], dim=-1))
        return WorldModelOutput(z_t=z_t, z_h=z_pred, z_pred=z_pred, text_emb=text_emb)

    def compute_world_loss(self, output: WorldModelOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_loss = self.world_model.pred_loss(output)
        sigreg_loss = self.sigreg(torch.stack([output.z_t, output.z_h], dim=0))
        world_loss = pred_loss + self.config.world_sigreg_weight * sigreg_loss
        return world_loss, {
            "world_pred_loss": pred_loss.detach(),
            "world_sigreg_loss": sigreg_loss.detach(),
            "world_loss": world_loss.detach(),
        }

    def apply_world_condition(self, fused_tokens: torch.Tensor, output: WorldModelOutput) -> torch.Tensor:
        condition = build_film_condition(
            output.z_t,
            output.z_pred,
            detach=self.config.detach_world_for_action,
        ).to(device=fused_tokens.device, dtype=fused_tokens.dtype)
        gamma, beta = self.film_generator(condition)
        return apply_film(fused_tokens, gamma.to(fused_tokens.dtype), beta.to(fused_tokens.dtype))

    @torch.no_grad()
    def run_inference(
        self,
        images,
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None,
        wm_current_image: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if wm_current_image is None:
            raise ValueError("EVOWorld inference requires `wm_current_image` from the front view.")

        if image_mask.dim() == 1:
            image_mask = image_mask.unsqueeze(0)
        if wm_current_image.ndim == 3:
            wm_current_images = wm_current_image.unsqueeze(0)
        else:
            wm_current_images = wm_current_image
        wm_current_images = wm_current_images.to(device=self._device, dtype=torch.float32)

        fused_tokens = self.get_vl_embeddings(
            images=[images],
            image_mask=image_mask,
            prompt=[prompt],
            return_cls_only=return_cls_only,
        )
        world_output = self.compute_world_output_for_inference(
            wm_current_images=wm_current_images,
            prompts=[prompt],
        )
        fused_tokens = self.apply_world_condition(fused_tokens, world_output)

        state_tensor = self.prepare_state(state_input)
        action = self.predict_action(fused_tokens, state_tensor, action_mask=action_mask)
        if isinstance(action, torch.Tensor) and action.dtype == torch.bfloat16:
            action = action.to(torch.float32)
        return action

    def set_finetune_flags(self):
        super().set_finetune_flags()
        if self.config.freeze_world_model:
            self.world_model.freeze_image_encoder = True
            self.world_model.freeze_text_encoder = True
            self.world_model.set_encoder_trainable()
            self._set_module_trainable(self.world_model.image_projector, False, "World image projector")
            self._set_module_trainable(self.world_model.text_projector, False, "World text projector")
            self._set_module_trainable(self.world_model.predictor, False, "World predictor")
        else:
            self.world_model.freeze_image_encoder = self.config.freeze_world_image_encoder
            self.world_model.freeze_text_encoder = self.config.freeze_world_text_encoder
            self.world_model.set_encoder_trainable()
            self._set_module_trainable(self.world_model.image_projector, True, "World image projector")
            self._set_module_trainable(self.world_model.text_projector, True, "World text projector")
            self._set_module_trainable(self.world_model.predictor, True, "World predictor")
        self._set_module_trainable(self.film_generator, True, "World FiLM generator")
