# evoworld_server_json.py

import sys
import os
import asyncio
import logging
import websockets
import numpy as np
import cv2
import json
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from fvcore.nn import FlopCountAnalysis



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.EvoWorld import EVOWorld
from dataset.lerobot_dataset_pretrain_mp import IMAGENET_MEAN, IMAGENET_STD, NormalizationType



class Normalizer:
    def __init__(self, stats_or_path, normalization_type: NormalizationType = NormalizationType.BOUNDS):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                self.stats_map = json.load(f)
        else:
            self.stats_map = stats_or_path

        if isinstance(normalization_type, str):
            normalization_type = NormalizationType(normalization_type)
        self.normalization_type = normalization_type
        print(f"Using normalization type: {self.normalization_type}")
        self.target_dim = 24
        self._cache_stats = {}

    def _pad_vector(self, values, name):
        tensor = torch.tensor(values, dtype=torch.float32)
        length = tensor.shape[0]
        if length < self.target_dim:
            pad = torch.zeros(self.target_dim - length, dtype=torch.float32)
            tensor = torch.cat([tensor, pad], dim=0)
        elif length > self.target_dim:
            raise ValueError(f"{name} length {length} exceeds expected {self.target_dim}")
        return tensor

    def _prepare_stats(self, stats_dict, stats_name):
        prepared = {}
        for key, values in stats_dict.items():
            prepared[key] = self._pad_vector(values, f"{stats_name}.{key}")
        return prepared

    def _stat_to_device(self, stats_dict, key, device, dtype):
        tensor = stats_dict.get(key)
        if tensor is None:
            return None
        return tensor.to(device=device, dtype=dtype)
    
    def _get_stats_for(self, arm_key, dataset_key, stats_type):
        """get the relevant stats dict for the given arm/dataset and stat type (state/action)"""
        cache_key = (arm_key, dataset_key, stats_type)
        if cache_key in self._cache_stats:
            return self._cache_stats[cache_key]
        
        if arm_key not in self.stats_map:
             raise ValueError(f"Arm key '{arm_key}' not found in normalization stats.")
             
        if "observation.state" in self.stats_map[arm_key] or "action" in self.stats_map[arm_key]:
            raw_stats = self.stats_map[arm_key]
        else:
            if dataset_key not in self.stats_map[arm_key]:
                 raise ValueError(f"Dataset key '{dataset_key}' not found in normalization stats for arm '{arm_key}'.")
            raw_stats = self.stats_map[arm_key][dataset_key]
        
        dict_key = "observation.state" if stats_type == "state" else "action"
        
        if dict_key not in raw_stats:
            raise ValueError(f"Key '{dict_key}' not found in stats for {arm_key}/{dataset_key}")
            
        prepared = self._prepare_stats(raw_stats[dict_key], dict_key)
        self._cache_stats[cache_key] = prepared
        return prepared

    def _normalize_tensor(self, tensor: torch.Tensor, stats_dict, clamp: bool) -> torch.Tensor:
        eps = 1e-8
        device, dtype = tensor.device, tensor.dtype
        norm_type = self.normalization_type
        
        current_dim = tensor.shape[-1]

        if norm_type == NormalizationType.NORMAL:
            mean = self._stat_to_device(stats_dict, "mean", device, dtype)
            std = self._stat_to_device(stats_dict, "std", device, dtype)
            if mean is not None: mean = mean[..., :current_dim]
            if std is not None: std = std[..., :current_dim]
            
            if mean is None or std is None:
                raise ValueError("Normal normalization selected but mean/std are missing in norm_stats.json")
            return (tensor - mean) / (std + eps)

        low_key, high_key = ("min", "max")
        if norm_type == NormalizationType.BOUNDS_Q99:
            low_key, high_key = ("q01", "q99")

        low = self._stat_to_device(stats_dict, low_key, device, dtype)
        high = self._stat_to_device(stats_dict, high_key, device, dtype)

        if (low is None or high is None) and norm_type == NormalizationType.BOUNDS_Q99:
            logging.warning("Missing q01/q99 stats; falling back to min/max bounds normalization.")
            low = self._stat_to_device(stats_dict, "min", device, dtype)
            high = self._stat_to_device(stats_dict, "max", device, dtype)

        if low is None or high is None:
            raise ValueError("Bounds normalization selected but min/max stats are missing in norm_stats.json")

        low = low[..., :current_dim]
        high = high[..., :current_dim]

        normalized = 2 * (tensor - low) / (high - low + eps) - 1
        if clamp:
            normalized = torch.clamp(normalized, -1.0, 1.0)
        return normalized

    def _denormalize_tensor(self, tensor: torch.Tensor, stats_dict) -> torch.Tensor:
        eps = 1e-8
        device, dtype = tensor.device, tensor.dtype
        norm_type = self.normalization_type

        if norm_type == NormalizationType.NORMAL:
            mean = self._stat_to_device(stats_dict, "mean", device, dtype)
            std = self._stat_to_device(stats_dict, "std", device, dtype)
            if mean is None or std is None:
                raise ValueError("Normal denormalization requested but mean/std stats are missing")
            return tensor * (std + eps) + mean

        low_key, high_key = ("min", "max")
        if norm_type == NormalizationType.BOUNDS_Q99:
            low_key, high_key = ("q01", "q99")

        low = self._stat_to_device(stats_dict, low_key, device, dtype)
        high = self._stat_to_device(stats_dict, high_key, device, dtype)

        if (low is None or high is None) and norm_type == NormalizationType.BOUNDS_Q99:
            logging.warning("Missing q01/q99 stats; falling back to min/max bounds denormalization.")
            low = self._stat_to_device(stats_dict, "min", device, dtype)
            high = self._stat_to_device(stats_dict, "max", device, dtype)

        if low is None or high is None:
            raise ValueError("Bounds denormalization requested but min/max stats are missing")
        
        current_dim = tensor.shape[-1]
        if low.shape[-1] > current_dim:
            low = low[..., :current_dim]
            high = high[..., :current_dim]

        return (tensor + 1.0) / 2.0 * (high - low + eps) + low

    def normalize_state(self, state: torch.Tensor, arm_key: str, dataset_key: str) -> torch.Tensor:
        stats = self._get_stats_for(arm_key, dataset_key, "state")
        norm_state = self._normalize_tensor(state, stats, clamp=True)

        if norm_state.shape[-1] < self.target_dim:
            padding_size = self.target_dim - norm_state.shape[-1]
            pad_tensor = torch.zeros(
                (*norm_state.shape[:-1], padding_size), 
                dtype=norm_state.dtype, 
                device=norm_state.device
            )
            norm_state = torch.cat([norm_state, pad_tensor], dim=-1)
            
        return norm_state

    def denormalize_action(self, action: torch.Tensor, arm_key: str, dataset_key: str) -> torch.Tensor:
        if action.ndim == 1:
            action = action.view(1, -1)
        stats = self._get_stats_for(arm_key, dataset_key, "action")
        denorm_action = self._denormalize_tensor(action, stats)

        # Padding if action dim is less than target_dim
        if denorm_action.shape[-1] < self.target_dim:
            padding_size = self.target_dim - denorm_action.shape[-1]
            pad_tensor = torch.zeros(
                (*denorm_action.shape[:-1], padding_size), 
                dtype=denorm_action.dtype, 
                device=denorm_action.device
            )
            denorm_action = torch.cat([denorm_action, pad_tensor], dim=-1)

        return denorm_action


# Config fields whose value is passed straight to HuggingFace `from_pretrained`.
# A checkpoint's config.json may carry the absolute local path of the machine it
# was trained on (e.g. /data/.../pretrain_model/dinov2-small). Those paths don't
# exist for anyone who downloads the checkpoint from HF, so from_pretrained() would
# crash on load. See `_resolve_pretrained_names`.
_PRETRAINED_NAME_FIELDS = ("vlm_name", "world_image_encoder_name", "world_text_encoder_name")


def _resolve_pretrained_names(config_dict):
    """Fall back to the HF repo-id when a pretrained-encoder name is a missing local path.

    If a field holds an absolute path that is NOT present on this machine, replace it
    with the `EvoConfig` dataclass default (an HF repo-id such as facebook/dinov2-small).
    An absolute path that *does* exist is left untouched (back-compat on the training
    machine); relative names / HF repo-ids are never touched.
    """
    import dataclasses
    from config import EvoConfig

    defaults = {f.name: f.default for f in dataclasses.fields(EvoConfig)}
    for key in _PRETRAINED_NAME_FIELDS:
        val = config_dict.get(key)
        if isinstance(val, str) and os.path.isabs(val) and not os.path.exists(val):
            fallback = defaults.get(key)
            print(
                f"[server] config.json '{key}' is a missing local path ({val!r}); "
                f"falling back to HF repo-id {fallback!r}.",
                flush=True,
            )
            config_dict[key] = fallback
    return config_dict


def load_model_and_normalizer(ckpt_dir):
    config_dict = json.load(open(os.path.join(ckpt_dir, "config.json")))
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))

    config_dict["finetune_vlm"] = False
    config_dict["finetune_action_head"] = False
    config_dict["num_inference_timesteps"] = 32
    config_dict = _resolve_pretrained_names(config_dict)

    from config import EvoConfig
    config = EvoConfig.from_dict(config_dict)
    if not config.use_world_model:
        raise ValueError("EvoWorld_server.py expects a checkpoint with use_world_model=True.")
    model = EVOWorld(config).eval()
    ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["module"], strict=True)
    model = model.to("cuda")

    normalization_type = config_dict.get("normalization_type", NormalizationType.BOUNDS.value)
    normalizer = Normalizer(stats, normalization_type=normalization_type)
    return model, normalizer


WORLD_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def decode_pil_from_list(img_list, image_size=None):
    img_array = np.array(img_list, dtype=np.uint8)
    img = img_array if image_size is None else cv2.resize(img_array, (image_size, image_size))
    # # 保存 convert 之前的 img
    # img_before_convert = img.copy()
    # # 保存到本地文件夹
    # os.makedirs("debug_images", exist_ok=True)
    # debug_image_path = os.path.join("debug_images", f"debug_{len(os.listdir('debug_images'))}.png")
    # cv2.imwrite(debug_image_path, img_before_convert)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # 保存 convert 之后的 img
    # debug_image_path_converted = os.path.join("debug_images", f"debug_converted_{len(os.listdir('debug_images'))}.png")
    # cv2.imwrite(debug_image_path_converted, img)
    return Image.fromarray(img)


def decode_image_from_list(img_list):
    pil = decode_pil_from_list(img_list, 448)
    return transforms.ToTensor()(pil).to("cuda")


def decode_world_image_from_list(img_list):
    pil = decode_pil_from_list(img_list)
    return WORLD_IMAGE_TRANSFORM(pil).to("cuda")



def infer_from_json_dict(data: dict, model, normalizer, arm_key, dataset_key):
    device = "cuda"
    model_dtype = next(model.parameters()).dtype
    # Per-request keys: RoboTwin batch eval runs many tasks against ONE server, so
    # honor the arm_key/dataset_key the client sends. Falls back to the launch-time
    # args when the payload omits them (backward compatible with other clients).
    arm_key = data.get("arm_key", arm_key)
    dataset_key = data.get("dataset_key", dataset_key)

  
    images = [decode_image_from_list(img) for img in data["image"]]
    assert len(images) == 3, "Must provide exactly 3 images."
    for img in images:
        assert img.shape == (3, 448, 448), "image_size must be (3,448,448)"
    wm_current_image = decode_world_image_from_list(data["image"][0])

 
    state = torch.tensor(data["state"], dtype=torch.float32, device=device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    # if state.shape[1] < 24:
        # state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device=device)], dim=1)
    norm_state = normalizer.normalize_state(state, arm_key, dataset_key).to(dtype=torch.float32)

    
    prompt = data["prompt"]
    image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.tensor([data["action_mask"]],dtype=torch.int32, device=device)

    print(f"image_mask,{image_mask}")
    print(f"action_mask,{action_mask}")
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask,
            wm_current_image=wm_current_image,
        )
        action = action.reshape(1, -1, 24)
        action = normalizer.denormalize_action(action[0], arm_key, dataset_key)
        return action.cpu().numpy().tolist()


async def handle_request(websocket, model, normalizer, arm_key, dataset_key):
    print("Client connected")
    try:
        async for message in websocket:
           
            json_data = json.loads(message)
            print(f"Received JSON observation")
            actions = infer_from_json_dict(json_data, model, normalizer, arm_key, dataset_key)
            await websocket.send(json.dumps(actions))
            print("Sent action chunk")


    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evo-World inference server")
    parser.add_argument("--ckpt_dir", required=True,
                        help="Checkpoint dir containing config.json / norm_stats.json / mp_rank_00_model_states.pt")
    parser.add_argument("--port", type=int, default=8000, help="Websocket port (default 8000, matches client)")
    parser.add_argument("--arm_key", default="metaworld_robot",
                        help="Top-level key in norm_stats.json (default metaworld_robot)")
    parser.add_argument("--dataset_key", default="metaworld_Mint",
                        help="Nested key under arm_key in norm_stats.json (default metaworld_Mint)")
    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    port = args.port
    arm_key = args.arm_key
    dataset_key = args.dataset_key

    print(f"Loading Evo-World model from {ckpt_dir} (arm_key={arm_key}, dataset_key={dataset_key}, port={port})...")
    model, normalizer = load_model_and_normalizer(ckpt_dir)

    async def main():
        print(f"Evo-World server running at ws://0.0.0.0:{port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer, arm_key, dataset_key),
            "0.0.0.0", port, max_size=100_000_000, ping_interval=None
        ):
            await asyncio.Future()

    asyncio.run(main())
