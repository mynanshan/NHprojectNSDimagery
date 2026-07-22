"""Minimal GNet image-to-brain inference for Table 1 reproduction.

The model architecture is adapted from the MIT-licensed MindEyeV2 GNet
implementation (Copyright 2023 MedARC). This file intentionally excludes the
reconstruction models and their large dependency tree: it only turns images
into predicted ``nsdgeneral`` beta patterns using ``gnet_multisubject.pt``.
See ``THIRD_PARTY_NOTICES.md`` for the full license notice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch import nn


class _TrunkBlock(nn.Module):
    def __init__(self, features_in: int, features_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features_in, features_out, 3, padding=1)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.bn1 = nn.BatchNorm2d(
            features_in, eps=1e-5, momentum=0.25
        )
        nn.init.xavier_normal_(self.conv1.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.constant_(self.conv1.bias, 0.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(
            self.conv1(self.drop1(self.bn1(values)))
        )


class _PreFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(values))


class _EncoderStage(nn.Module):
    def __init__(self, trunk_width: int = 64, pass_through: int = 192):
        super().__init__()
        self.conv3 = nn.Conv2d(192, 128, kernel_size=3)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.bn1 = nn.BatchNorm2d(192, eps=1e-5, momentum=0.25)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.tw = int(trunk_width)
        self.pt = int(pass_through)
        shared_width = int(trunk_width + pass_through)
        self.conv4a = _TrunkBlock(128, shared_width)
        self.conv5a = _TrunkBlock(shared_width, shared_width)
        self.conv6a = _TrunkBlock(shared_width, shared_width)
        self.conv4b = _TrunkBlock(shared_width, shared_width)
        self.conv5b = _TrunkBlock(shared_width, shared_width)
        self.conv6b = _TrunkBlock(shared_width, self.tw)
        nn.init.xavier_normal_(self.conv3.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.constant_(self.conv3.bias, 0.0)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        c3 = torch.nn.functional.relu(
            self.conv3(self.drop1(self.bn1(values))), inplace=False
        )
        c4a = self.conv4a(c3)
        c4b = self.conv4b(c4a)
        c5a = self.conv5a(self.pool1(c4b))
        c5b = self.conv5b(c5a)
        c6a = self.conv6a(c5b)
        c6b = self.conv6b(c6a)
        width = self.tw
        feature_maps = [
            torch.cat([c3, c4a[:, :width], c4b[:, :width]], dim=1),
            torch.cat(
                [c5a[:, :width], c5b[:, :width], c6a[:, :width], c6b], dim=1
            ),
        ]
        return feature_maps, c6b


class _ImageEncoder(nn.Module):
    def __init__(self, image_mean: np.ndarray):
        super().__init__()
        self.mu = nn.Parameter(
            torch.as_tensor(image_mean, dtype=torch.float32), requires_grad=False
        )
        self.pre = _PreFilter()
        self.enc = _EncoderStage(trunk_width=64, pass_through=192)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        feature_maps, hidden = self.enc(self.pre(values - self.mu))
        return values, feature_maps, hidden


class _VoxelwiseReadout(nn.Module):
    def __init__(self, feature_maps: list[torch.Tensor], n_voxels: int):
        super().__init__()
        self.feature_map_shapes = [list(values.size()) for values in feature_maps]
        self.n_features = int(sum(shape[1] for shape in self.feature_map_shapes))
        self.softmax = nn.Softmax(dim=1)
        self.receptive_fields: list[nn.Parameter] = []
        for index, shape in enumerate(self.feature_map_shapes):
            parameter = nn.Parameter(
                torch.ones((n_voxels, shape[2], shape[2]), dtype=torch.float32)
            )
            self.register_parameter(f"rf{index}", parameter)
            self.receptive_fields.append(parameter)
        self.w = nn.Parameter(torch.empty((n_voxels, self.n_features)))
        self.b = nn.Parameter(torch.empty(n_voxels))

    @staticmethod
    def _signed_log(values: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.abs(values)) * torch.tanh(values)

    def forward(self, feature_maps: list[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for feature_map, receptive_field in zip(
            feature_maps, self.receptive_fields
        ):
            weights = self.softmax(torch.flatten(receptive_field, start_dim=1))
            features = self._signed_log(torch.flatten(feature_map, start_dim=2))
            pooled.append(torch.tensordot(weights, features, dims=[[1], [2]]))
        design = self._signed_log(torch.cat(pooled, dim=2))
        return torch.bmm(
            design, torch.unsqueeze(self.w, 2)
        ).squeeze(-1).transpose(0, 1) + torch.unsqueeze(self.b, 0)


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_chw(image: object) -> np.ndarray:
    """Apply the same RGB conversion and 227-pixel resize used by GNet."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            pil_image = opened.convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        values = _as_numpy(image)
        if values.ndim != 3:
            raise ValueError("Every reconstruction must be a three-dimensional image")
        if values.shape[0] in {1, 3, 4} and values.shape[-1] not in {1, 3, 4}:
            values = np.moveaxis(values, 0, -1)
        if values.shape[-1] == 1:
            values = np.repeat(values, 3, axis=-1)
        if values.shape[-1] == 4:
            values = values[..., :3]
        if values.shape[-1] != 3:
            raise ValueError("Image arrays must have one, three, or four channels")
        values = values.astype(np.float32)
        if values.min() < 0:
            values = (values + 1) / 2
        elif values.max() > 1:
            values = values / 255
        values = np.clip(values, 0, 1)
        pil_image = Image.fromarray(np.rint(values * 255).astype(np.uint8), "RGB")
    resized = pil_image.resize((227, 227), resample=Image.Resampling.LANCZOS)
    values = np.asarray(resized, dtype=np.float32) / 255
    return np.moveaxis(values, -1, 0)


class GNetEncoder:
    """Predict one NSD subject's ``nsdgeneral`` response to RGB images."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        subject: int,
        *,
        device: str = "cuda",
    ):
        if subject not in range(1, 9):
            raise ValueError("subject must be 1 through 8")
        self.subject = int(subject)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        # This checkpoint contains NumPy objects as well as tensor state dicts.
        # Only load the official MedARC checkpoint or another trusted file.
        try:
            self.checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:  # PyTorch before the weights_only argument existed
            self.checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.subject_key = self._find_subject_key(self.checkpoint["val_cc"])

    def _find_subject_key(self, mapping: dict) -> object:
        candidates = (self.subject, str(self.subject), f"subj{self.subject:02d}")
        for candidate in candidates:
            if candidate in mapping:
                return candidate
        raise KeyError(
            f"Could not find subject {self.subject} in GNet checkpoint keys "
            f"{list(mapping)}"
        )

    @property
    def n_voxels(self) -> int:
        return len(self.checkpoint["val_cc"][self.subject_key])

    def predict(
        self,
        images: Iterable[object],
        *,
        batch_size: int = 32,
        voxel_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an ``images x voxels`` array of GNet beta predictions."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        prepared_images = [_image_to_chw(image) for image in images]
        if not prepared_images:
            raise ValueError("images must not be empty")
        image_array = np.stack(prepared_images)

        if voxel_mask is None:
            selected = np.ones(self.n_voxels, dtype=bool)
        else:
            selected = np.asarray(voxel_mask)
            if selected.dtype != bool or selected.shape != (self.n_voxels,):
                raise ValueError("voxel_mask must be boolean and match GNet voxels")
            if not selected.any():
                raise ValueError("voxel_mask selects no voxels")

        best = self.checkpoint["best_params"]
        image_mean = _as_numpy(self.checkpoint["input_mean"]).astype(np.float32)
        encoder = _ImageEncoder(image_mean).to(self.device)
        encoder.load_state_dict(best["enc"])
        encoder.eval()

        with torch.inference_mode():
            example = torch.from_numpy(image_array[:1]).to(self.device)
            _, feature_maps, _ = encoder(example)
        readout = _VoxelwiseReadout(feature_maps, int(selected.sum())).to(self.device)
        readout_subject_key = self._find_subject_key(best["fwrfs"])
        readout_state = best["fwrfs"][readout_subject_key]
        if not selected.all():
            indices = torch.from_numpy(selected)
            readout_state = {
                name: value[indices] for name, value in readout_state.items()
            }
        readout.load_state_dict(readout_state)
        readout.eval()

        predictions = []
        with torch.inference_mode():
            for start in range(0, len(image_array), batch_size):
                batch = torch.from_numpy(
                    image_array[start : start + batch_size]
                ).to(self.device)
                _, feature_maps, _ = encoder(batch)
                predictions.append(readout(feature_maps).cpu().numpy())
        return np.concatenate(predictions, axis=0)
