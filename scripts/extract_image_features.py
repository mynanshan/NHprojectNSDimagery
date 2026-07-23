#!/usr/bin/env python
"""Cache multilayer ViT features for core NSD or ordinary image manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nsdimagery.encoding import pool_transformer_hidden_state  # noqa: E402


def resolved_path(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value).expanduser().resolve()


def parse_integer_list(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("provide at least one comma-separated integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract hidden-state spatial-pyramid features from a Hugging Face "
            "vision transformer. The output rows preserve manifest order."
        )
    )
    parser.add_argument("--manifest", type=resolved_path, required=True)
    parser.add_argument("--output", type=resolved_path, required=True)
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument(
        "--layers",
        type=parse_integer_list,
        default=(3, 6, 9, 12),
        help="Hidden-state indices, including 0=patch embedding (default: 3,6,9,12)",
    )
    parser.add_argument(
        "--pyramid-levels",
        type=parse_integer_list,
        default=(1, 2),
        help="Adaptive spatial-pooling grids (default: 1,2)",
    )
    parser.add_argument(
        "--no-cls", action="store_true", help="Exclude the CLS token"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--nsd-stimuli",
        type=resolved_path,
        help=(
            "Path to nsd_stimuli.hdf5; manifest must then contain nsd_73k_id. "
            "Without this option the manifest must contain image_path."
        ),
    )
    parser.add_argument(
        "--image-root",
        type=resolved_path,
        help="Base for relative image_path values (default: manifest directory)",
    )
    return parser.parse_args()


def as_rgb_image(values: np.ndarray) -> Image.Image:
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image; found {values.shape}")
    if values.shape[0] in {1, 3, 4} and values.shape[-1] not in {1, 3, 4}:
        values = np.moveaxis(values, 0, -1)
    if values.shape[-1] == 1:
        values = np.repeat(values, 3, axis=-1)
    if values.shape[-1] == 4:
        values = values[..., :3]
    if values.shape[-1] != 3:
        raise ValueError(f"Image channel shape is unsupported: {values.shape}")
    if np.issubdtype(values.dtype, np.floating):
        if values.min() < 0:
            values = (values + 1) / 2
        if values.max() <= 1:
            values = np.rint(np.clip(values, 0, 1) * 255)
    return Image.fromarray(values.astype(np.uint8), mode="RGB")


def image_batches(
    manifest: pd.DataFrame,
    *,
    batch_size: int,
    nsd_stimuli: Path | None,
    image_root: Path,
):
    if nsd_stimuli is not None:
        if "nsd_73k_id" not in manifest:
            raise ValueError("Core-NSD manifest needs an nsd_73k_id column")
        if not nsd_stimuli.is_file():
            raise FileNotFoundError(nsd_stimuli)
        with h5py.File(nsd_stimuli, "r") as handle:
            images = handle["imgBrick"]
            for start in range(0, len(manifest), batch_size):
                ids = manifest["nsd_73k_id"].iloc[start : start + batch_size]
                yield [as_rgb_image(images[int(value) - 1]) for value in ids]
    else:
        if "image_path" not in manifest:
            raise ValueError("Image manifest needs an image_path column")
        for start in range(0, len(manifest), batch_size):
            batch = []
            for value in manifest["image_path"].iloc[start : start + batch_size]:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = image_root / path
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as opened:
                    batch.append(opened.convert("RGB"))
            yield batch


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.manifest = args.manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Image manifest not found: {args.manifest}")
    if args.nsd_stimuli is not None:
        args.nsd_stimuli = args.nsd_stimuli.expanduser().resolve()
        if not args.nsd_stimuli.is_file():
            raise FileNotFoundError(
                f"NSD stimulus HDF5 not found: {args.nsd_stimuli}\n"
                "The beta-only download option --skip-stimuli does not install "
                "this 36.84 GiB file. Run download_core_nsd_encoder_mvp.sh "
                "with --stimuli-only, or pass the actual external HDF5 path."
            )
    if args.image_root is not None:
        args.image_root = args.image_root.expanduser().resolve()
    manifest = pd.read_csv(args.manifest)
    if manifest.empty:
        raise ValueError("manifest is empty")
    if "row_id" not in manifest:
        manifest.insert(0, "row_id", np.arange(len(manifest)))
    if manifest["row_id"].duplicated().any():
        raise ValueError("manifest row_id values must be unique")

    from transformers import (
        AutoImageProcessor,
        AutoModel,
        CLIPVisionModel,
        __version__ as transformers_version,
    )

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    if "clip" in args.model_id.lower():
        model = CLIPVisionModel.from_pretrained(args.model_id)
    else:
        model = AutoModel.from_pretrained(args.model_id)
    model = model.to(device).eval()

    image_size = getattr(model.config, "image_size", None)
    patch_size = getattr(model.config, "patch_size", None)
    if isinstance(image_size, (tuple, list)):
        image_height, image_width = (int(value) for value in image_size)
    else:
        image_height = image_width = int(image_size)
    if isinstance(patch_size, (tuple, list)):
        patch_height, patch_width = (int(value) for value in patch_size)
    else:
        patch_height = patch_width = int(patch_size)
    grid_size = (image_height // patch_height, image_width // patch_width)

    all_features = []
    root = args.image_root or args.manifest.parent
    seen = 0
    for images in image_batches(
        manifest,
        batch_size=args.batch_size,
        nsd_stimuli=args.nsd_stimuli,
        image_root=root,
    ):
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(pixel_values=pixel_values, output_hidden_states=True)
            hidden_states = output.hidden_states
            resolved_layers = tuple(
                layer if layer >= 0 else len(hidden_states) + layer
                for layer in args.layers
            )
            if any(layer < 0 or layer >= len(hidden_states) for layer in resolved_layers):
                raise ValueError(
                    f"Requested layers {args.layers}; model exposes 0 through "
                    f"{len(hidden_states) - 1}"
                )
            pieces = [
                pool_transformer_hidden_state(
                    hidden_states[layer],
                    grid_size=grid_size,
                    pyramid_levels=args.pyramid_levels,
                    include_cls=not args.no_cls,
                )
                for layer in resolved_layers
            ]
            batch_features = torch.cat(pieces, dim=1).float().cpu().numpy()
        all_features.append(batch_features)
        seen += len(images)
        print(f"Extracted {seen}/{len(manifest)} images", flush=True)

    features = np.concatenate(all_features).astype(np.float32, copy=False)
    metadata = {
        "model_id": args.model_id,
        "hidden_state_layers": list(args.layers),
        "resolved_hidden_state_layers": list(resolved_layers),
        "pyramid_levels": list(args.pyramid_levels),
        "include_cls": not args.no_cls,
        "grid_size": list(grid_size),
        "feature_dimension": int(features.shape[1]),
        "transformers_version": transformers_version,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        features=features,
        row_id=manifest["row_id"].to_numpy(),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"Wrote {args.output}: {features.shape}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
