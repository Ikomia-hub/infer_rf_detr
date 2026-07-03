import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import yaml

import torch

from rfdetr.util.coco_classes import COCO_CLASSES
from rfdetr.config import (
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRMediumConfig,
    RFDETRBaseConfig,
    RFDETRLargeConfig,
)
from rfdetr.detr import RFDETR, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge

try:
    import rfdetr.main as rfdetr_main
except ImportError:
    rfdetr_main = None

try:
    from rfdetr.util.files import download_file
except ImportError:
    download_file = None


MODEL_CLASSES = {
    "rf-detr-nano": RFDETRNano,
    "rf-detr-small": RFDETRSmall,
    "rf-detr-medium": RFDETRMedium,
    "rf-detr-base": RFDETRBase,
    "rf-detr-base-2": RFDETRBase,
    "rf-detr-large": RFDETRLarge,
}

MODEL_CONFIG_CLASSES = {
    "rf-detr-nano": RFDETRNanoConfig,
    "rf-detr-small": RFDETRSmallConfig,
    "rf-detr-medium": RFDETRMediumConfig,
    "rf-detr-base": RFDETRBaseConfig,
    "rf-detr-base-2": RFDETRBaseConfig,
    "rf-detr-large": RFDETRLargeConfig,
}


def adjust_input_size(model_name: str, input_size: int) -> tuple:
    """Get input size multiple of block_size."""
    config = MODEL_CONFIG_CLASSES.get(model_name)()
    block_size = config.patch_size * config.num_windows
    return (input_size // block_size) * block_size, block_size


def get_class_names(param) -> tuple:
    if param.model_weight_file:
        if not param.config_file:
            raise ValueError("The config_file 'class_names.yaml' is required when using a custom model file.")
        else:
            with open(param.config_file, 'r') as f:
                config = yaml.safe_load(f)
                classes = list(config.get('classes', []))
                class_ids = None
    else:
        classes = list(COCO_CLASSES.values())
        class_ids = list(COCO_CLASSES.keys())

    return classes, class_ids


@contextmanager
def _chdir(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _weights_dir():
    weights_dir = Path(__file__).resolve().parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def _hosted_models():
    if rfdetr_main is None:
        return {}

    return (
        getattr(rfdetr_main, "HOSTED_MODELS", None)
        or getattr(rfdetr_main, "OPEN_SOURCE_MODELS", None)
        or {}
    )


def _download_with_rfdetr_helper(filename, output_path):
    if rfdetr_main is None:
        return False

    download_pretrain_weights = getattr(rfdetr_main, "download_pretrain_weights", None)
    if download_pretrain_weights is None:
        return False

    with _chdir(output_path.parent):
        download_pretrain_weights(filename)

    return output_path.exists()


def _download_with_url(filename, output_path):
    hosted_models = _hosted_models()
    if filename not in hosted_models or download_file is None:
        return False

    download_file(hosted_models[filename], str(output_path))
    return output_path.exists()


def _copy_from_roboflow_cache(filename, output_path):
    cache_path = Path.home() / ".roboflow" / "models" / filename
    if not cache_path.exists():
        return False

    shutil.copy2(cache_path, output_path)
    print(f"Copied RF-DETR weights from {cache_path} to {output_path}")
    return True

def load_model(param, class_count: int) -> RFDETR:
    """
    Loads the appropriate model architecture with either custom or pre-trained weights.

    Args:
        param: An object containing necessary attributes such as model_weight_file,
               config_file, model_name, and input_size.
        class_count: number of classes the model is trained on

    Returns:
        An instance of the loaded model.
    """
    # Determine model weights and architecture
    if param.model_weight_file and os.path.exists(param.model_weight_file):
        print(f"Using custom weights file: {param.model_weight_file}")
        model_weights = param.model_weight_file

        with open(param.config_file, 'r') as f:
            config = yaml.safe_load(f)

        model_architecture = config.get('model_name', param.model_name)
    else:
        model_weights = download_pretrain_weights(param.model_name)
        model_architecture = param.model_name

    # Select and initialize the model
    model_class = MODEL_CLASSES.get(model_architecture)
    if model_class is None:
        raise ValueError(
            f"Unsupported model architecture: {model_architecture}")

    # Adjust input size
    new_input_size, block_size = adjust_input_size(model_architecture, param.input_size)
    if new_input_size != param.input_size:
        param.input_size = new_input_size
        print(f"Updating input size to {param.input_size} to be a multiple of {block_size}")

    device = "cuda" if param.cuda and torch.cuda.is_available() else "cpu"
    model_kwargs = {
        "resolution": param.input_size,
        "num_classes": class_count,
        "device": device,
    }
    if model_weights is not None:
        model_kwargs["pretrain_weights"] = model_weights

    model = model_class(**model_kwargs)
    return model


def download_pretrain_weights(model_name: str) -> str:
    """Download the pre-trained weights for the specified model if not already available."""
    weights_dir = _weights_dir()
    candidate_filenames = [
        f"{model_name}.pt",
        f"{model_name}.pth",
    ]

    for filename in candidate_filenames:
        output_path = weights_dir / filename
        if output_path.exists():
            print(f"Using existing weights file: {output_path}")
            return str(output_path)

    for filename in candidate_filenames:
        output_path = weights_dir / filename
        if _download_with_rfdetr_helper(filename, output_path):
            print(f"Downloaded RF-DETR weights to {output_path}")
            return str(output_path)

        if _download_with_url(filename, output_path):
            print(f"Downloaded RF-DETR weights to {output_path}")
            return str(output_path)

        if _copy_from_roboflow_cache(filename, output_path):
            return str(output_path)

    print(
        "Could not pre-cache RF-DETR weights in the plugin weights folder. "
        "Falling back to RF-DETR's default weight resolution."
    )
    return None
