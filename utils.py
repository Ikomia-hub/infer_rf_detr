import os
import yaml

import torch

from rfdetr.util.coco_classes import COCO_CLASSES
from rfdetr.util.files import download_file
from rfdetr.detr import RFDETR, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge
from rfdetr.main import HOSTED_MODELS


def adjust_to_multiple(value, base=32):
    """Adjust value down to the nearest multiple of 'base'."""
    return (value // base) * base


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
    model_classes = {
        "rf-detr-nano": RFDETRNano,
        "rf-detr-small": RFDETRSmall,
        "rf-detr-medium": RFDETRMedium,
        "rf-detr-base": RFDETRBase,
        "rf-detr-base-2": RFDETRBase,
        "rf-detr-large": RFDETRLarge,
    }

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
    model_class = model_classes.get(model_architecture)
    if model_class is None:
        raise ValueError(
            f"Unsupported model architecture: {model_architecture}")

    device = "cuda" if param.cuda and torch.cuda.is_available() else "cpu"
    model = model_class(
        resolution=param.input_size,
        pretrain_weights=model_weights,
        num_classes=class_count,
        device=device,
    )
    return model


def download_pretrain_weights(model_name: str) -> str:
    """Download the pre-trained weights for the specified model if not already available."""
    # Ensure the weights folder exists
    model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")
    os.makedirs(model_folder, exist_ok=True)
    weight_filename = f"{model_name}.pth"
    weight_path = os.path.join(model_folder, weight_filename)

    # Check if the weight file exists
    if not os.path.exists(weight_path):
        if weight_filename in HOSTED_MODELS:
            print(f"Downloading pre-trained weights for {model_name}...")
            download_file(HOSTED_MODELS[weight_filename], weight_path)
            print(f"Download complete: {weight_path}")
            return weight_path
        else:
            raise ValueError(f"No pre-trained weights available for {model_name}")
    else:
        print(f"Using existing weights file: {weight_path}")
        return weight_path
