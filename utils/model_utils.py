import os
import yaml
from infer_rf_detr.rf_detr.rfdetr.util.files import download_file
from infer_rf_detr.rf_detr.rfdetr import RFDETRBase, RFDETRLarge


HOSTED_MODELS = {
    "rf-detr-base": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    "rf-detr-base-2": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth"
}


def load_model(param, n_classes):
    """
    Loads the appropriate model architecture with either custom or pre-trained weights.

    Args:
        param: An object containing necessary attributes such as model_weight_file,
               class_file, model_name, and input_size.

    Returns:
        An instance of the loaded model.
    """

    # Determine model weights and architecture
    if param.model_weight_file and os.path.exists(param.model_weight_file):
        print(f"Using custom weights file: {param.model_weight_file}")
        model_weights = param.model_weight_file

        with open(param.class_file, 'r') as f:
            config = yaml.safe_load(f)
        model_architecture = config.get('model_name', param.model_name)
    else:
        model_weights = download_pretrain_weights(param.model_name)
        model_architecture = param.model_name

    # Select and initialize the model
    model_classes = {
        "rf-detr-base": RFDETRBase,
        "rf-detr-base-2": RFDETRBase,
        "rf-detr-large": RFDETRLarge,
    }

    model_class = model_classes.get(model_architecture)
    if model_class is None:
        raise ValueError(
            f"Unsupported model architecture: {model_architecture}")

    model = model_class(
        resolution=param.input_size,
        pretrain_weights=model_weights,
        num_classes=n_classes-1,  # 0 index
    )

    return model


def download_pretrain_weights(model_name):
    """Download the pre-trained weights for the specified model if not already available."""
    # Ensure the weights folder exists
    model_folder = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "..", "weights")
    os.makedirs(model_folder, exist_ok=True)
    weight_filename = f"{model_name}.pth"
    weight_path = os.path.join(model_folder, weight_filename)

    # Check if the weight file exists
    if not os.path.exists(weight_path):
        if model_name in HOSTED_MODELS:
            print(f"Downloading pre-trained weights for {model_name}...")
            download_file(
                HOSTED_MODELS[model_name],
                weight_path
            )
            print(f"Download complete: {weight_path}")
            return weight_path
        else:
            raise ValueError(
                f"No pre-trained weights available for {model_name}")
    else:
        print(f"Using existing weights file: {weight_path}")
        return weight_path
