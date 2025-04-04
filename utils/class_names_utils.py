import yaml
from infer_rf_detr.rf_detr.rfdetr.util.coco_classes import COCO_CLASSES_UPDATE


def get_class_names(param):
    if param.model_weight_file:
        if not param.class_file:
            raise ValueError(
                "The config_file 'class_names.yaml' is required when using a custom model file.")
        else:
            with open(param.class_file, 'r') as f:
                config = yaml.safe_load(f)
                classes_list = list(config.get('classes', []))
    else:
        classes_list = list(COCO_CLASSES_UPDATE.values())

    return classes_list
