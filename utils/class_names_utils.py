from rfdetr.util.coco_classes import COCO_CLASSES
import yaml


# COCO_80_CLASSES_AS_LIST should be the standard list of 80 COCO classes
def get_class_names(param):
    if param.model_weight_file:
        if not param.class_file:
            raise ValueError(
                "The config_file 'class_names.yaml' is required when using a custom model file.")
        else:
            # load class names from file .yaml
            with open(param.class_file, 'r') as f:
                config = yaml.safe_load(f)
                self.classes = config.get('classes', [])
    else:
        # Load COCO default class names
        classes_list = list(COCO_CLASSES.values())

    return classes_list
