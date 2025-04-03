import copy
from ikomia import core, dataprocess, utils
import torch
import yaml

from PIL import Image
from rfdetr.util.coco_classes import COCO_CLASSES
from infer_rf_detr.utils.model_utils import load_model


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------


class InferRfDetrParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "rf-detr-base"
        self.cuda = torch.cuda.is_available()
        self.input_size = 560
        self.conf_thres = 0.50
        self.update = False
        self.model_weight_file = ""
        self.class_file = ""

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.conf_thres = float(param_map["conf_thres"])
        self.model_weight_file = str(param_map["model_weight_file"])
        self.class_file = str(param_map["class_file"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["cuda"] = str(self.cuda)
        param_map["input_size"] = str(self.input_size)
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["update"] = str(self.update)
        param_map["model_weight_file"] = str(self.model_weight_file)
        param_map["class_file"] = str(self.class_file)
        return param_map


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferRfDetr(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        # Create parameters object
        if param is None:
            self.set_param_object(InferRfDetrParam())
        else:
            self.set_param_object(copy.deepcopy(param))
        self.device = torch.device("cpu")
        self.input_size = 560
        self.classes = None
        self.model = None
        self.model_name = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def adjust_to_multiple(self, value, base=14):
        """Adjust value down to the nearest multiple of 'base'."""
        return (value // base) * base

    def set_class_names(self, param):
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
            self.classes = list(COCO_CLASSES.values())

        self.set_names(self.classes)
        return len(self.classes)

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()
        # Core function of your process

        # Get parameters :
        param = self.get_param_object()

        # Get input :
        input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = input.get_image()

        # Convert numpy array to PIL image
        image = Image.fromarray(src_image)

        # Load model
        if param.update or self.model is None:
            # Set device as string instead of torch.device object
            device_str = "cuda" if param.cuda and torch.cuda.is_available() else "cpu"
            # Keep this for other torch operations
            self.device = torch.device(device_str)

            # Check input size is a multiple of 14
            new_input_size = self.adjust_to_multiple(param.input_size)
            if new_input_size != param.input_size:
                self.input_size = new_input_size
                print(
                    f"Updating input size to {self.input_size} to be a multiple of 14")
            else:
                self.input_size = param.input_size

            num_classes = self.set_class_names(param)
            model = load_model(param, num_classes)
            self.model_name = param.model_name

            print(f"Model {param.model_name} loaded successfully")

            param.update = False

        # Inference
        try:
            detections = model.predict(image, threshold=param.conf_thres)

            # Get detections
            boxes = detections.xyxy
            confidences = detections.confidence
            class_idx = detections.class_id

            idx_adjust = 0 if param.model_weight_file else 1
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, class_idx)):
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                width = x2 - x1
                height = y2 - y1
                self.add_object(
                    i,
                    int(cls-idx_adjust),
                    float(conf),
                    float(x1),
                    float(y1),
                    float(width),
                    float(height)
                )
        except Exception as e:
            self.print_error_message(f"Error during inference: {str(e)}")

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()

    def print_error_message(self, message):
        """Print error message in a consistent format."""
        print(f"ERROR: {message}")


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferRfDetrFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_rf_detr"
        self.info.short_description = "Inference with RF-DETR models"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Robinson, Isaac and Robicheaux, Peter and Popov, Matvei"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2025
        self.info.license = "Apache-2.0"

        # Ikomia API compatibility
        # self.info.min_ikomia_version = "0.11.1"
        # self.info.max_ikomia_version = "0.11.1"

        # Python compatibility
        self.info.min_python_version = "3.11.0"
        # self.info.max_python_version = "3.11.0"

        # URL of documentation
        self.info.documentation_link = "https://blog.roboflow.com/rf-detr/"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_rf_detr"
        self.info.original_repository = "https://github.com/roboflow/rf-detr"

        # Keywords used for search
        self.info.keywords = "DETR, object, detection, roboflow, real-time"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create algorithm object
        return InferRfDetr(self.info.name, param)
