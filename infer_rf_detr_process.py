import copy

import torch
from PIL import Image

from ikomia import core, dataprocess, utils

from infer_rf_detr.utils import get_class_names, load_model


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferRfDetrParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "rf-detr-medium"
        self.cuda = torch.cuda.is_available()
        self.input_size = 576
        self.conf_thres = 0.50
        self.update = False
        self.model_weight_file = ""
        self.config_file = ""
        self.config_file = ""

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.conf_thres = float(param_map["conf_thres"])
        self.model_weight_file = str(param_map["model_weight_file"])
        self.config_file = str(param_map["config_file"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "model_name": str(self.model_name),
            "cuda": str(self.cuda),
            "input_size": str(self.input_size),
            "conf_thres": str(self.conf_thres),
            "update": str(self.update),
            "model_weight_file": str(self.model_weight_file),
            "config_file": str(self.config_file)
        }
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
        self.model = None
        self.class_ids = None

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def _load_model(self):
        param = self.get_param_object()

        # Set class names
        class_list, self.class_ids = get_class_names(param)
        self.set_names(class_list)
        nb_classes = len(class_list)

        # Load model
        self.model = load_model(param, nb_classes)
        print(f"Model {param.model_name} loaded successfully")

        param.update = False

    def init_long_process(self):
        self._load_model()
        super().init_long_process()

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()
        # Get input :
        img_input = self.get_input(0)
        # Get image from input/output (numpy array):
        src_image = img_input.get_image()

        # Convert numpy array to PIL image
        image = Image.fromarray(src_image)

        # Load model
        if param.update:
            self._load_model()

        # Inference
        try:
            # Get detections
            detections = self.model.predict(image, threshold=param.conf_thres)
            boxes = detections.xyxy
            confidences = detections.confidence
            class_idx = detections.class_id

            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, class_idx)):
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                width = x2 - x1
                height = y2 - y1

                if self.class_ids is not None:
                    class_index = self.class_ids.index(cls)
                else:
                    class_index = cls

                self.add_object(
                    i,
                    int(class_index),
                    float(conf),
                    float(x1),
                    float(y1),
                    float(width),
                    float(height)
                )
        except Exception as e:
            print(f"Error during inference: {str(e)}")

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


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
        self.info.version = "1.1.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Robinson, Isaac and Robicheaux, Peter and Popov, Matvei"
        self.info.article = ""
        self.info.journal = ""
        self.info.year = 2025
        self.info.license = "Apache-2.0"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.15.0"
        # self.info.max_ikomia_version = "0.11.1"

        # Python compatibility
        self.info.min_python_version = "3.9.0"
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

        # Min hardware config
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        # Create algorithm object
        return InferRfDetr(self.info.name, param)
