from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_rf_detr.infer_rf_detr_process import InferRfDetrParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
from torch.cuda import is_available


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferRfDetrWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferRfDetrParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Cuda
        self.check_cuda = pyqtutils.append_check(
            self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        # Model name
        self.combo_model = pyqtutils.append_combo(
            self.grid_layout, "Model name")
        self.combo_model.addItem("rf-detr-base")
        self.combo_model.addItem("rf-detr-large")

        self.combo_model.setCurrentText(self.parameters.model_name)

        # Hyper-parameters
        custom_weight = bool(self.parameters.model_weight_file)
        self.check_cfg = QCheckBox("Custom model")
        self.check_cfg.setChecked(custom_weight)
        self.grid_layout.addWidget(
            self.check_cfg, self.grid_layout.rowCount(), 0, 1, 2)
        self.check_cfg.stateChanged.connect(self.on_custom_weight_changed)

        self.label_hyp = QLabel("Model weight (.pt)")
        self.browse_weight_file = pyqtutils.BrowseFileWidget(
            path=self.parameters.model_weight_file,
            tooltip="Select file",
            mode=QFileDialog.ExistingFile
        )
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_hyp, row, 0)
        self.grid_layout.addWidget(self.browse_weight_file, row, 1)

        # Class file selection
        self.label_class_file = QLabel("Class file (.yaml)")
        self.browse_class_file = pyqtutils.BrowseFileWidget(
            path=self.parameters.class_file,
            tooltip="Select class file",
            mode=QFileDialog.ExistingFile
        )
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_class_file, row, 0)
        self.grid_layout.addWidget(self.browse_class_file, row, 1)

        # Ensure visibility based on custom weight toggle
        self.label_hyp.setVisible(custom_weight)
        self.browse_weight_file.setVisible(custom_weight)

        self.label_class_file.setVisible(custom_weight)
        self.browse_class_file.setVisible(custom_weight)

        # Input size
        self.spin_input_size = pyqtutils.append_spin(
            self.grid_layout,
            "Input size",
            self.parameters.input_size
        )

        # Confidence threshold
        self.spin_conf_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "Confidence threshold",
            self.parameters.conf_thres,
            min=0.,
            max=1.,
            step=0.01,
            decimals=2
        )

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_custom_weight_changed(self, int):
        self.label_hyp.setVisible(self.check_cfg.isChecked())
        self.browse_weight_file.setVisible(self.check_cfg.isChecked())
        self.label_class_file.setVisible(self.check_cfg.isChecked())
        self.browse_class_file.setVisible(self.check_cfg.isChecked())

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.input_size = self.spin_input_size.value()
        self.parameters.conf_thres = self.spin_conf_thres.value()
        if self.check_cfg.isChecked():
            self.parameters.model_weight_file = self.browse_weight_file.path
            self.parameters.class_file = self.browse_class_file.path
        self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferRfDetrWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_rf_detr"

    def create(self, param):
        # Create widget object
        return InferRfDetrWidget(param, None)
