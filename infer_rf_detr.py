from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_rf_detr.infer_rf_detr_process import InferRfDetrFactory
        return InferRfDetrFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_rf_detr.infer_rf_detr_widget import InferRfDetrWidgetFactory
        return InferRfDetrWidgetFactory()
