import numpy as np
import onnxruntime as ort

from models import BaseInference

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


class MetricNet(BaseInference):
    def __init__(self, model_path):
        self.load_model(model_path)

    def load_model(self, model_path: str):
        self.model = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.inp1_name = self.model.get_inputs()[0].name
        self.inp2_name = self.model.get_inputs()[1].name
        self.opt_name = self.model.get_outputs()[0].name

    def inference(self, embd1, embd2):
        scores = self.model.run(
            [self.opt_name],
            {
                self.inp1_name: embd1.astype("float32"),
                self.inp2_name: embd2.astype("float32")
            }
        )[0]

        scores = np.squeeze(scores, axis=0)

        return scores