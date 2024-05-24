import cv2
import numpy as np
import onnxruntime as ort

from models import BaseInference
from utils import face, image


sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL



class SpoofingNet(BaseInference):
    def __init__(self, model_path) -> None:
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
        self.inp_name = self.model.get_inputs()[0].name
        self.opt_name = self.model.get_outputs()[0].name
        _, _, h, w = self.model.get_inputs()[0].shape
        self.model_inpsize = (w, h)
    
    def inference(self, img: np.array) -> np.array:
        if isinstance(img, list):
            img = np.array(img)
            
        result = self.model.run(
            [self.opt_name], {self.inp_name: img.astype("float32")}
        )[0]
        
        return result
    
    def preprocess(self, img, xyxys, kpts):
        crops = []
        # dets are of different sizes so batch preprocessing is not possible
        for box, kpt in zip(xyxys, kpts):
            x1, y1, _, _ = box
            crop = image.crops(img, box)
            # Align face
            # Scale the keypoints to the face size
            kpt[::3] = kpt[::3] - x1
            kpt[1::3] = kpt[1::3] - y1
    
            crop = face.align_5_points(crop, kpt)
            crop = cv2.resize(crop, self.model_inpsize)
            crop = (crop - 127.5) * 0.0078125
            crop = crop.transpose(2, 0, 1)
            crop = np.expand_dims(crop, axis=0)
            crops.append(crop)
        crops = np.concatenate(crops, axis=0)
        
        return crops