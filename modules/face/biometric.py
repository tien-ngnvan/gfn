import cv2
import numpy as np
import onnxruntime as ort

from modules import HeadFace

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


class BiometricNet:
    def __init__(self, feature_path, metric_path) -> None:
        self.load_featurenet(feature_path)
        self.load_metricnet(metric_path)

    def load_featurenet(self, model_path: str):
        self.featurenet = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.feat_inp_name = self.featurenet.get_inputs()[0].name
        self.feat_opt_name = self.featurenet.get_outputs()[0].name
        _, h, w, _ = self.featurenet.get_inputs()[0].shape
        self.feaure_input_size = (w, h)
        
    def load_metricnet(self, model_path: str):
        self.metricnet = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.input_name1 = self.metricnet.get_inputs()[0].name
        self.input_name2 = self.metricnet.get_inputs()[1].name
        self.output_name = self.metricnet.get_outputs()[0].name

    def preprocess(self, img, xyxys, kpts):
        crops = []
        h, w = img.shape[:2]
        # dets are of different sizes so batch preprocessing is not possible
        for box, kpt in zip(xyxys, kpts):
            x1, y1, x2, y2 = box
            # Limit the face to the image
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(w, x2))
            y2 = int(min(h, y2))

            box = [x1, y1, x2, y2]
            crop = img[y1:y2, x1:x2]
            # Align face
            # Scale the keypoints to the face size
            kpt[::3] = kpt[::3] - x1
            kpt[1::3] = kpt[1::3] - y1
            
            crop = HeadFace.face_align(crop, kpt)
            
            crop = cv2.resize(crop, self.feaure_input_size)
            crop = (crop - 127.5) * 0.0078125
            # crop = crop.transpose(2, 0, 1)
            crop = np.expand_dims(crop, axis=0)
            crops.append(crop)
        crops = np.concatenate(crops, axis=0)
        return crops

    def forward(self, crops):
        embeddings = self.featurenet.run(
            [self.feat_opt_name], {self.feat_inp_name: crops.astype("float32")}
        )[0]
        
        return embeddings

    def get_features(self, img, xyxys, kpts, norm=False):
        if len(xyxys) == 0:
            return np.array([])
        
        crops = self.preprocess(img, xyxys, kpts)
        embeddings = self.forward(crops)
        if norm:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
        return embeddings
    
    def cosim(self, embd1, embd2):
        if isinstance(embd1, list):
            embd1 = np.expand_dims(np.array(embd1), axis=0)

        if isinstance(embd2, list) or len(embd2.shape) == 1:
            embd2 = np.expand_dims(np.array(embd2), axis=0)
            
        scores = self.metricnet.run(
            [self.output_name], 
            {
                self.input_name1: embd1.astype("float32"),
                self.input_name2: embd2.astype("float32")
            }
        )
        
        pred = np.squeeze(scores, axis=0)
        
        scores = [((x+1)/2)[0].tolist() for x in pred]
        
        return scores if len(scores) > 1 else scores[0]