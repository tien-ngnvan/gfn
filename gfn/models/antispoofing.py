import cv2
import numpy as np
import onnxruntime as ort

from . import HeadFace


sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


class SpoofingNet:
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
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        _, _, h, w = self.model.get_inputs()[0].shape
        self.model_input_size = (w, h)

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
            
            crop = cv2.resize(crop, self.model_input_size)
            crop = (crop - 127.5) * 0.0078125
            crop = crop.transpose(2, 0, 1)
            crop = np.expand_dims(crop, axis=0)
            crops.append(crop)
        crops = np.concatenate(crops, axis=0)
        return crops

    def forward(self, crops):
        result = self.model.run(
            [self.output_name], {self.input_name: crops.astype("float32")}
        )
        return result[0]

    def get_features(self, img, xyxys, kpts):
        if len(xyxys) == 0:
            return np.array([])
        
        crops = self.preprocess(img, xyxys, kpts)
        result = self.forward(crops)
        
        return result
    
    # def test(self, img):
    #     def softmax(x):
    #         s= np.sum(np.exp(x))
    #         return np.exp(x)/s

    #     result = self.forward(img)
    #     print("Predict: ", result)
        
    #     ttresult = torch.Tensor(result)
    #     probs = torch.softmax(ttresult, dim=1)[:,0]
    #     print("torch softmax: ", probs)
        
    #     mresult = softmax(result)[:,0]
    #     print("custom softmax: ", mresult)
        
    #     result = normalize(result)[:,0]
    #     print("sklearn norm: ", result)
        

if __name__ == '__main__':
    import cv2
    import torch
    from sklearn.preprocessing import normalize
    
    model = SpoofingNet('weights\\OCI2M.onnx')
    
    image = cv2.imread('database\\img_nvt.png')
    image = cv2.resize(image, (256,256))[:, :, ::-1]
    image = np.ascontiguousarray(image)
    image = image.transpose(2, 0, 1)
    image= np.expand_dims(image, axis=0)
    
    model.test(image)