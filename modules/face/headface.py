import numpy as np
import math
import cv2

  
class HeadFace:
    def __init__(self, model_path):
        self.load_model(model_path)
        
    def load_model(self, model_path):
        model_type = model_path.split(".")[-1]
        
        if model_type == "onnx":
            import onnxruntime as ort

            self.model = ort.InferenceSession(
                model_path,
                providers=[
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )
        elif model_type == 'pt':
            import torch

            self.model = torch.load(model_path)
        else:
            raise ValueError("Model type not supported")
        
    def preprocess(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im, dtype=np.float32)
        im /= 255
        
        return im, r, (dw, dh)
    
    def post_process(self, pred, ratio, dwdh, conf_thre = 0.7, conf_kpts=0.9, get_layer=None):
        """process function for cleaning output prediction

        Args:
            pred (array): predict output model 
                if pred = 'head'
                    pred[batch, 7]: _, xmin, ymin, xmax, ymax, bbox_label, bbox_score
                else:
                    pred[batch, 21]: _, xmin, ymin, xmax, ymax, bbox_label, bbox_score, 
                                    x_keypoint1, y_keypoint1, keypoint1_score,
                                    x_keypoint2, y_keypoint2, keypoint2_score,
                                    ...
                    
            conf_thre (float, optional): _description_. Defaults to 0.7.
            conf_kpts (float, optional): _description_. Defaults to 0.9.
            get_layer (str, optional): get detection head layer [head, face]. Defaults to 'face'.

        Returns:
            _type_: [bbox, score, class_name, keypoints]
        """
        
        if isinstance(pred, list):
            pred = np.array(pred)
            
        padding = dwdh*2
        det_bboxes, det_scores, det_labels  = pred[:, 1:5], pred[:, 6], pred[:, 5]
        kpts = None
        
        if get_layer == 'face':
            kpts = pred[:, 7:]
            
        det_bboxes = (det_bboxes[:, 0::] - np.array(padding)) /ratio
        if get_layer == 'face':
            kpts[:,0::3] = (kpts[:,0::3] - np.array(padding[0])) / ratio
            kpts[:,1::3] = (kpts[:,1::3]- np.array(padding[1])) / ratio

        return det_bboxes, det_scores, det_labels, kpts
    
    def detect(self, img, test_size=(640, 640), conf_det=0.6, nmsthre=0.45, get_layer=None):
        tensor_img, ratio, dwdh = self.preprocess(img, test_size, auto=False)
        
        # inference head, face
        outputs = self.model.run([], {self.model.get_inputs()[0].name: tensor_img})
        pred = outputs[1] if get_layer == 'face' else outputs[0]
        bboxes, scores, labels, kpts = self.post_process(pred, ratio, dwdh, get_layer=get_layer)
        
        return bboxes, scores, labels, kpts
    
    @staticmethod
    def face_align(img, kpts):
        # 5 points align
        # left eye, right eye, nose, left mouth, right mouth
        left_eye = kpts[0:2]
        right_eye = kpts[3:5]

        # draw the center of eyes
        cv2.circle(
            img, (int(left_eye[0]), int(left_eye[1])), 1, (0, 255, 0), thickness=-1
        )
        cv2.circle(
            img, (int(right_eye[0]), int(right_eye[1])), 1, (0, 0, 255), thickness=-1
        )
        # Calculate the vertical difference between eye keypoints
        dy = left_eye[1] - right_eye[1]
        # get the center of eyes
        eye_center = ((left_eye + right_eye) / 2).astype("int")
        cv2.circle(
            img, (int(eye_center[0]), int(eye_center[1])), 1, (255, 0, 0), thickness=-1
        )
        if dy < 0:
            # need to rotate counter-clockwise
            angle = math.degrees(math.atan(abs(dy) / (right_eye[0] - left_eye[0])))
        else:
            # need to rotate clockwise
            angle = -math.degrees(math.atan(abs(dy) / (right_eye[0] - left_eye[0])))

        # get the scale for eyes
        scale = 1
        # get the rotation matrix
        M = cv2.getRotationMatrix2D(left_eye.astype(float), angle, scale)
        # apply the affine transformation
        aligned_face = cv2.warpAffine(
            img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC
        )

        return aligned_face
        
        
if __name__ == '__main__':
    import os
    
    yolo_hf = HeadFace(
        model_path=os.path.join('weights', 'yolov7-hf-v1.onnx')
    )
    
    img = cv2.imread(os.path.join("samples", "picnic.jpg"))[:, :, ::-1]
    
    bboxes, bbox_scores, labels, kpts = yolo_hf.detect(img, get_layer='face')
    
    print("Bbox\n", bboxes, "\n")
    print("Keypoint: ", kpts, "\n")
    print("BBox score: ", bbox_scores, "\n")
    print("Labels: ", labels, "\n")
    