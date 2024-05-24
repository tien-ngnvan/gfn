import cv2
import numpy as np
import onnxruntime as ort

from models import BaseInference

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


  
class HeadFace(BaseInference):
    def __init__(self, model_path):
        super(HeadFace, self).__init__()
        self.load_model(model_path)
        
    def load_model(self, model_path:str):
        self.model = ort.InferenceSession(
            model_path,
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.inp_name = self.model.get_inputs()[0].name
        self.opt1_name = self.model.get_outputs()[0].name
        self.opt2_name = self.model.get_outputs()[1].name
        _, _, h, w = self.model.get_inputs()[0].shape
        self.model_inpsize = (w, h)

    def inference(self, img: np.array, test_size=(640, 640), det_thres=0.6, get_layer='face') -> list:
        """ Execute the main process

        Args:
            img (np.array): _description_
            test_size (tuple, optional): _description_. Defaults to (640, 640).
            det_thres (float, optional): _description_. Defaults to 0.6.
            get_layer (_type_, optional): _description_. Defaults to None.

        Returns:
            bbox: xyxy object
            score: bbox score of object detection
            label: both face = head = 0 (the same class name)
            kpts: if get_layer == 'face' return keypoints else None
        """
        # preprocess input
        tensor_img, ratio, dwdh = self.preprocess(img, test_size)
        
        # model prediction
        outputs = self.model.run([self.opt1_name, self.opt2_name], {self.inp_name: tensor_img})

        assert len(outputs) == 2, f'This class support head and face detection'
        
        pred = outputs[1] if get_layer == 'face' else outputs[0]
        
        # postprocess output
        bboxes, scores, labels, kpts = self.postprocess(pred, ratio, dwdh, det_thres, get_layer)
        
        return bboxes, scores, labels, kpts
        
    def preprocess(self, im:np.array, new_shape=(640, 640), color=(114, 114, 114), scaleup=True) -> list:
        """ Preprocessing function with reshape and normalize input

        Args:
            im (np.array, optional): input image
            new_shape (tuple, optional): new shape to resize. Defaults to (640, 640).
            color (tuple, optional): _description_. Defaults to (114, 114, 114).
            scaleup (bool, optional): resize small to large input size. Defaults to True.

        Returns:
            im: image after normalize and resize
            r: scale ratio between original and new shape 
            dw, dh: padding follow by yolo processing
        """
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

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=color
                                )  # add border
        
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, 0)
        im = np.ascontiguousarray(im, dtype=np.float32)
        im /= 255
        
        return im, r, (dw, dh)
    
    def postprocess(self, pred, ratio, dwdh, det_thres = 0.7, get_layer=None):
        """Processing output model match output format

        Args:
            pred (array): predict output model 
                base_opt: batch_index, xmin, ymin, xmax, ymax, bbox_label, bbox_score
                - pred w/o keypoint
                    pred[batch, 7]: base_opt
                - pred with keypoint
                    pred[batch, 7 + kpts]: base_opt,
                                            x_keypoint1, y_keypoint1, keypoint1_score,
                                            x_keypoint2, y_keypoint2, keypoint2_score,
                                            ...
                                            x_keypoint, y_keypoint2, keypoint2_score,
            ratio (float, optional): 
            dwdh (float, optional): 
            det_thres (float, optional): _description_. Defaults to 0.7.
            get_layer (str, optional): get detection output layer if the ouput has:
                                        3 items [head, face, body]
                                        2 items [face, head].

        Returns:
            [bbox, score, class_name, keypoints]
        """
        assert get_layer != None, f'get_layer is not None'
        
        if isinstance(pred, list):
            pred = np.array(pred)
            
        pred = pred[pred[:, 6] > det_thres] # get sample higher than threshold
        
        padding = dwdh*2
        det_bboxes, det_scores, det_labels  = pred[:,1:5], pred[:,6], pred[:, 5]
        kpts = pred[:, 7:] if pred.shape[1] > 6 else None
        det_bboxes = (det_bboxes[:, 0::] - np.array(padding)) / ratio
        
        if kpts is not None:
            kpts[:,0::3] = (kpts[:,0::3] - np.array(padding[0])) / ratio
            kpts[:,1::3] = (kpts[:,1::3]- np.array(padding[1])) / ratio

        return det_bboxes, det_scores, det_labels, kpts