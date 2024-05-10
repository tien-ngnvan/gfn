import cv2
import numpy as np

from modules import HeadFace, BiometricNet 


palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color



class Processor:
    def __init__(
        self,
        det_model_path,
        feature_model_path,
        metric_model_path,
        det_thresh,
        *args,
        **kwargs,
    ) -> None:        
        
        self.headface = HeadFace(det_model_path)
        self.biometric = BiometricNet(feature_model_path, metric_model_path)
        self.det_thresh = det_thresh
        self.feature_model_path = feature_model_path
        self.args = args
        self.kwargs = kwargs
        self.mapper = None

    def __call__(self, img, db_embed=None, mode="image", get_layer='face'): 
        fimage = img.copy()
        
        # Face detection
        boxes, scores, _, kpts = self.headface.detect(img, get_layer=get_layer)
        
        if len(boxes) == 1:
            steps = 3
            for xyxy, _, kpt in zip(boxes, scores, kpts):
                x1, y1, x2, y2 = xyxy.astype(int)
                color = get_color(int(kpt)) if mode == "video" else (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)

                # draw keypoints
                num_step = len(kpt) // steps
                for kid in range(num_step):
                    r, g, b = pose_kpt_color[kid]
                    x_coord, y_coord = kpt[steps * kid], kpt[steps * kid + 1]
                    cv2.circle(img, (int(x_coord), int(y_coord)), 2, (int(r), int(g), int(b)), -1)
            
            # featurenet
            embedings = self.biometric.get_features(fimage, boxes, kpts, norm=False) # [faces, 512]
            if db_embed is None:
                return embedings
            
            if db_embed.shape[0] > 1:
                extend = np.tile(embedings, (db_embed.shape[0],1))
            else:
                extend = embedings
  
            bio_scores = self.biometric.cosim(extend, db_embed)
            
            return boxes, bio_scores, kpts, embedings
        return [], None, None, None
        
    def softmax(self, x):
        s= np.sum(np.exp(x))
        
        return np.exp(x)/s