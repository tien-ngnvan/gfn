import cv2
import torch
import numpy as np

from modules import HeadFace, GhostFaceNet, SpoofingNet
from sklearn.preprocessing import normalize


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
        reid_model_path,
        spoofing_model_path,
        det_thresh,
        spoof_thresh,
        fps=30,
        *args,
        **kwargs,
    ) -> None:        
        
        self.headface = HeadFace(det_model_path)
        self.ghostnet = GhostFaceNet(reid_model_path)
        self.spoofing = SpoofingNet(spoofing_model_path)
        self.det_thresh = det_thresh
        self.spoof_thresh = spoof_thresh
        self.reid_model_path = reid_model_path
        self.args = args
        self.kwargs = kwargs
        self.mapper = None

    def __call__(self, img, meta, mode="image", get_layer='face', frame_idx=0): 
        fimage = img.copy()
        
        # Face detection
        boxes, scores, _, kpts = self.headface.detect(img, get_layer=get_layer)

        # Track
        if mode == "video":
            # humans = self.track_det(img, humans)
            faces = self.track_face(img, boxes, scores, kpts)

        if len(boxes) != 0 and len(boxes) == 1:
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
                    
                # cv2.putText(
                #     img,
                #     f"{face_id} {conf:.2f}",
                #     (x1, y1 - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.6,
                #     color,
                #     2,
                # )
        
        # Check anti-face spoofing
        spoofing_result = self.spoofing.get_features(fimage, boxes, kpts) # [batch, 2]
        spoofing_result = self.softmax(spoofing_result)[:,0]
        
        # Face recognition
        if spoofing_result > self.spoof_thresh:
            ghostnet_result = self.ghostnet.get_features(fimage, boxes, kpts) # [faces, 512]
        else:
            ghostnet_result = []
            
        return boxes, scores, kpts, ghostnet_result
        
    
    def softmax(self, x):
        s= np.sum(np.exp(x))
        return np.exp(x)/s

    
    def track_face(self, img, bboxes, scores, kpts):
        if len(bboxes) == 0:
            tracks = self.tracker.update(
                np.zeros((0, 6)),
                img,
                kpts,
            )
            return np.zeros((0, 4)), np.zeros((0, 1)), np.zeros((0, 1))
        # xywh to xyxy
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]

        tracks = self.tracker.update(
            np.hstack(
                (bboxes, scores.reshape(-1, 1), np.zeros(len(bboxes)).reshape(-1, 1))
            ),
            img,
            kpts,
        )
        if len(tracks.xyxy) == 0:
            return np.zeros((0, 4)), np.zeros((0, 1))

        tracks.xyxy[:, 0] = np.clip(tracks.xyxy[:, 0], 0, img.shape[1])
        tracks.xyxy[:, 1] = np.clip(tracks.xyxy[:, 1], 0, img.shape[0])
        tracks.xyxy[:, 2] = np.clip(tracks.xyxy[:, 2], 0, img.shape[1])
        tracks.xyxy[:, 3] = np.clip(tracks.xyxy[:, 3], 0, img.shape[0])

        return tracks.xyxy, tracks.confidence, tracks.track_id

    def mosaic_blur(self, img, face, strength=2):
        if len(face) == 0:
            return
        # Apply mosaic blur to the face
        x1, y1, x2, y2 = face.astype(int)

        # Get the face from the image
        _face = img[y1:y2, x1:x2].copy()
        face_height, face_width = _face.shape[:2]

        # Apply a simple mosaic blur
        _face = cv2.resize(
            _face, None, fx=strength, fy=strength, interpolation=cv2.INTER_LINEAR
        )

        _face = cv2.resize(
            _face, (face_width, face_height), interpolation=cv2.INTER_NEAREST
        )

        # Create an oval mask
        mask = np.zeros((face_height, face_width), np.uint8)
        ellipse_radius_x = face_width // 2
        ellipse_radius_y = face_height // 2
        cv2.ellipse(
            mask,
            (ellipse_radius_x, ellipse_radius_y),
            (ellipse_radius_x, ellipse_radius_y),
            0,
            0,
            360,
            1,
            -1,
        )

        # Apply the mask to the mosaic blur
        _face = cv2.bitwise_and(_face, _face, mask=mask)

        # Paste the mosaic blur onto the original image without the black background
        original_face = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.bitwise_and(
            original_face,
            original_face,
            mask=cv2.bitwise_not(mask) // 255,
        )
        img[y1:y2, x1:x2] = cv2.bitwise_or(original_face, _face)