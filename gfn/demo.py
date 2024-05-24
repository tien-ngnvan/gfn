import os
import cv2
import glob
import pickle
import argparse
import numpy as np
from pathlib import Path

from models import HeadFace, GhostFaceNet, SpoofingNet, MetricNet
from utils import image

IMG_TYPES = [".jpg", ".jpeg", ".png"]
VIDEO_TYPES = [".mp4", ".avi", ".mkv"]
INFOR_RUN_FILE = "infor_run.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Human Tracking")
    parser.add_argument("--inp_folder", type=str, default='/database',
                        help="Path to the input image/video file. Support image (jpg, png, jpeg) "\
                            "and video (mp4, avi, mkv)")
    
    parser.add_argument("--opt_folder", type=str, default="output",
                        help="Path to the output video file",)
    
    parser.add_argument("--database_path", type=str, default='database')
    
    parser.add_argument("--det_model_path", type=str, default="weights/yolov7-hf-v0.onnx",
                        help="Path to the human detection model")
    
    parser.add_argument("--det_thresh", type=float, default=0.75,
                        help="Threshold for human detection")
    
    parser.add_argument("--feature_model_path", type=str, default="weights/ghostnetv1.onnx",
                        help="Path to the human reid model")
    
    parser.add_argument("--face_thresh", type=float, default=0.4,
                        help="Threshold for human detection")
    
    parser.add_argument("--metric_model_path", type=str, default="weights/metricnet.onnx",
                        help="Path to the human reid model")
    
    parser.add_argument("--spoofing_model_path", type=str, default="weights/OCI2M-spoofing.onnx",
                        help="Path to the human live or spoofing")
    
    parser.add_argument("--spoofing_thresh", type=float, default=0.5,
                        help="Threshold for live or spoofing")
    
    parser.add_argument("--mode", type=str,
                        help="Mode running pipeline support Live, Image, Video")
    
    return parser.parse_args()


def init_database(args, processor):
    global db_embed
    global register_name
    
    # Setup local database
    os.makedirs(args.database_path, exist_ok=True)
    datapath = os.path.join(args.database_path, 'db.pickle')

    if os.path.isfile(os.path.join(datapath)):
        with open(datapath, 'rb') as handle:
            db_embed = pickle.load(handle)
        with open(os.path.join(datapath).replace('pickle', 'txt'), 'r') as f:
            register_name = f.readlines()
        register_name = [x.strip() for x in register_name]
    else:
        img_formats = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
        p = str(Path(args.database_path).absolute())  # os-agnostic absolute path
        
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isfile(p):
            files = [p]  # files
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "**/*")))  # dir
        else:
            raise Exception(f"ERROR: {p} does not exist")

        img_file_list = [x for x in files if x.split(".")[-1].lower() in img_formats]
        
        # read samples folder database
        db_embed, register_name = [], []
        
        for x in img_file_list:
            name = x.split("/")[-1]
            img = cv2.imread(x)
            embd = processor(img, embedding=True)
            db_embed.append(embd[0])
            register_name.append(name.split('.')[0])
  
        with open(os.path.join(datapath), 'wb') as handle:
            pickle.dump(db_embed, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(os.path.join(datapath).replace('pickle', 'txt'), 'w') as f:
            for x in register_name:
                f.writelines(f'{x}\n')

    db_embed = np.array(db_embed)
    
    
class Processor:
    def __init__(
        self,
        det_model_path:str = None,
        det_thresh:float = 0.75,
        feature_model_path:str = None,
        face_reg_thresh:float = 0.4,
        spoofing_model_path:str = None,
        spoofing_thresh:float = 0.75,
        metric_model_path:str = None,
        metric_thresh:float = 0.5
        ):
        # detection model
        self.det_model = None if det_model_path is None else HeadFace(det_model_path)
        self.det_thresh = det_thresh
        
        # Feature model
        self.feature_model = None if feature_model_path is None else GhostFaceNet(feature_model_path)
        self.face_reg_thresh = face_reg_thresh
        
        # Spoofing model 
        self.spoofing_model = None if spoofing_model_path is None else SpoofingNet(spoofing_model_path)
        self.spoofing_thresh = spoofing_thresh
        
        # metric_model_path
        self.metric_model = None if metric_model_path is None else MetricNet(metric_model_path)
        self.metric_thresh = metric_thresh
        
    def __call__(self, img, embedding=False):
        # Face detection
        bbox, scores, _, kpts = self.det_model.inference(img, det_thres=self.det_thresh)

        # Face spoofing
        if self.spoofing_thresh is None:
            spoof_result = None
        else:
            # crop image
            spoof_crop = self.spoofing_model.preprocess(img, bbox, kpts)
            spoof_result = self.spoofing_model.inference(spoof_crop)        
        
        # Face recognition
        if self.feature_model is None:
            gfn_result = None
        else:
            gfn_crop = self.feature_model.preprocess(img, bbox, kpts)
            gfn_result = self.feature_model.inference(gfn_crop)
            
        if embedding:
            return gfn_result
       
        return bbox, scores, kpts, spoof_result, gfn_result

if __name__ == "__main__":
    
    args = parse_args()
    processor = Processor(
        det_model_path = args.det_model_path,
        det_thresh = args.det_thresh,
        feature_model_path = args.feature_model_path,
        face_reg_thresh = args.face_thresh,
        spoofing_model_path = args.spoofing_model_path,
        spoofing_thresh = args.spoofing_thresh,
        metric_model_path = args.metric_model_path,
        metric_thresh = args.spoofing_thresh
    )
    
    # initialize database
    init_database(args, processor)
    
    
    # check folder 
    if args.mode == 'video':
        # check support file
        assert str(args.inp_folder)[:-4] in VIDEO_TYPES, f'Not support except {VIDEO_TYPES} type'
        
        cap = cv2.VideoCapture(args.inp_folder)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if args.target_fps == 0 or args.target_fps > fps:
            target_fps = fps
        else:
            target_fps = args.target_fps
        
        file_name = os.path.splitext(args.inp_folder)[0]
        out = cv2.VideoWriter(
            os.path.join('runs', file_name) + ".mp4",
            fourcc,
            target_fps,
            (width, height),
        )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes, scores, kpts, spoof_result, gfn_embed = processor(frame)
        
            if len(boxes) != 0:
                pred, idx = image.cosine(gfn_embed, db_embed)  
                pred_score = max(pred)
                      
                if pred_score > args.face_reg_thresh:
                    check = True
                    name = 'NV.Tien'
                else:
                    check=False
                    name="Unknow"
                
                steps = 3
                for xyxy, kpt in zip(boxes, kpts):
                    color = (0, 255, 0) if check else (0, 0, 255)
                    x1, y1, x2, y2 = xyxy.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=2)
                    cv2.putText(
                        frame, f"{pred_score}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2
                    )
            else:
                cv2.putText(frame,"Error Detection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)    
            
            out.write(frame)
            
        cap.release()
        out.release()   
    elif args.mode == 'image':
        # Check image file type
        if args.inp_folder[:-4] in IMG_TYPES:
            print(f"Processing image: {args.inp_folder}")
            img = cv2.imread(args.inp_folder)
            bbox, scores, kpts, spoof_result, gfn_result = processor(img)
        else:
            print("Not thing to process . . .")
    else: # live
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            bbox, scores, kpts, spoof_result, gfn_result = processor(frame)

            if len(bbox) == 1:
                prob, idx = image.cosine(gfn_result, db_embed)
                name = register_name[idx] if prob > 0.1 else 'Unknow'

                steps = 3
                for xyxy, conf, kpt in zip(bbox, scores, kpts):
                    x1, y1, x2, y2 = xyxy.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    cv2.putText(
                        frame, f"{name}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2
                    )
            else:
                cv2.putText(frame,"Error Detection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.imshow("Detection", frame)

        cap.release()
        
            
        cap = cv2.VideoCapture(0)
        
    