import os
import torch
import cv2
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from glob import glob
import glob

from modules.processor.processor import Processor
from modules.logging.logger import setup_logger, LoggerFormat


IMG_TYPES = [".jpg", ".jpeg", ".png"]
VIDEO_TYPES = [".mp4", ".avi", ".mkv"]
INFOR_RUN_FILE = "infor_run.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Human Tracking")
    parser.add_argument(
        "--input_folder",
        # default="sample",
        type=str,
        help="Path to the input image/video file. Support image (jpg, png, jpeg) and video (mp4, avi, mkv)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Path to the output video file",
    )
    parser.add_argument(
        "--det_model_path",
        type=str,
        default="weights/det/yolox_darknet.onnx",
        help="Path to the human detection model",
    )
    parser.add_argument(
        "--det_thresh",
        type=float,
        default=0.75,
        help="Threshold for human detection",
    )
    
    parser.add_argument(
        "--face_iou_thres",
        type=float,
        default=0.5,
        help="Threshold for head|face iou",
    )
    parser.add_argument(
        "--reid_model_path",
        type=str,
        default="weights/reid/REID_ghostnetv1.onnx",
        help="Path to the human reid model",
    )
    parser.add_argument(
        "--tracker_type",
        type=str,
        default="botsort",
        help="Tracker type",
    )
    parser.add_argument(
        "--reid_thresh",
        type=float,
        default=0.78,
        help="Threshold for reid matching",
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.22734550911325851,
        help="Threshold for matching tracks",
    )
    parser.add_argument(
        "--camera_log_path",
        type=str,
        default="logs.csv",
        help="Path to the camera log file",
    )
    parser.add_argument(
        "--system_log_path",
        type=str,
        default="system_logs.txt",
        help="Path to the system log file",
    )
    parser.add_argument(
        "--area_level",
        type=int,
        default=0,
        help="Level to filter the restrict area, 0: green, 1: green + yellow, 2: green + yellow + red",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=0,
        help="Target fps for the output video, 0 to keep the original fps",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="database",
        help="Host for qdrant database",
    )
    return parser.parse_args()

def compare_cosine(embed, anchor):
    b = torch.Tensor(embed)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    
    result = torch.mm(anchor, b_norm.transpose(0,1))
  
    return result 

def init_database(args, processor):
    global db_embed
    global register_name
    
    # Setup local database
    os.makedirs(args.db_path, exist_ok=True)
    datapath = os.path.join(args.db_path, 'db.pickle')

    if os.path.isfile(os.path.join(datapath)):
        with open(datapath, 'rb') as handle:
            db_embed = pickle.load(handle)
        with open(os.path.join(datapath).replace('pickle', 'txt'), 'r') as f:
            register_name = f.readlines()
        register_name = [x.strip() for x in register_name]
    else:
        img_formats = [ "bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo", ]
        p = str(Path(args.db_path).absolute())  # os-agnostic absolute path
        
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f"ERROR: {p} does not exist")

        img_file_list = [x for x in files if x.split(".")[-1].lower() in img_formats]
        
        # read samples folder database
        db_embed, register_name = [], []
        
        for x in img_file_list:
            name = x.split("\\")[-1]
            image = cv2.imread(x)
            _, _, _, _, embd = processor(image, meta, mode="image")
            db_embed.append(embd[0])
            register_name.append(name.split('.')[0])
  
        with open(os.path.join(datapath), 'wb') as handle:
            pickle.dump(db_embed, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(os.path.join(datapath).replace('pickle', 'txt'), 'w') as f:
            for x in register_name:
                f.writelines(f'{x}\n')
                
    db_embed = torch.stack([torch.Tensor(x) for x in db_embed])
    db_embed = torch.nn.functional.normalize(db_embed, p=2, dim=1)


if __name__ == "__main__":
    args = parse_args()
    # Initialize processor and logger
    sys_logger = setup_logger(args.system_log_path, format=LoggerFormat.SYSTEM)

    now = datetime.now()
    meta = {"day": now}
    processor = Processor(**vars(args))
    
    init_database(args, processor)
   
    # open camera
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        boxes, scores, _, kpts, embed = processor(frame, meta, mode="image")
    
        if len(boxes) == 1:            
            result = compare_cosine(embed, db_embed)
            prob, idx = torch.max(result, dim=0)
    
            name = register_name[idx] if prob > 0.1 else 'Unknow'
            
            steps = 3
            for xyxy, conf, kpt in zip(boxes, scores, kpts):
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
    cv2.destroyAllWindows()