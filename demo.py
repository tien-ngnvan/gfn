import os
import cv2
import glob
import pickle
import argparse
import numpy as np
from pathlib import Path

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
        default="weights/yolov7-hf-v1.onnx",
        help="Path to the human detection model",
    )
    parser.add_argument(
        "--det_thresh",
        type=float,
        default=0.75,
        help="Threshold for human detection",
    )
    parser.add_argument(
        "--feature_model_path",
        type=str,
        default="weights/ghostnetv1.onnx",
        help="Path to the human reid model",
    )
    parser.add_argument(
        "--metric_model_path",
        type=str,
        default="weights/metricnet.onnx",
        help="Path to the human reid model",
    )
    parser.add_argument(
        "--biometric_thresh",
        type=float,
        default=0.5,
        help="Threshold for recognition",
    )
    parser.add_argument(
        "--spoofing_model_path",
        type=str,
        default="weights/OCI2M_spoofing.onnx",
        help="Path to the human live or spoofing",
    )
    parser.add_argument(
        "--spoofing_thresh",
        type=float,
        default=0.5,
        help="Threshold for live or spoofing",
    )
    parser.add_argument(
        "--tracker_type",
        type=str,
        default="botsort",
        help="Tracker type",
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
            image = cv2.imread(x)
            embd = processor(image, mode="image")
            db_embed.append(embd[0])
            register_name.append(name.split('.')[0])
  
        with open(os.path.join(datapath), 'wb') as handle:
            pickle.dump(db_embed, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(os.path.join(datapath).replace('pickle', 'txt'), 'w') as f:
            for x in register_name:
                f.writelines(f'{x}\n')

    db_embed = np.array(db_embed)
        
if __name__ == "__main__":
    args = parse_args()
    # Initialize processor and logger
    sys_logger = setup_logger(args.system_log_path, format=LoggerFormat.SYSTEM)

    processor = Processor(**vars(args))
    
    init_database(args, processor)
   
    # open camera
    videofile = 'nvt.mp4'
    if videofile is not None:
        cap = cv2.VideoCapture(videofile)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if args.target_fps == 0 or args.target_fps > fps:
            target_fps = fps
        else:
            target_fps = args.target_fps
        
        file_name = os.path.splitext(videofile)[0]
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

            boxes, bio_score, kpts, embed = processor(frame, db_embed, mode="image")
        
            if len(boxes) != 0:
                pred_score = sum(bio_score) / len(bio_score)        
                if pred_score > args.biometric_thresh:
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
    else:
        cap = cv2.VideoCapture(0)
        
    