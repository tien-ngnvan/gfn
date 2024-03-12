import datetime
import json
import os
import cv2
import argparse
from glob import glob
import json

import numpy as np
from modules.processor.processor import Processor
from modules.logging.logger import setup_logger, log_file_size, LoggerFormat


IMG_TYPES = [".jpg", ".jpeg", ".png"]
VIDEO_TYPES = [".mp4", ".avi", ".mkv"]
INFOR_RUN_FILE = "infor_run.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Human Tracking")
    parser.add_argument(
        "--input_folder",
        # default="sample",
        type=str,
        required=True,
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
        "--host",
        type=str,
        # default="localhost",
        help="Host for qdrant database",
    )
    parser.add_argument(
        "--port",
        type=int,
        # default=6333,
        help="Port for qdrant database",
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    # Initialize processor and logger
    sys_logger = setup_logger(args.system_log_path, format=LoggerFormat.SYSTEM)
    # try:
    processor = Processor(**vars(args))

    # Make output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # Run inference
    for file in glob(os.path.join(args.input_folder, "*")):
        file_name = os.path.basename(file)
        file_type = os.path.splitext(file_name)[-1].lower()
        if file_type not in IMG_TYPES + VIDEO_TYPES:
            sys_logger.warning(
                f"Unsupported file type: {file_type} for file: {file_name}"
            )
            continue

        file_name = os.path.splitext(file_name)[0]

        # Save infor run (file_name, model, etc for logging benchmark)
        infor_run = {
            "file_name": file_name,
            "det_model":  args.det_model_path.split("/")[-1],              
            "reid_model": args.reid_model_path.split("/")[-1],
        }
        with open(INFOR_RUN_FILE, 'w') as json_file:
            json.dump(infor_run, json_file)
        
        meta = {
            "day": file_name.split("_")[0],
            "time": file_name.split("_")[1],
        }
        # Check image file type
        if file_type in IMG_TYPES:
            sys_logger.info(f"Processing image: {file_name}")
            image = cv2.imread(file)
            output = processor(image, meta, mode="image")
            saved_path = os.path.join(args.output_folder, file.split('/')[-1])
            
            cv2.imwrite(os.path.join(args.output_folder, f'{file_name}.jpg'), output)

        # Check video file type
        elif file_type in VIDEO_TYPES:
            sys_logger.info(f"Processing video: {file_name}")
            # Read video and get information
            cap = cv2.VideoCapture(file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            if args.target_fps == 0 or args.target_fps > fps:
                target_fps = fps
            else:
                target_fps = args.target_fps
            out = cv2.VideoWriter(
                os.path.join(args.output_folder, file_name) + ".mp4",
                fourcc,
                target_fps,
                (width, height),
            )
            # 20231116_120951_1857_axis-a4-south-delivery
            init_time = datetime.datetime.strptime(
                file_name.split("_")[0] + "_" + file_name.split("_")[1],
                "%Y%m%d_%H%M%S",
            )
            frame_idx = 0
            skip_interval = int(fps / target_fps)

            while True:
                # TODO: create
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % skip_interval == 0:
                    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    frame_time = init_time + datetime.timedelta(
                        seconds=(frame_number / target_fps)
                    )
                    meta.update({"time": frame_time.strftime("%H%M%S")})
                    output = processor(
                        frame, meta, mode="video", frame_idx=frame_idx
                    )
                    out.write(output)
                frame_idx += 1
            cap.release()
            out.release()

        # Log image size in mb
        log_file_size(
            input_path=file,
            output_path=os.path.join(args.output_folder, file_name) + ".mp4",
            logger=sys_logger,
        )
    # except Exception as e:
    #     sys_logger.exception(e)