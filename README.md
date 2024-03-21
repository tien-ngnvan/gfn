# gfn

## Installed package
```
%cd gfn
pip install -r requirements.txt
```
## Checkpoint
Download checkpoint [here](https://drive.google.com/drive/folders/1nD-NCMlCLuRyFl9VBq1umKk-hMwEJsgU?usp=sharing)
```
|__weights
|  |__ghostnet.onnx
|  |__yolov7-hf.onnx
```

## 
Create database folder and share your images. The folder should be contain at less 9 images (up, down, left, right, top left, top right, bottom left, bottom right and center)
```
|__modules
|__samples
|__database
|  |__ vantien
|     |__ vantien_left.jpg
|     |__ vantien_right.jpg
|     |__ ...
|  |__ trongthuy
|     |__ trongthuy_up.jpg
|     |__ trongthuy_down.jpg
|  |__ ...
|__demo.py
```

## DEMO
```sh
python -m demo \
    --det_model_path weights/yolov7-hf-v1.onnx \
    --reid_model_path weights/ghostnetv1.onnx \
    --reid_thresh 0.4 \
    --spoofing_model_path weights/OCI2M.onnx \
    --spoofing_thresh 0.5
```