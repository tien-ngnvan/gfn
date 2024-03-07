# gfn

## Installed package
```
%cd gfn
pip install -q onnxruntime opencv-python
```
## Checkpoint
Download checkpoint [here](https://drive.google.com/drive/folders/1nD-NCMlCLuRyFl9VBq1umKk-hMwEJsgU?usp=sharing)
```
|__weights
|  |__ghostnet.onnx
|  |__yolov7-hf.onnx
```

## 
Create database folder and share your images
```
|__modules
|__samples
|__database
|  |__ vantien.jpg
|  |__ trongthuy.jpg
|  |__ ...
|__demo.py
```

## DEMO
```sh
python -m demo \
    --det_model_path weights/yolov7-hf-v1.onnx \
    --reid_model_path weights/ghostnetv1.onnx 
```