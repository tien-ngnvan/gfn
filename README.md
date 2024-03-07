# gfn

## Installed package

```
pip install -q onnxruntime opencv-python
```


## DEMO
```sh
python -m demo \
    --input_folder samples \
    --output_folder output \
    --det_model_path weights/yolov7-hf-v1.onnx \
    --reid_model_path weights/ghostnetv1.onnx 
```