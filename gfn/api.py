import os
import io
import cv2
import json
import asyncio
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize 
from fastapi import *
from gfn.models import HeadFace, SpoofingNet, GhostFaceNet
from gfn.utils import image as imgutils


"""
Load models and thresh.
"""
# For face storage.
FACES = list()
# For detection.
DETECTION = HeadFace(os.getenv('DETECTION_MODEL_PATH', default='weights/yolov7-hf-v1.onnx'))
DETECTION_THRESH = os.getenv('DETECTION_THRESH', default=0.5)
# For anti-spoofing.
SPOOFING = SpoofingNet(os.getenv('SPOOFING_MODEL_PATH', default='weights/OCI2M.onnx'))
SPOOFING_THRESH = os.getenv('SPOOFING_THRESH', default=0.68)
# For recognition.
RECOGNITION = GhostFaceNet(os.getenv('RECOGNITION_MODEL_PATH', default='weights/ghostnetv1.onnx'))
RECOGNITION_THRESH = os.getenv('RECOGNITION_THRESH', default=0.4)

"""
Major recognition functions.
"""
def softmax(x):
  return np.exp(x)/np.sum(np.exp(x))

def compare_cosine(embed, anchor):
    return np.dot(normalize(embed), anchor.T)[0]

def detect_face(img):
  boxes, scores, _, kpts = DETECTION.detect(img, get_layer='face', conf_det=DETECTION_THRESH)
  if len(boxes) == 1:
    spoof = SPOOFING.get_features(img, boxes, kpts)
    spoof = softmax(spoof)[:,0]
    if spoof[0] > SPOOFING_THRESH:
      embs = RECOGNITION.get_features(img, boxes, kpts)
      return boxes[0], embs[0]
    else:
      return boxes[0], None
  return None, None

def recognize_face(img, emb):
  result = compare_cosine(emb, FACES)
  rec_idx = result.argmax(-1)
  return result[rec_idx] > RECOGNITION_THRESH


"""
FastAPI
"""
app = FastAPI()

"""
For face register APIs
"""
IMAGE_FILE_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/heic", "image/heif", "image/heics", "png", "jpeg", "jpg", "heic", "heif", "heics"]

@app.post('/face')
async def post_face_image(request: Request, file: UploadFile = File(...)):
  request_object_content = await file.read()
  try:
    img = Image.open(io.BytesIO(request_object_content))
    npimg = np.array(img)
    face = detect_face(npimg)
    if face[0] is not None and face[1] is not None:
      face_item = {
        'image': imgutils.to_base64(img),
        'box': face[0],
        'emb': face[1] 
      }
      FACES.append(face_item)
    return {'Ok'}
    pass
  except:
    raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type") 
  
@app.get('/face')
async def get_fae_image():
  return 

"""
For websocket streaming inference.
"""
@app.websocket('/')
async def socket(websocket: WebSocket):
  """
  Websocket endpoint that will be sending request to from the frontend.
  """
  await websocket.accept()
  queue: asyncio.Queue = asyncio.Queue(maxsize=10)
  task = asyncio.create_task(kyc(websocket, queue))
  try:
    while True:
      await socket_receive(websocket, queue)
  except WebSocketDisconnect:
    task.cancel()
    await websocket.close()

async def socket_receive(websocket: WebSocket, queue: asyncio.Queue):
  """
  Asynchronous function that will be used to receive websocket connections from frontend.
  """
  bytes = await websocket.receive_bytes()
  try:
    queue.put_nowait(bytes)
  except asyncio.QueueFull:
    pass

async def kyc(websocket: WebSocket, queue: asyncio.Queue):
  while True:
    bytes = await queue.get()
    await websocket.send_bytes(data)


