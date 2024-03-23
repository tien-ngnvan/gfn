import os
import io
import cv2
import hashlib
import pathlib
import asyncio
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize 
from fastapi import *
from fastapi.staticfiles import StaticFiles
from gfn.models import HeadFace, SpoofingNet, GhostFaceNet
from gfn.utils import image as imgutils


"""
Load models and thresh.
"""
# For face storage.
FACES = list()
FACES_IMG_DIR = os.getenv('IMG_DIR', default='face_images')
FACES_IMG_DIR = pathlib.Path(FACES_IMG_DIR)
FACES_IMG_DIR.mkdir(exist_ok=True)
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

def recognize_face(emb):
  result = compare_cosine(emb, FACES)
  rec_idx = result.argmax(-1)
  return result[rec_idx] > RECOGNITION_THRESH

def build_emb():
  for images in os.listdir(str(FACES_IMG_DIR)):
    # check if the image ends with png
    if (images.endswith(".jpg")):
      name, ext = os.path.splitext(images)
      path = FACES_IMG_DIR.joinpath(images)
      image = Image.open(path)
      box, emb = detect_face(np.array(image))
      if emb is not None:
        FACES.append(emb)
      else:
        os.remove(path)
build_emb()

"""
FastAPI
"""
app = FastAPI()
app.mount(f'/imgs', StaticFiles(directory=str(FACES_IMG_DIR)), name='images')

"""
For face register APIs
"""
@app.post('/register')
async def post_face_image(request: Request, file: UploadFile = File(...)):
  request_object_content = await file.read()
  try:
    img = Image.open(io.BytesIO(request_object_content))
    box, emb = detect_face(np.array(img))
  except:
    raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type") 
  if emb is not None:
    hash = hashlib.md5(img.tobytes()).hexdigest()
    FACES.append(emb)
    img.save(str(FACES_IMG_DIR.joinpath(f'{hash}.jpg')))
    return {'image': f'/imgs/{hash}.jpg'}
  elif box is None:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='No face founded')
  else:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Maybe fake image')
  
@app.get('/faces')
async def get_fae_image():
  res = [f'/imgs/{k}' for k in os.listdir(str(FACES_IMG_DIR)) if k.endswith(".jpg")]
  return res

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
  bytes = await websocket.receive_text()
  try:
    queue.put_nowait(bytes)
  except asyncio.QueueFull:
    pass

async def kyc(websocket: WebSocket, queue: asyncio.Queue):
  while True:
    b64 = await queue.get()
    img = imgutils.from_base64(b64)
    box, emb = detect_face(np.array(img))
    
    res = {}
    if box is not None:
      pass
    # await websocket.send_bytes(data)


