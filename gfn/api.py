import os
import io
import json
import base64
import hashlib
import pathlib
import asyncio
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize 
from fastapi import *
from fastapi.staticfiles import StaticFiles
from gfn.models import HeadFace, SpoofingNet, GhostFaceNet
from gfn.db import FaceDatabaseInterface, QdrantFaceDatabase
from io import BytesIO


"""
Load models and thresh.
"""
# For face storage.
FACES: FaceDatabaseInterface = QdrantFaceDatabase()
FACES_IMG_DIR = pathlib.Path(os.getenv('IMG_DIR', default='face_images'))
FACES_IMG_DIR.mkdir(exist_ok=True)
# For detection.
DETECTION = HeadFace(os.getenv('DETECTION_MODEL_PATH', default='weights/yolov7-hf-v1.onnx'))
DETECTION_THRESH = os.getenv('DETECTION_THRESH', default=0.5)
# For anti-spoofing.
SPOOFING = SpoofingNet(os.getenv('SPOOFING_MODEL_PATH', default='weights/OCI2M.onnx'))
SPOOFING_THRESH = os.getenv('SPOOFING_THRESH', default=0.6)
# For recognition.
RECOGNITION = GhostFaceNet(os.getenv('RECOGNITION_MODEL_PATH', default='weights/ghostnetv1.onnx'))
RECOGNITION_THRESH = os.getenv('RECOGNITION_THRESH', default=0.3)

"""
Major recognition functions.
"""
def softmax(x):
  return np.exp(x)/np.sum(np.exp(x))

def compare_cosine(embed, anchor):
    norm = normalize([embed])
    t = anchor.T
    return np.dot(norm, t)[0]

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

def recognize_face(emb, person_id):
  # result = compare_cosine(emb, np.array(target))
  # rec_idx = result.argmax(-1)
  rec_idx = FACES.check_face(person_id, emb, RECOGNITION_THRESH)
  # return rec_idx > RECOGNITION_THRESH
  return rec_idx

def build_emb():
  for _, dirs, _ in FACES_IMG_DIR.walk():
    for dir in dirs:
      FACES[dir] = list()
      folder = FACES_IMG_DIR.joinpath(dir)
      for _, _, files in folder.walk():
        for file in files:
          if file.endswith('.jpg'):
            path = folder.joinpath(file)
            img = np.array(Image.open(path))
            box, emb = detect_face(img)
            if emb is not None:
              FACES[dir].append(emb)
            else:
              os.remove(path)
#build_emb()

"""
FastAPI
"""
app = FastAPI()
app.mount(f'/imgs', StaticFiles(directory=str(FACES_IMG_DIR)), name='images')

"""
For face register APIs
"""
@app.get('/configure')
async def configure():
  return json.dumps({
    'DETECTION_THRESH': DETECTION_THRESH,
    'SPOOFING_THRESH': SPOOFING_THRESH,
    'RECOGNITION_THRESH': RECOGNITION_THRESH
  })

@app.post('/configure')
async def configure(detection_thresh: float, spoofing_thresh: float, recognition_thresh: float):
  detect_thresh = 0 if detection_thresh < 0 else 1 if detection_thresh > 1 else detection_thresh
  SPOOFING_THRESH = 0 if spoofing_thresh < 0 else 1 if spoofing_thresh > 1 else spoofing_thresh
  RECOGNITION_THRESH = 0 if recognition_thresh < 0 else 1 if recognition_thresh > 1 else recognition_thresh

@app.get('/ids')
async def get_id():
  return json.dumps(FACES.list_person())

@app.post('/id')
async def create_id(id: str):
  try:
    FACES.create_person(person_id=id)
  except:
    raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail='Cannot create ID')

@app.post('/register')
async def post_face_image(id: str, request: Request, file: UploadFile = File(...)):
  if id not in FACES.list_person():
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f'ID {id} is not founded')
  request_object_content = await file.read()
  try:
    img = Image.open(io.BytesIO(request_object_content))
    box, emb = detect_face(np.array(img))
  except:
    raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type") 
  if emb is not None:
    hash = hashlib.md5(img.tobytes()).hexdigest()
    FACES.insert_face(id, hash, emb)
    #
    folder = FACES_IMG_DIR.joinpath(f'{id}')
    folder.mkdir(exist_ok=True)
    #
    img.save(folder.joinpath(f'{hash}.jpg'))
    return {'image': f'/imgs/{id}/{hash}.jpg'}
  elif box is None:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='No face founded')
  else:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='This image maybe faked, please try another image.')
  
@app.get('/register')
async def get_face_image(id: str):
  if id not in FACES.list_person():
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f'ID {id} is not founded')
  # res = [f'/imgs/{id}/{k}' for k in os.listdir(FACES_IMG_DIR.joinpath(f'{id}')) if k.endswith(".jpg")]
  res = [''.join(x.id.split('-')) for x in FACES.list_face(id)[0] if x is not None]
  res = [f'/imgs/{id}/{x}.jpg' for x in res]
  return res

@app.post('/check')
async def face_check(id: str, file: UploadFile = File(...)):
  if id not in FACES.list_person():
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f'ID {id} is not founded')
  request_object_content = await file.read()
  try:
    img = Image.open(io.BytesIO(request_object_content))
    box, emb = detect_face(np.array(img))
  except:
    raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Unsupported file type")
  if box is not None:
    if emb is not None:
      reg = recognize_face(emb, id)
      if reg:
        return json.dumps({'status': 'ok'})
      else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f'This face is not {id}')
    else:
      raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Face maybe faked.')
  else:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Face not founded or there are more than 1 face.')
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
    b64 = await queue.get()
    img = Image.open(BytesIO(base64.b64decode(b64)))
    if not img.mode == 'RGB':
      img = img.convert('RGB')
    box, emb = detect_face(np.array(img))
    res = {}
    if box is not None:
      res['box'] = str(box)
      if emb is not None:
        reg = recognize_face(emb)
        res['found'] = 'true' if reg else 'false'
      else:
        res['found'] = 'fake'
    await websocket.send_text(json.dumps(res))


