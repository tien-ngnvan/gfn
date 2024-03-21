import cv2
import asyncio
import numpy as np
import pickle
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from gfn.models import HeadFace, SpoofingNet


app = FastAPI()
#
headface = HeadFace('weights/yolov7-hf-v1.onnx')
#
spoofing = SpoofingNet('weights/OCI2M.onnx')
spoofing_thresh = 0.7


def softmax(x):
  return np.exp(x)/np.sum(np.exp(x))


@app.websocket('/ws')
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
  logging.info("Receive")
  bytes = await websocket.receive_bytes()
  try:
    queue.put_nowait(bytes)
  except asyncio.QueueFull:
    pass


async def kyc(websocket: WebSocket, queue: asyncio.Queue):
  while True:
    bytes = await queue.get()
    img = pickle.loads(bytes)
    fimg = img.copy()
    res = dict()
    #
    boxes, scores, _, kpts = headface.detect(img, get_layer='face', conf_det=0.5)
    if len(boxes) == 1:
      res['box'] = boxes
      #
      spoof = spoofing.get_features(fimg, boxes, kpts)
      spoof = softmax(spoof)[:,0]
      if spoof[0] > spoofing_thresh:
        res['id'] = 'real'
      else:
        res['id'] = 'fake'
    data = pickle.dumps(res)
    await websocket.send_bytes(data)
