import cv2
import asyncio
import numpy as np
import pickle
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from gfn.models import HeadFace


app = FastAPI()

headface = HeadFace('weights/yolov7-hf-v1.onnx')


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
    #
    boxes, scores, _, kpts = headface.detect(img, get_layer='face')
    res = {
      'boxes': boxes
    }
    data = pickle.dumps(res)
    # await websocket.send_text("receive")
    await websocket.send_bytes(data)
