import cv2
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


app = FastAPI()

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
  print("X")
  bytes = await websocket.receive_bytes()
  try:
    queue.put_nowait(bytes)
  except asyncio.QueueFull:
    pass


async def kyc(websocket: WebSocket, queue: asyncio.Queue):
  while True:
    bytes = await queue.get()
    data = np.frombuffer(bytes, dtype=np.unit8)
    image = cv2.imdecode(data, 1)
    result = dict()
    await websocket.send_json(result)
