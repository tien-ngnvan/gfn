import cv2
import numpy as np
import websocket
import websockets
import sys
import pickle
import struct ### new code
import asyncio

# def on_message(ws, message):
#     print(f"Received message: {message}")

# def on_error(ws, error):
#     print(f"Encountered error: {error}")

# def on_close(ws, close_status_code, close_msg):
#     print("Connection closed")

# def on_open(ws):
#     print("Connection opened")
#     ws.send("Hello, Server!")

# if __name__ == "__main__":
#     ws = websocket.WebSocketApp("ws://127.0.0.1:8000/ws",
#                                 on_message=on_message,
#                                 on_error=on_error,
#                                 on_close=on_close)
#     ws.on_open = on_open
#     ws.run_forever()

async def hello(uri):
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)
        while True:
            #
            ret, frame = cap.read()
            data = pickle.dumps(frame)
            await websocket.send(data)
            #
            data = await websocket.recv()
            data: dict = pickle.loads(data)
            #
            boxes = data['boxes']
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                print(f"box: {box}")
                

asyncio.run(hello('ws://127.0.0.1:8000/ws'))