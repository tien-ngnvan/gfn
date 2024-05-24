import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def crops(image, bbox):
  h, w = image.shape[:2]
  x1, y1, x2, y2 = bbox
  x1 = int(max(0, x1))
  y1 = int(max(0, y1))
  x2 = int(min(w, x2))
  y2 = int(min(h, y2))
  return image[y1:y2, x1:x2]


def to_base64(image: Image) -> str:
  buffered = BytesIO()
  image.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue())
  return img_str


def from_base64(base64str: str) -> Image:
  im = Image.open(BytesIO(base64.b64decode(base64str)))
  return im


def softmax(x:np.array) -> list[float]:
    """ Softmax function

    Args:
        x (np.array): an numpy array of predict model

    Returns:
        list[float]: a probability of x
    """
    s= np.sum(np.exp(x))
    return np.exp(x)/s
    
    
def cosine(embed: np.array, anchor: np.array):
    """ Calculate cosine similarity between two vector embedding

    Args:
        embed (np.array): The embedding predict
        anchor (np.array): The embedding target

    Returns:
        prob: list of probability of between 2 vectors
        idx: index of item
    """
   
    anchor = torch.Tensor(anchor)
    b = torch.Tensor(embed)
    
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    result = torch.mm(anchor, b_norm.transpose(0,1))
    
    prob, idx = torch.max(result, dim=0)

    return prob, idx 
