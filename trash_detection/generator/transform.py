from PIL import Image
from .bg_remover.utils import get_crop_mask
import random


def random_rotate(objects):
  for obj in objects:
    obj.img.rotate(random.randint(0, 360), resample=Image.Resampling.BICUBIC, expand=True)
    obj.img, _ = get_crop_mask(obj.img)
  return objects