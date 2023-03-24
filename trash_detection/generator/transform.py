from PIL import Image
import random


def random_rotate(objects):
  for obj in objects:
    obj.img.rotate(random.randint(0, 360), resample=Image.Resampling.BICUBIC, expand=True)
  return objects