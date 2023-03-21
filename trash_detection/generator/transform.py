from PIL import Image
import random


def random_rotate(imgs):
  return [img.rotate(random.randint(0, 360), resample=Image.Resampling.BICUBIC, expand=True) for img in imgs]