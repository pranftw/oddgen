import os
import secrets


def get_bbox(left, upper, width, height):
  right = left+width
  lower = upper+height
  return left, upper, width, height


def save_objects(objects, fpath):
  for object_img, category in objects:
    object_fname = f'{category}_{secrets.token_hex(8)}.png'
    object_img.save(os.path.join(fpath, object_fname))