from PIL import Image
from rembg import remove as rembg_remove
from .utils import get_bbox


def crop(img_fpaths, annotations):
  '''
    annotations - (left, upper, width, height, category)
  '''
  cropped_objects = []
  for img_fpath in self.img_fpaths:
    with Image.open(img_fpath) as orig_img:
      for annotation in annotations:
        left, upper, width, height, category = annotation
        bbox = get_bbox(left, upper, width, height)
        cropped_objects.append((orig_img.crop(bbox), category))
  return cropped_objects


def remove_bg(cropped_objects):
  wo_bg_objects = []
  for cropped_img,category in cropped_objects:
    wo_bg_objects.append((rembg_remove(cropped_img), category))
  return wo_bg_objects