from PIL import Image
from utils import get_bbox


def crop(img_paths, annotations):
  '''
    annotations - (left, upper, width, height, category)
  '''
  cropped_objects = []
  for img_path in self.img_paths:
    with Image.open(img_path) as orig_img:
      for annotation in annotations:
        left, upper, width, height, category = annotation
        bbox = get_bbox(left, upper, width, height)
        cropped_objects.append((orig_img.crop(bbox)), category)
  return cropped_objects
