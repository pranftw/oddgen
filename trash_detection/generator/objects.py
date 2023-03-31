from PIL import Image
from pathos.multiprocessing import ProcessingPool as Pool
from concurrent.futures import ThreadPoolExecutor
from .utils import get_bbox, save_objects, get_annotations
import json
import random


class ObjectImage:
  def __init__(self, img, category, padding_amounts):
    self.img = img
    self.category = category
    self.padding_amounts = padding_amounts
    self.bbox = None


def extract_objects(annotations_fpath, num_workers, bg_remover_batch_size, crop_padding=0, save_to_fpath=None):
  annotations_dict = get_annotations(annotations_fpath)
  cropped_objects = crop(annotations_dict, num_workers, padding=crop_padding)
  wo_bg_objects = remove_bg(cropped_objects, bg_remover_batch_size)
  if save_to_fpath is not None:
    save_objects(wo_bg_objects, save_to_fpath)
  return wo_bg_objects


def paste_objects(objects, new_img):
  new_img_width, new_img_height = new_img.img.size
  for obj in objects:
    left, upper = random.randint(0, new_img_width), random.randint(0,new_img_height)
    bbox = [left, upper, *obj.img.size] # should be combined with category to get annotation
    obj.bbox = bbox
    new_img.img.paste(obj.img, (left, upper), obj.img)


def crop(annotations_dict, num_workers, padding):
  '''
    annotations - (left, upper, width, height, category)
    padding is the extra number of pixels to consider while cropping the image so that it isn't too closely cropped
  '''
  cropped_objects = []
  def _crop(annotation_dict_item):
    objects = []
    img_fpath, annotations = annotation_dict_item
    with Image.open(img_fpath) as orig_img:
      for annotation in annotations:
        left, upper, width, height, category = annotation
        bbox = get_bbox(left, upper, width, height)
        padded_bbox = add_padding(padding, bbox, *orig_img.size)
        obj = ObjectImage(img=orig_img.crop(padded_bbox), category=category, padding_amounts=get_padding_amounts(padded_bbox, bbox))
        objects.append(obj)
    return objects

  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    items = [(img_fpath, annotations) for img_fpath, annotations in annotations_dict.items()]
    results = pool.map(_crop, items)  

  for result in results:
    cropped_objects+=result
  return cropped_objects


def remove_bg(objects, batch_size):
  from .bg_remover import remove_bg as bg_remover
  bg_remover(objects, batch_size)
  return objects


def resize(objects, size):
  for obj in objects:
    obj_width, obj_height = obj.img.size
    if obj_width>=obj_height: # horizontal
      obj.img.thumbnail(size)
    else: # vertical
      obj.img.thumbnail((size[1], size[0]))


def add_padding(padding, bbox, width, height):
  left, upper, right, lower = bbox
  padding_left = left-padding
  padding_upper = upper-padding
  padding_right = right+padding
  padding_lower = lower+padding
  if padding_left<0:padding_left=0
  if padding_upper<0:padding_upper=0
  if padding_right>width:padding_right=width
  if padding_lower>height:padding_lower=height
  return padding_left, padding_upper, padding_right, padding_lower


def get_padding_amounts(padded_bbox, bbox):
  padding_amounts = [abs(padded_bbox_element-bbox_element) for padded_bbox_element, bbox_element in zip(padded_bbox, bbox)]
  return padding_amounts


def get_crop_box(padding_amounts, padded_img_size):
  padded_img_width, padded_img_height = padded_img_size
  padding_amount_left, padding_amount_upper, padding_amount_right, padding_amount_lower = padding_amounts
  left = padding_amount_left
  upper = padding_amount_upper
  right = padded_img_width-padding_amount_right
  lower = padded_img_height-padding_amount_lower
  return left, upper, right, lower