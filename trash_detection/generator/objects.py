from PIL import Image
from rembg import remove as rembg_remove
from .utils import get_bbox, save_objects, get_annotations
import json
import random


def extract_objects(annotations_fpath, save_to_fpath=None):
  annotations_dict = get_annotations(annotations_fpath)
  cropped_objects = crop(annotations_dict)
  wo_bg_objects = remove_bg(cropped_objects)
  if save_to_fpath is not None:
    save_objects(wo_bg_objects, save_to_fpath)
  return wo_bg_objects


def paste_objects(objects, new_img):
  new_img_width, new_img_height = new_img.size
  for obj in objects:
    new_img.paste(obj, box=(random.randint(0, new_img_width), random.randint(0,new_img_height)))
  return new_img


def crop(annotations_dict):
  '''
    annotations - (left, upper, width, height, category)
  '''
  cropped_objects = []
  for img_fpath, annotations in annotations_dict.items():
    with Image.open(img_fpath) as orig_img:
      for annotation in annotations:
        left, upper, width, height, category = annotation
        bbox = get_bbox(left, upper, width, height)
        cropped_objects.append((orig_img.crop(bbox), category))
  return cropped_objects


def remove_bg(objects):
  wo_bg_objects = []
  for obj,category in objects:
    wo_bg_objects.append((rembg_remove(obj), category))
  return wo_bg_objects


def resize(objects, size):
  for obj in objects:
    obj_width, obj_height = obj.size
    if obj_width>=obj_height: # horizontal
      obj.thumbnail(size)
    else: # vertical
      obj.thumbnail((size[1], size[0]))