from PIL import Image
from pathos.multiprocessing import ProcessingPool as Pool
from concurrent.futures import ThreadPoolExecutor
from rembg import remove as rembg_remove
from .utils import get_bbox, save_objects, get_annotations
import json
import random
import numpy as np


class ObjectImage:
  def __init__(self, img, category):
    self.img = img
    self.category = category
    self.bbox = None


def extract_objects(annotations_fpath, num_workers, save_to_fpath=None):
  annotations_dict = get_annotations(annotations_fpath)
  cropped_objects = crop(annotations_dict, num_workers)
  wo_bg_objects = remove_bg(cropped_objects, num_workers)
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


def crop(annotations_dict, num_workers):
  '''
    annotations - (left, upper, width, height, category)
  '''
  cropped_objects = []
  def _crop(annotation_dict_item):
    objects = []
    img_fpath, annotations = annotation_dict_item
    with Image.open(img_fpath) as orig_img:
      for annotation in annotations:
        left, upper, width, height, category = annotation
        bbox = get_bbox(left, upper, width, height)
        obj = ObjectImage(img=orig_img.crop(bbox), category=category)
        objects.append(obj)
    return objects

  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    items = [(img_fpath, annotations) for img_fpath, annotations in annotations_dict.items()]
    results = pool.map(_crop, items)  

  for result in results:
    cropped_objects+=result
  return cropped_objects


def remove_bg(objects, num_workers):
  np.seterr(all="ignore")
  object_imgs = [obj.img for obj in objects]
  with Pool(num_workers) as pool:
    results = pool.map(rembg_remove, object_imgs)
  for obj, result in zip(objects, results):
    obj.img = result
  return objects


def resize(objects, size):
  for obj in objects:
    obj_width, obj_height = obj.img.size
    if obj_width>=obj_height: # horizontal
      obj.img.thumbnail(size)
    else: # vertical
      obj.img.thumbnail((size[1], size[0]))