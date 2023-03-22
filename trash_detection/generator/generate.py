from PIL import Image
from .objects import extract_objects, paste_objects, resize
from .utils import save_imgs
import random
import os
import secrets


class GeneratedImage:
  def __init__(self, img_size):
    self.img = Image.new(mode='RGBA', size=img_size)
    self.objects = None
    self.fname = f'{secrets.token_hex(8)}.png'

  def add_objects(self, objects):
    self.objects = objects


def generate(num_imgs, img_size, objects, max_objects_in_each_img, object_size, object_transformations, fpath):
  if object_size[0]<object_size[1]:
    raise ValueError(f'First dim({object_size[0]}) should be greater than or equal to second dim({object_size[1]})')
  new_imgs = [Image.new(mode='RGBA', size=img_size) for _ in range(num_imgs)]
  for new_img in new_imgs:
    selected_objects = random.sample(objects, random.randint(0, max_objects_in_each_img))
    for object_transformation in object_transformations:
      transformation_fn, *args = object_transformation
      selected_objects = transformation_fn(selected_objects, *args)
    resize(selected_objects, object_size)
    paste_objects(selected_objects, new_img)
  save_imgs(new_imgs, fpath)


def generate_from_annotations(annotations_fpath, num_imgs, img_size, max_objects_in_each_img, object_size, object_transformations, fpath):
  objects_fpath = os.path.join(fpath, 'objects')
  try:
    os.mkdir(objects_fpath)
  except FileExistsError:
    pass
  extracted_objects = [obj for obj,_ in extract_objects(annotations_fpath, save_to_fpath=objects_fpath)]
  generate(num_imgs, img_size, extracted_objects, max_objects_in_each_img, object_size, object_transformations, fpath)