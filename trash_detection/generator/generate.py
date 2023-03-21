from PIL import Image
from .objects import extract_objects, paste_objects
from .utils import save_imgs
import random
import os


def generate(num_imgs, img_size, objects, max_objects_in_each_img, object_transformations, fpath):
  new_imgs = [Image.new(mode='RGBA', size=img_size) for _ in range(num_imgs)]
  for new_img in new_imgs:
    selected_objects = random.sample(objects, random.randint(0, max_objects_in_each_img))
    for object_transformation in object_transformations:
      transformation_fn, *args = object_transformation
      selected_objects = transformation_fn(selected_objects, *args)
    paste_objects(selected_objects, new_img)
  save_imgs(new_imgs, fpath)


def generate_from_annotations(annotations_fpath, num_imgs, img_size, max_objects_in_each_img, object_transformations, fpath):
  objects_fpath = os.path.join(fpath, 'objects')
  try:
    os.mkdir(objects_fpath)
  except FileExistsError:
    pass
  extracted_objects = [obj for obj,_ in extract_objects(annotations_fpath, save_to_fpath=objects_fpath)]
  generate(num_imgs, img_size, extracted_objects, max_objects_in_each_img, object_transformations, fpath)