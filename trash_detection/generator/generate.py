from PIL import Image
from .objects import extract_objects
from .utils import save_imgs
import random


def generate(num_imgs, img_size, objects, max_objects_in_each_img, object_transformations, fpath):
  new_imgs = [Image.new(mode='RGBA', size=img_size) for _ in range(num_imgs)]
  for new_img in new_imgs:
    selected_objects = random.sample(objects, random.randint(0, max_objects_in_each_img))
    for transformation_fn, *args in object_transformations:
      selected_objects = transformation_fn(selected_objects, *args)
  save_imgs(new_imgs, fpath)


def generate_from_annotations(annotations_fpath, num_imgs, img_size, max_objects_in_each_img, object_transformations, fpath):
  generate(num_imgs, img_size, extract_objects(annotations_fpath), max_objects_in_each_img, object_transformations, fpath)