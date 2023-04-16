from PIL import Image
from pathos.multiprocessing import ProcessingPool as Pool
from .objects import extract_objects, paste_objects, resize
from .utils import save_generated_imgs, save_generated_annotations, get_imgs_from_dir
from bg_remover.remove import BGRemover
from copy import deepcopy
import random
import os
import secrets


class GeneratedImage:
  def __init__(self, img_size=None):
    self.img = Image.new(mode='RGBA', size=img_size) if img_size is not None else None
    self.fname = f'{secrets.token_hex(8)}.png'
    self.objects = None
    self.annotations = None

  def add_objects(self, objects):
    self.objects = objects


def generate(num_imgs, img_size, objects, max_objects_in_each_img, object_size, object_transformations, fpath, num_workers=4):
  if object_size[0]<object_size[1]:
    raise ValueError(f'First dim({object_size[0]}) should be greater than or equal to second dim({object_size[1]})')
  new_imgs = [GeneratedImage(img_size=img_size) for _ in range(num_imgs)]

  def _generate(new_img):
    selected_objects = random.sample(objects, random.randint(0, max_objects_in_each_img))
    for object_transformation in object_transformations:
      transformation_fn, *args = object_transformation
      selected_objects = transformation_fn(selected_objects, *args)
    resize(selected_objects, object_size)
    new_img.add_objects(selected_objects)
    paste_objects(selected_objects, new_img)
    return new_img
  
  with Pool(num_workers) as pool:
    new_imgs = pool.map(_generate, new_imgs)

  save_generated_imgs(new_imgs, fpath)
  save_generated_annotations(new_imgs, os.path.join(fpath, 'annotations.json'))
  return new_imgs


def generate_from_annotations(annotations_fpath, num_imgs, img_size, max_objects_in_each_img, object_size, object_transformations, fpath, bg_remover=BGRemover(), crop_padding=0, num_workers=4):
  objects_fpath = os.path.join(fpath, 'objects')
  if not os.path.exists(objects_fpath):
    os.mkdir(objects_fpath)
  extracted_objects = extract_objects(annotations_fpath, bg_remover, num_workers, crop_padding, objects_fpath)
  return generate(num_imgs, img_size, extracted_objects, max_objects_in_each_img, object_size, object_transformations, fpath, num_workers)


def add_texture(generated_imgs, textures_fpath, max_textures_per_img, img_size, save_to, num_workers=4):
  if not os.path.exists(save_to):
    os.mkdir(save_to)
  
  def get_textures(textures_fpath, img_size):
    textures = get_imgs_from_dir(textures_fpath, ext='.jpg', num_workers=num_workers)
    for texture in textures:
      texture.putalpha(255)
      texture.thumbnail(img_size)
    return textures
  textures = get_textures(textures_fpath, img_size)

  def _add(generated_img):
    textured_generated_imgs = []
    selected_textures = random.sample(textures, random.randint(1, max_textures_per_img))
    for i,texture in enumerate(selected_textures):
      bg = texture.copy()
      bg.paste(generated_img.img, (0,0), generated_img.img)
      textured_generated_img = GeneratedImage()
      textured_generated_img.img = bg
      textured_generated_img.annotations = deepcopy(generated_img.annotations)
      textured_generated_img.fname = f'{i}--{textured_generated_img.fname}'
      textured_generated_imgs.append(textured_generated_img)
    return textured_generated_imgs

  with Pool(max_workers=num_workers) as pool:
    outputs = pool.map(_add, generated_imgs)
    textured_generated_imgs = []
    for output in outputs:
      textured_generated_imgs+=output
  save_generated_imgs(textured_generated_imgs, save_to)
  save_generated_annotations(textured_generated_imgs, os.path.join(save_to, 'annotations.json'))