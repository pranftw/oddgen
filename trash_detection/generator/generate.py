from PIL import Image
from pathos.multiprocessing import ProcessingPool as Pool
from .objects import extract_objects, paste_objects, resize
from .utils import save_generated_imgs, save_generated_annotations, get_imgs_from_dir
import random
import os
import secrets


class GeneratedImage:
  def __init__(self, img_size):
    self.img = Image.new(mode='RGBA', size=img_size)
    self.fname = f'{secrets.token_hex(8)}.png'
    self.objects = None

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

  try:
    os.mkdir(fpath)
  except FileExistsError:
    pass
  save_generated_imgs(new_imgs, fpath)
  save_generated_annotations(new_imgs, os.path.join(fpath, 'annotations.json'))
  return new_imgs


def generate_from_annotations(annotations_fpath, num_imgs, img_size, max_objects_in_each_img, object_size, object_transformations, fpath, crop_padding=0, bg_remover_batch_size=4, num_workers=4):
  try:
    os.mkdir(fpath)
  except FileExistsError:
    pass
  objects_fpath = os.path.join(fpath, 'objects')
  try:
    os.mkdir(objects_fpath)
  except FileExistsError:
    pass
  return generate(num_imgs, img_size, extract_objects(annotations_fpath, num_workers=num_workers, bg_remover_batch_size=bg_remover_batch_size, crop_padding=crop_padding, save_to_fpath=objects_fpath), max_objects_in_each_img, object_size, object_transformations, fpath, num_workers)


def add_texture(generated_imgs, textures_fpath, max_textures_per_img, img_size, save_to, num_workers=4):
  try:
    os.mkdir(save_to)
  except FileExistsError:
    pass
  
  def get_textures(textures_fpath, img_size):
    textures = get_imgs_from_dir(textures_fpath, ext='.jpg', num_workers=num_workers)
    for texture in textures:
      texture.putalpha(255)
      texture.thumbnail(img_size)
    return textures
  textures = get_textures(textures_fpath, img_size)

  def _add(generated_img):
    all_imgs = []
    selected_textures = random.sample(textures, random.randint(1, max_textures_per_img))
    for texture in selected_textures:
      bg = texture.copy()
      bg.paste(generated_img.img, (0,0), generated_img.img)
      all_imgs.append(bg)
    return all_imgs

  with Pool(max_workers=num_workers) as pool:
    textured_imgs = pool.map(_add, generated_imgs)
  
  for textured_img_group in textured_imgs:
    fname = f'{secrets.token_hex(8)}.png'
    for i,textured_img in enumerate(textured_img_group):
      textured_img.save(os.path.join(save_to, f'{i}-{fname}'))