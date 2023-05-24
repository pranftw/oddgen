from PIL import Image
from pathos.multiprocessing import ProcessingPool as Pool
from .objects import extract_objects, paste_objects, resize, ObjectImage
from .utils import save_generated_imgs, save_generated_annotations, get_imgs_from_dir, get_generated_imgs_from_annotations
from .bg_remover.remove import BGRemover
from itertools import islice
from copy import deepcopy
import random
import os
import secrets
import glob


class GeneratedImage:
  def __init__(self, img_size=None):
    self.img = Image.new(mode='RGBA', size=img_size) if img_size is not None else None
    self.img_size = img_size
    self.fname = f'{secrets.token_hex(8)}.png'
    self.objects = None
    self.annotations = None

  def add_objects(self, objects):
    self.objects = objects


def generate(num_imgs, img_size, objects_fpath, max_objects_in_each_img, object_size, object_transformations, fpath, min_objects_in_each_img=0, num_workers=4, batch_size=100):
  if object_size[0]<object_size[1]:
    raise ValueError(f'First dim({object_size[0]}) should be greater than or equal to second dim({object_size[1]})')
  all_object_fpaths = [os.path.join(objects_fpath, object_fname) for object_fname in os.listdir(objects_fpath)]

  def get_objects_from_fpaths(object_fpaths):
    objs = []
    for object_fpath in object_fpaths:
      img = Image.open(object_fpath)
      category,_ = os.path.basename(object_fpath).split('--')
      objs.append(ObjectImage(img, int(category), [0]*4))
    return objs
  
  def _generate(new_img):
    selected_object_fpaths = random.sample(all_object_fpaths, random.randint(min_objects_in_each_img, max_objects_in_each_img))
    selected_objects = get_objects_from_fpaths(selected_object_fpaths)
    for object_transformation in object_transformations:
      transformation_fn, *args = object_transformation
      selected_objects = transformation_fn(selected_objects, *args)
    resize(selected_objects, object_size)
    new_img.add_objects(selected_objects)
    paste_objects(selected_objects, new_img)
    return new_img
  
  new_imgs = []
  num_generated = 0

  while num_generated<num_imgs:
    if num_generated+batch_size>num_imgs:
      iter_batch_size = num_imgs-num_generated
    else:
      iter_batch_size = batch_size
    generated_imgs = [GeneratedImage(img_size=img_size) for _ in range(iter_batch_size)]
    with Pool(max_workers=num_workers) as pool:
      generated_imgs = pool.map(_generate, generated_imgs)
    save_generated_imgs(generated_imgs, fpath)
    for generated_img in generated_imgs: del generated_img.img
    new_imgs+=generated_imgs
    num_generated+=iter_batch_size
    print(f'\r{num_generated}/{num_imgs}', end='')
    save_generated_annotations(new_imgs, os.path.join(fpath, 'annotations.json'))
  print()
  
  return new_imgs


def add_texture(generated_imgs_fpath, textures_fpath, max_textures_per_img, img_size, save_to, batch_size=100, num_workers=4):
  if not os.path.exists(save_to):
    os.mkdir(save_to)
  
  def get_textures(fnames):
    textures = get_imgs_from_dir(textures_fpath, ext='.jpg', num_workers=num_workers, fnames=fnames)
    processed_textures = []
    for texture in textures:
      texture.putalpha(255)
      processed_textures.append(texture.resize(img_size))
    return processed_textures

  all_texture_fpaths = os.listdir(textures_fpath)
  def _add(generated_img):
    textured_generated_imgs = []
    selected_textures = get_textures(random.sample(all_texture_fpaths, random.randint(1, max_textures_per_img)))
    for i,texture in enumerate(selected_textures):
      bg = texture.copy()
      bg.paste(generated_img.img, (0,0), generated_img.img)
      textured_generated_img = GeneratedImage()
      textured_generated_img.img = bg
      textured_generated_img.annotations = deepcopy(generated_img.annotations)
      textured_generated_img.fname = f'{i}--{textured_generated_img.fname}'
      textured_generated_img.img_size = deepcopy(generated_img.img_size)
      textured_generated_imgs.append(textured_generated_img)
    return textured_generated_imgs
  
  all_generated_imgs_fpaths = glob.glob(f'{generated_imgs_fpath}/*.png')

  # Referenced from: /a/62913856/11516790
  def batcher():
    iterator = iter(all_generated_imgs_fpaths)
    while batch:=list(islice(iterator, batch_size)):
      yield batch
  
  all_textured_generated_imgs = []
  num_textured = 0
  num_all_generated_imgs_fpaths = len(all_generated_imgs_fpaths)
  for batch in batcher():
    batch_generated_imgs = get_generated_imgs_from_annotations(os.path.join(generated_imgs_fpath, 'annotations.json'), batch, num_workers)
    with Pool(max_workers=num_workers) as pool:
      outputs = pool.map(_add, batch_generated_imgs)
      textured_generated_imgs = []
      for output in outputs:
        textured_generated_imgs+=output
    save_generated_imgs(textured_generated_imgs, save_to)
    for textured_generated_img in textured_generated_imgs: del textured_generated_img.img
    all_textured_generated_imgs+=textured_generated_imgs
    num_textured+=len(batch)
    print(f'\r{num_textured}/{num_all_generated_imgs_fpaths}', end='')
    save_generated_annotations(all_textured_generated_imgs, os.path.join(save_to, 'annotations.json'))
  print()