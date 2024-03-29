from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os
import secrets
import json
import math


def get_bbox(left, upper, width, height):
  right = left+width
  lower = upper+height
  return left, upper, right, lower


def save_objects(objects, fpath):
  for obj in objects:
    fname = f'{obj.category}--{secrets.token_hex(8)}.png'
    obj.img.save(os.path.join(fpath, fname))


def save_generated_imgs(generated_imgs, fpath):
  for generated_img in generated_imgs:
    generated_img.img.save(os.path.join(fpath, generated_img.fname)) # images will be stored in png because then later on background will have to be added. During that time we can convert it to jpg


def save_generated_annotations(generated_imgs, fpath):
  annotations_dict = {'images':[], 'annotations':[]}
  obj_id = 1 # object id always starts from 1 in COCO
  for i,generated_img in enumerate(generated_imgs):
    generated_img_width, generated_img_height = generated_img.img_size
    img_details = {'id':i, 'file_name':generated_img.fname, 'width':generated_img_width, 'height':generated_img_height}
    if generated_img.annotations is None:
      generated_img.annotations = []
      for obj in generated_img.objects:
        obj_details = {'id':obj_id, 'image_id':i, 'bbox':obj.bbox, 'category_id':obj.category}
        generated_img.annotations.append(obj_details)
        obj_id+=1
    else:
      for annotation in generated_img.annotations:
        annotation['id'] = obj_id
        annotation['image_id'] = i
        obj_id+=1
    annotations_dict['annotations']+=generated_img.annotations
    annotations_dict['images'].append(img_details)
  with open(fpath, 'w') as fp:
    json.dump(annotations_dict, fp)


def get_annotations(annotations_fpath):
  with open(annotations_fpath) as fp:
    annotations_dict = json.load(fp)
  images = annotations_dict['images']
  annotations = annotations_dict['annotations']
  img_fpath = {}
  img_annotations = {}

  for img in images:
    img_fpath[os.path.join(os.path.dirname(annotations_fpath), img['file_name'])] = img['id']
    img_annotations[img['id']] = []
  for annotation in annotations:
    bbox = annotation['bbox']
    reqd_annotation = tuple(bbox + [annotation['category_id']]) # the annotation we require is left, upper, width, height, category
    img_annotations[annotation['image_id']].append(reqd_annotation)
  
  final_annotations = {}
  for img_fname, img_id in img_fpath.items():
    final_annotations[img_fname] = img_annotations[img_id]
  return final_annotations


def get_imgs_from_dir(dir_path, ext=None, num_workers=4, fnames=None):
  if fnames is None:
    fnames = os.listdir(dir_path)
  def _get_img(fname):
    if (ext is not None) and (fname.endswith(ext)): # ext can also be a tuple of str
      return Image.open(os.path.join(dir_path, fname))#.convert('RGBA')
  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    imgs = pool.map(_get_img, fnames)
  imgs = list(filter(lambda img:img is not None, imgs))
  return imgs


def get_generated_imgs_from_annotations(annotations_fpath, img_fnames=None, num_workers=4):
  # img_fnames should be a list of full absolute paths
  from .generate import GeneratedImage
  annotations = get_annotations(annotations_fpath)
  if img_fnames is None:
    generated_imgs_fpath = os.path.dirname(annotations_fpath)
    img_fnames = [os.path.join(generated_imgs_fpath, fname) for fname in list(annotations.keys())]
  def _get_generated_img(fname):
    generated_img = GeneratedImage()
    generated_img.img = Image.open(fname)
    generated_img.fname = os.path.basename(fname)
    generated_img.img_size = generated_img.img.size
    annotation_tuples = annotations[fname]
    generated_img.annotations = [{'bbox': annotation_tuple[:-1], 'category_id': annotation_tuple[-1]} for annotation_tuple in annotation_tuples]
    return generated_img
  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    return pool.map(_get_generated_img, img_fnames)


def get_objects(dir_path):
  from .objects import ObjectImage
  imgs = get_imgs_from_dir(dir_path, ext='.png')
  obj_imgs = []
  for img in imgs:
    category,_ = os.path.basename(img.filename).split('--')
    obj_img = ObjectImage(img, int(category), [0]*4)
    obj_imgs.append(obj_img)
  return obj_imgs


def get_dir_categorized_objects(dirpath, ext='.png'):
  # category must be an <category-name_category-idx>
  from .objects import ObjectImage
  obj_imgs = []
  category_paths = os.listdir(dirpath)
  if '.DS_Store' in category_paths: category_paths.remove('.DS_Store') # for macos
  for category_path in category_paths:
    category = category_path.split('_')[1]
    imgs = get_imgs_from_dir(os.path.join(dirpath, category_path), ext=ext)
    for img in imgs:
      obj_img = ObjectImage(img, int(category), [0]*4)
      obj_imgs.append(obj_img)
  return obj_imgs