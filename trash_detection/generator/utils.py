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
    img_details = {'id':i, 'file_name':generated_img.fname}
    if generated_img.annotations is None:
      generated_img.annotations = []
      for obj in generated_img.objects:
        obj_details = {'id':obj_id, 'image_id':i, 'bbox':obj.bbox, 'category':obj.category}
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


def get_annotations(annotations_fpath, with_segmentation=False):
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
    reqd_annotation = bbox + [annotation['category_id']] # the annotation we require is left, upper, width, height, category
    if with_segmentation:
      reqd_annotation+=annotation['segmentation']
    else:
      reqd_annotation+=[None]
    reqd_annotation = tuple(reqd_annotation)
    img_annotations[annotation['image_id']].append(reqd_annotation)
  
  final_annotations = {}
  for img_fname, img_id in img_fpath.items():
    final_annotations[img_fname] = img_annotations[img_id]
  return final_annotations


def get_imgs_from_dir(dir_path, ext=None, num_workers=4):
  fpaths = os.listdir(dir_path)
  def _get_img(fpath):
    if (ext is not None) and (fpath.endswith(ext)):
      return Image.open(os.path.join(dir_path, fpath))
  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    imgs = pool.map(_get_img, fpaths)
  imgs = list(filter(lambda img:img is not None, imgs))
  return imgs


def get_generated_imgs_from_annotations(annotations_fpath, num_workers):
  from .generate import GeneratedImage
  parent_dir = os.path.dirname(annotations_fpath)
  annotations = get_annotations(annotations_fpath)
  all_img_fnames = list(annotations['images'].keys())
  def _get_generated_img(fname):
    generated_img = GeneratedImage()
    generated_img.img = Image.open(os.path.join(parent_dir+fname))
    generated_img.fname = fname
    generated_img.annotations = annotations['annotations']
    return generated_img
  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    return pool.map(_get_generated_img)