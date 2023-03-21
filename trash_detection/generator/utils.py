import os
import secrets
import json


def get_bbox(left, upper, width, height):
  right = left+width
  lower = upper+height
  return left, upper, right, lower


def save_objects(objects, fpath):
  for object_img, category in objects:
    object_fname = f'{category}--{secrets.token_hex(8)}.png'
    object_img.save(os.path.join(fpath, object_fname))


def save_imgs(imgs, fpath):
  for img in imgs:
    img_fname = f'{secrets.token_hex(8)}.png'
    img.save(os.path.join(fpath, img_fname))


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