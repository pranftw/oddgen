from .objects import crop, remove_bg
from .utils import save_objects
import json


def generate_objects(annotations_fpath):
  annotations_with_fpaths = get_annotations(annotations_fpath)
  img_fpaths, annotations = list(annotations_with_fpath.keys()), list(annotations_with_fpath.values())
  cropped_objects = crop(img_fpaths, annotations)
  wo_bg_objects = remove_bg(cropped_objects)
  save_objects(wo_bg_objects)


def get_annotations(annotations_fpath):
  with open(annotations_fpath) as fp:
    annotations_dict = json.load(fp)
  images = annotations_dict['images']
  annotations = annotations_dict['annotations']
  img_fpath = {}
  img_annotations = {}

  for img in images:
    img_fpath[img['file_name']] = img['id']
    img_annotations[img['id']] = []
  for annotation in annotations:
    bbox = annotation['bbox']
    reqd_annotation = tuple(bbox + [annotation['category_id']]) # the annotation we require is left, upper, width, height, category
    img_annotations[annotation['image_id']].append(reqd_annotation)
  
  final_annotations = {}
  for img_fname, img_id in img_fpath.items():
    final_annotations[img_fname] = img_annotations[img_id]
  return final_annotations