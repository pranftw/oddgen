from oddgen.utils import get_dir_categorized_objects, save_objects, get_annotations
from oddgen.transform import random_rotate
from oddgen import generate, add_texture
from PIL import Image, ImageDraw
import os



oddgen_dir = os.getenv('oddgen_path') # empty directory path where the generated object detection dataset will be saved in NOTE: REPLACE THIS or SET ENV
textures_fpath = os.getenv('oddgen_textures_path') # path to backgrounds for generated images NOTE: REPLACE THIS or SET ENV
dataset_dir = os.getenv('dataset_path') # This can be replaced by absolute path to your downloaded dataset NOTE: REPLACE THIS or SET ENV


# <-------------- DATASET GENERATION -------------->
def generate_oddgen_dataset():
  os.mkdir(os.path.join(oddgen_dir, 'objects')) # create objects directory

  objs = get_dir_categorized_objects(dataset_dir)
  # objs = BGRemover(weights_path='')(dir_objs) # BGRemover to remove background from images in the dataset NOTE: Change weights_path
  save_objects(objs, os.path.join(oddgen_dir, 'objects'))

  generate(
    num_imgs=50,
    img_size=(300, 150),
    objects_fpath = os.path.join(oddgen_dir, 'objects'),
    max_objects_in_each_img=5,
    object_size=(50, 50),
    object_transformations=[(random_rotate,)],
    fpath=oddgen_dir,
    min_objects_in_each_img=1,
    num_workers=8,
    batch_size=5
  )


# <-------------- VISUALIZE BBOX -------------->
def visualize(gen_img_fpath, anns_fpath=os.path.join(oddgen_dir, 'annotations.json')):
  # gen_img_fpath is any img fpath generated in oddgen_dir

  annotations = get_annotations(anns_fpath)
  categories = ['phone', 'ship', 'plane']

  def get_bbox(left, upper, width, height):
    right = left+width
    lower = upper+height
    return left, upper, right, lower

  def annotate(img_fpath):
    annotation = annotations[img_fpath]
    final_annotations = [(get_bbox(*ann[0:-1]), categories[ann[-1]]) for ann in annotation]
    return Image.open(img_fpath), final_annotations

  def vis(img, anns, color='lawngreen'):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    font_size = 10
    for bbox, cls in anns:
      draw.rectangle(bbox, outline=color, width=2)
      draw.text([bbox[0],bbox[1]-font_size], cls)
    img.show()

  vis(*annotate(gen_img_fpath))


# <-------------- ADD BACKGROUND TO GENERATED DATASET -------------->
def add_background():
  add_texture(
    generated_imgs_fpath=oddgen_dir,
    textures_fpath=textures_fpath,
    max_textures_per_img=1,
    img_size=(300,150), # should be the same as size of generated image
    save_to=os.path.join(oddgen_dir, 'with_bg')
  )


if __name__=='__main__':
  # <---- RUN THE FOLLOWING ONE BY ONE BY UNCOMMENTING ---->
  # generate_oddgen_dataset()
  # visualize(os.path.join(oddgen_dir, '0af61fda02d0d937.png')) # change image fname
  # add_background()
  # visualize(os.path.join(oddgen_dir, 'with_bg', '0--9c44aa28d28e2838.png'), anns_fpath=os.path.join(oddgen_dir, 'with_bg', 'annotations.json')) # change image fname
  pass