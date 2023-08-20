from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms import RandomCrop
import numpy as np
import torch
import secrets
import os
import random
import math


def load_model(model_path, weights_path, strict_weights_loading=True, device=None):
  parent_dir = os.path.dirname(__file__)
  if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = torch.jit.load(os.path.join(parent_dir, model_path)).to(device)
  if weights_path is not None:
    model.load_state_dict(torch.load(os.path.join(parent_dir, weights_path), map_location=device), strict=strict_weights_loading)
  model.eval()
  return model, device


def get_crop_mask(img):
  img_np = np.asarray(img).transpose((2,0,1))
  alpha = img_np[3]
  alpha_img = Image.fromarray(alpha)
  sum_0 = alpha.sum(axis=0)!=0
  sum_1 = alpha.sum(axis=1)!=0
  sum_0_nz = np.flatnonzero(sum_0)
  sum_1_nz = np.flatnonzero(sum_1)
  left, right = sum_0_nz[0], sum_0_nz[-1]
  upper, lower = sum_1_nz[0], sum_1_nz[-1]
  cropped_img = img.crop((left, upper, right, lower))
  cropped_alpha = alpha_img.crop((left, upper, right, lower))
  return cropped_img, cropped_alpha


def generate_crop_masks(fpaths, save_to, num_workers=4):
  crop_dir = os.path.join(save_to, 'crop')
  mask_dir = os.path.join(save_to, 'mask')
  os.mkdir(crop_dir)
  os.mkdir(mask_dir)

  def _crop_mask(fpath):
    img = Image.open(fpath)
    cropped_img, cropped_alpha = get_crop_mask(img)

    fname = f'{secrets.token_hex(8)}.png'
    cropped_img.save(os.path.join(crop_dir, fname))
    cropped_alpha.save(os.path.join(mask_dir, fname))

  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    pool.map(_crop_mask, fpaths)


def get_fpaths_crop_mask(crop_mask_dir):
  crop_dir = os.path.join(crop_mask_dir, 'crop')
  mask_dir = os.path.join(crop_mask_dir, 'mask')
  def _get_fpaths(dir):
    filter_fn = lambda fpath:fpath.endswith('.png') # only select png images
    fpaths = sorted(list(filter(filter_fn, os.listdir(dir))))
    fpaths = [os.path.join(dir, fpath) for fpath in fpaths]
    return fpaths
  return _get_fpaths(crop_dir), _get_fpaths(mask_dir)


def save_txt_fpaths_crop_masks(crop_mask_dir, num_reqd=None):
  seed = random.randint(1000, 10000)
  def _save(name, fpaths):
    random.seed(seed)
    dir = os.path.join(crop_mask_dir, name)
    if num_reqd is not None:
      fpaths = random.sample(fpaths, num_reqd)
    fpaths_str = '\n'.join(fpaths)
    with open(os.path.join(crop_mask_dir, f'{name}.txt'), 'w') as fp:
      fp.write(fpaths_str)

  crop_fpaths, mask_fpaths = get_fpaths_crop_mask(crop_mask_dir)
  _save('crop', crop_fpaths)
  _save('mask', mask_fpaths)


def get_crop_mask_fpaths_from_txt(crop_mask_dir):
  def _get_fpaths(name):
    with open(os.path.join(crop_mask_dir, f'{name}.txt')) as fp:
      fpaths_str = fp.read()
    fpaths = fpaths_str.split('\n')
    return fpaths
  return _get_fpaths('crop'), _get_fpaths('mask')


def add_padding_crop_masks(crop_mask_dir, max_padding, save_to, num_workers=4):
  padded_crop_dir = os.path.join(save_to, 'crop')
  padded_mask_dir = os.path.join(save_to, 'mask')
  os.makedirs(padded_crop_dir)
  os.makedirs(padded_mask_dir)
  crop_fpaths, mask_fpaths = get_crop_mask_fpaths_from_txt(crop_mask_dir) # this takes the paths directly from the saved txt fpaths

  def _add_padding(args):
    crop_fpath, mask_fpath = args
    crop_img, mask_img = Image.open(crop_fpath), Image.open(mask_fpath)
    crop_np, mask_np = np.asarray(crop_img), np.asarray(mask_img)
    pad_amts = [random.randint(0, max_padding) for _ in range(4)] # random pad amts for all four sides
    padded_crop_np = np.pad(crop_np, [pad_amts[0:2], pad_amts[2:4], [0,0]], constant_values=[0]*2)
    padded_mask_np = np.pad(mask_np, [pad_amts[0:2], pad_amts[2:4]], constant_values=[0]*2)
    padded_crop_img = Image.fromarray(padded_crop_np)
    padded_mask_img = Image.fromarray(padded_mask_np)
    padded_crop_img.save(os.path.join(padded_crop_dir, os.path.basename(crop_img.filename))) # saves padded images to the same filename as orig image
    padded_mask_img.save(os.path.join(padded_mask_dir, os.path.basename(mask_img.filename))) # saves padded images to the same filename as orig image
  
  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    pool.map(_add_padding, [(crop_fpath, mask_fpath) for crop_fpath, mask_fpath in zip(crop_fpaths, mask_fpaths)])


def add_bg_crop_masks(crop_mask_dir, bgs, max_imgs_per_bg, save_to, bg_random_crop=False, num_workers=4):
  bg_crop_dir = os.path.join(save_to, 'crop')
  bg_mask_dir = os.path.join(save_to, 'mask')
  os.makedirs(bg_crop_dir)
  os.makedirs(bg_mask_dir)
  crop_fpaths, mask_fpaths = get_crop_mask_fpaths_from_txt(crop_mask_dir) # this takes the paths directly from the saved txt fpaths

  def _add_bg(args):
    crop_fpath, mask_fpath = args
    num_bgs = random.randint(1, max_imgs_per_bg)
    crop_img = Image.open(crop_fpath)
    mask_img = Image.open(mask_fpath)
    chosen_bgs = random.sample(bgs, num_bgs)
    for i, bg in enumerate(chosen_bgs):
      bg_img = bg.copy()
      if bg_random_crop:
        try:
          bg_img = RandomCrop(crop_img.size)(crop_img)
        except ValueError:
          bg_img = bg_img.resize(crop_img.size)
      else:
        bg_img = bg_img.resize(crop_img.size)
      bg_img.paste(crop_img, (0,0), crop_img)
      crop_with_bg_img = bg_img
      crop_with_bg_img.save(os.path.join(bg_crop_dir, f'{i}--{os.path.basename(crop_img.filename)}'))
      mask_img.copy().save(os.path.join(bg_mask_dir, f'{i}--{os.path.basename(mask_img.filename)}'))
  with ThreadPoolExecutor(max_workers=num_workers) as pool:
    pool.map(_add_bg, [(crop_fpath, mask_fpath) for crop_fpath, mask_fpath in zip(crop_fpaths, mask_fpaths)])


# Different Background(BG) transforms for training

def random_noise_bg(bg):
  if random.random()>=0.5:
    rand_img = Image.fromarray(np.random.randint(0, 50, bg.size).T.astype(np.uint8))
    bg.paste(rand_img, (0,0), rand_img)
  return bg


def plain_bg_transform(img_size, with_noise=True):
  bg = Image.new(mode='RGBA', size=img_size, color=f'#{secrets.token_hex(3)}')
  return random_noise_bg(bg) if with_noise else bg


def vert_bg_transform(img_size, column_width=None):
  img_width, img_height = img_size
  if column_width is None:
    column_width = random.randint(1, img_width)
  num_cols = int(math.floor(img_width/column_width))
  strands = [plain_bg_transform((column_width, img_height), with_noise=False) for _ in range(num_cols)]
  strands+=[plain_bg_transform((img_width-(num_cols*column_width), img_height), with_noise=False)]
  strands_cat = np.concatenate([np.asarray(strand) for strand in strands], axis=1).astype(np.uint8)
  bg = Image.fromarray(strands_cat)
  return random_noise_bg(bg)


def horiz_bg_transform(img_size, row_height=None):
  img_width, img_height = img_size
  if row_height is None:
    row_height = random.randint(1, img_height)
  num_rows = int(math.floor(img_height/row_height))
  strands = [plain_bg_transform((img_width, row_height), with_noise=False) for _ in range(num_rows)]
  strands+=[plain_bg_transform((img_width, img_height-(num_rows*row_height)), with_noise=False)]
  strands_cat = np.concatenate([np.asarray(strand) for strand in strands], axis=0).astype(np.uint8)
  bg = Image.fromarray(strands_cat)
  return random_noise_bg(bg)


def random_bg_transform(img_size):
  bg_transforms = [plain_bg_transform, vert_bg_transform, horiz_bg_transform]
  return random.choice(bg_transforms)(img_size)