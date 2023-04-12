from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import secrets
import os


def get_crop_mask(img):
  img_np = np.asarray(img).transpose((2,0,1))
  alpha = img_np[3]
  alpha_img = Image.fromarray(alpha)
  sum_0 = alpha.sum(axis=0)!=0
  sum_1 = alpha.sum(axis=1)!=0
  sum_0_nz = np.flatnonzero(sum_0)
  sum_1_nz = np.flatnonzero(sum_1)
  le,ri = sum_0_nz[0], sum_0_nz[-1]
  up,lo = sum_1_nz[0], sum_1_nz[-1]
  cropped_img = img.crop((le,up,ri,lo))
  cropped_alpha = alpha_img.crop((le,up,ri,lo))
  return cropped_img, cropped_alpha


def generate_crop_masks(fpaths, save_to, bg_img=None):
  crop_dir = os.path.join(save_to, 'crop')
  mask_dir = os.path.join(save_to, 'mask')
  os.mkdir(crop_dir)
  os.mkdir(mask_dir)
  if bg_img is not None:
    bg_img = bg_img.convert('RGBA')

  def _crop_mask(fpath):
    img = Image.open(fpath)
    cropped_img, cropped_alpha = get_crop_mask(img)
    if bg_img is not None:
      cropped_img_with_bg = bg_img.copy()
      cropped_img_with_bg = cropped_img_with_bg.resize(cropped_img.size)
      cropped_img_with_bg.paste(cropped_img, (0,0), cropped_img)
      cropped_img = cropped_img_with_bg

    fname = f'{secrets.token_hex(8)}.png'
    cropped_img.save(os.path.join(crop_dir, fname))
    cropped_alpha.save(os.path.join(mask_dir, fname))

  with ThreadPoolExecutor() as pool:
    pool.map(_crop_mask, fpaths)


def load_model(model_path, weights_path, strict_weights_loading=True, device=None):
  parent_dir = os.path.dirname(__file__)
  if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = torch.jit.load(os.path.join(parent_dir, model_path)).to(device)
  if weights_path is not None:
    model.load_state_dict(torch.load(os.path.join(parent_dir, weights_path), map_location=device), strict=strict_weights_loading)
  model.eval()
  return model, device