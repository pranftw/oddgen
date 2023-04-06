from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import secrets
import os


def crop_mask(args):
  fpath, crop_dir, mask_dir = args
  img = Image.open(fpath)
  img_tensor = torch.tensor(np.asarray(img)).permute(2,0,1)
  alpha = img_tensor[3]
  if not(torch.all(alpha==255.0).item()): # only executes if not all elements of alpha are 255
    alpha_img = Image.fromarray(alpha.numpy())
    sum_0 = alpha.sum(dim=0)!=0
    sum_1 = alpha.sum(dim=1)!=0
    sum_0_nz = sum_0.nonzero()
    sum_1_nz = sum_1.nonzero()
    le,ri = sum_0_nz[0].item(), sum_0_nz[-1].item()
    up,lo = sum_1_nz[0].item(), sum_1_nz[-1].item()
    cropped_img = img.crop((le,up,ri,lo))
    cropped_alpha = alpha_img.crop((le,up,ri,lo))
    fname = f'{secrets.token_hex(8)}.png'
    cropped_img.save(os.path.join(crop_dir, fname))
    cropped_alpha.save(os.path.join(mask_dir, fname))


def generate_crop_masks(fpaths, save_to):
  crop_dir = os.path.join(save_to, 'crop')
  mask_dir = os.path.join(save_to, 'mask')
  os.mkdir(crop_dir)
  os.mkdir(mask_dir)
  with ThreadPoolExecutor() as pool:
    pool.map(crop_mask, [(fpath, crop_dir, mask_dir) for fpath in fpaths])


def load_model(model_path, weights_path, strict_weights_loading=True):
  parent_dir = os.path.dirname(__file__)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = torch.jit.load(os.path.join(parent_dir, model_path)).to(device)
  if weights_path is not None:
    model.load_state_dict(torch.load(os.path.join(parent_dir, weights_path)), strict=strict_weights_loading)
  model.eval()
  return model, device