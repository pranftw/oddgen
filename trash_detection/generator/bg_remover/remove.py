from PIL import Image
from torchvision.transforms.functional import normalize
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from ..objects import ObjectImage, get_crop_box
from .utils import load_model
import torch
import numpy as np
import torch.nn.functional as F
import os


# References: https://github.com/xuebinqin/DIS/blob/main/IS-Net/Inference.py https://huggingface.co/spaces/doevent/dis-background-removal/blob/main/app.py https://github.com/xuebinqin/U-2-Net/blob/master/u2net_portrait_demo.py


class BGRemoverDataset(Dataset):
  def __init__(self, objects, upsample_size, mean, std):
    self.objects = objects
    self.upsample_size = upsample_size
    self.mean = mean
    self.std = std
  
  def __len__(self):
    return len(self.objects)
  
  def __getitem__(self, idx):
    img = torch.tensor(self.objects[idx].img_np, dtype=torch.float32).permute(2,0,1)
    img = F.interpolate(torch.unsqueeze(img,0), self.upsample_size, mode='bilinear').type(torch.uint8)
    img = torch.divide(img, 255.0)
    img = normalize(img, self.mean, self.std)
    return img.squeeze(0)


class BGRemover:
  def __init__(self, remover=remove_bg_u2, batch_size=1, model_path='models/u2net_script_model.pt', weights_path='models/u2net.pth'):
    self.remover = remover
    self.batch_size = batch_size
    self.model_path = model_path
    self.weights_path = weights_path
  
  def __call__(self, objects):
    self.remover(objects, self.batch_size, self.model_path, self.weights_path)


def remove_bg_dis(objects, batch_size, model_path, weights_path):
  # model: models/isnet_script_model.pt
  # weights: models/isnet-general-use.pth
  model, device = load_model(model_path, weights_path)
  preprocess(objects)
  dataset = BGRemoverDataset(objects, (1024,1024), [0.5,0.5,0.5], [1.0,1.0,1.0])
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  
  with torch.no_grad():
    outputs = []
    for data_batch in dataloader:
      output = model(data_batch.to(device))
      outputs+=output[0][0]
      del output
    for obj, output in zip(objects, outputs):
      process_output(output, obj)
  return objects


def remove_bg_u2(objects, batch_size, model_path, weights_path):
  # model: models/u2net_script_model.pt
  # weights: models/u2net.pth
  model, device = load_model(model_path, weights_path, strict_weights_loading=False)
  preprocess(objects)
  dataset = BGRemoverDataset(objects, (320,320), [0.406,0.485,0.456], [0.225,0.229,0.224])
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  with torch.no_grad():
    outputs = []
    for data_batch in dataloader:
      output = model(data_batch.to(device))
      outputs+=output[0]
      del output
    for obj, output in zip(objects, outputs):
      process_output(output, obj)
  return objects


def preprocess(objects):
  for obj in objects:
    obj_img = obj.img.copy()
    if obj_img.mode=='RGBA' or obj_img.mode=='CMYK':
      obj_img = obj_img.convert('RGB')
    img = np.asarray(obj_img)
    if len(img.shape) < 3:
      img = img[:, :, np.newaxis]
    obj.img_np = img


def process_output(output, obj):
  output = torch.unsqueeze(output, 0)
  mask = torch.squeeze(F.interpolate(output, tuple(reversed(obj.img.size)), mode='bilinear'), 0) # since objects[i].img hasnt changed, the size will be the same as the original
  max_val = torch.max(mask)
  min_val = torch.min(mask)
  mask = (mask-min_val)/(max_val-min_val)
  mask = (mask*255.0).permute(1,2,0).cpu().data.numpy().astype(np.uint8).squeeze(2) # squeeze removes the third dim in mask
  pil_mask = Image.fromarray(mask).convert('L')
  orig_object = obj.img.copy()
  orig_object.putalpha(pil_mask)
  bg_removed_object = orig_object.crop(get_crop_box(obj.padding_amounts, orig_object.size))
  obj.img = bg_removed_object