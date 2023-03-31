from PIL import Image
from torchvision.transforms.functional import normalize
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
# from .objects import ObjectImage
import torch
import numpy as np
import torch.nn.functional as F
import os


# implementation references from https://github.com/xuebinqin/DIS/blob/main/IS-Net/Inference.py https://huggingface.co/spaces/doevent/dis-background-removal/blob/main/app.py

class ObjectImage:
  def __init__(self, img, category):
    self.img = img
    self.category = category
    self.bbox = None


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
    img = F.interpolate(torch.unsqueeze(img,0), self.upsample_size, mode="bilinear").type(torch.uint8)
    img = torch.divide(img,255.0)
    img = normalize(img, self.mean, self.std)
    return img.squeeze(0)


def remove_bg(objects, batch_size, model_path='bg_remover_models/isnet_script_model.pt', weights_path='bg_remover_models/isnet-general-use.pth'):
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


def remove_bg_u2(objects, batch_size, model_path='bg_remover_models/u2netp_script_model.pt', weights_path='bg_remover_models/u2netp.pth'):
  model, device = load_model(model_path, weights_path, strict_weights_loading=False)
  preprocess(objects)
  dataset = BGRemoverDataset(objects, (512,512), [0.406,0.485,0.456], [0.225,0.229,0.224])
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


def load_model(model_path, weights_path, strict_weights_loading=True):
  parent_dir = os.path.dirname(__file__)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = torch.jit.load(os.path.join(parent_dir, model_path)).to(device)
  model.load_state_dict(torch.load(os.path.join(parent_dir, weights_path)), strict=strict_weights_loading)
  model.eval()
  return model, device


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
  bg_removed_object = orig_object
  obj.img = bg_removed_object


objects = [ObjectImage(Image.open(fpath),1) for fpath in ['ignore/trashnet/train/700.jpg', 'ignore/trashnet/train/579.jpg']]
for obj in objects:
  obj.img.show()
remove_bg_u2(objects, 1)
for obj in objects:
  obj.img.show()