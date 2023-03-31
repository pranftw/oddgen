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
  def __init__(self, objects, upsample_size):
    self.objects = objects
    self.upsample_size = upsample_size
  
  def __len__(self):
    return len(self.objects)
  
  def __getitem__(self, idx):
    img = torch.tensor(self.objects[idx].img_np, dtype=torch.float32).permute(2,0,1)
    img = F.interpolate(torch.unsqueeze(img,0), self.upsample_size, mode="bilinear").type(torch.uint8)
    img = torch.divide(img,255.0)
    img = normalize(img,[0.5,0.5,0.5],[1.0,1.0,1.0])
    return img.squeeze(0)


def remove_bg(objects, batch_size, model_path='bg_remover_models/isnet_script_model.pt', weights_path='bg_remover_models/isnet-general-use.pth'):
  parent_dir = os.path.dirname(__file__)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = torch.jit.load(os.path.join(parent_dir, model_path)).to(device)
  model.load_state_dict(torch.load(os.path.join(parent_dir, weights_path)))
  model.eval()

  preprocess(objects)
  dataset = BGRemoverDataset(objects, (1024,1024))
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  
  with torch.no_grad():
    outputs = []
    for data_batch in dataloader:
      output = model(data_batch.to(device))
      outputs+=output[0][0]
      if device=='cuda':
        torch.cuda.empty_cache()
    
    for i,output in enumerate(outputs):
      output = torch.unsqueeze(output, 0)
      mask = torch.squeeze(F.interpolate(output, tuple(reversed(objects[i].img.size)), mode='bilinear'), 0) # since objects[i].img hasnt changed, the size will be the same as the original
      max_val = torch.max(mask)
      min_val = torch.min(mask)
      mask = (mask-min_val)/(max_val-min_val)
      mask = (mask*255.0).permute(1,2,0).cpu().data.numpy().astype(np.uint8).squeeze(2) # squeeze removes the third dim in mask
      pil_mask = Image.fromarray(mask).convert('L')
      orig_object = objects[i].img.copy()
      orig_object.putalpha(pil_mask)
      bg_removed_object = orig_object
      objects[i].img = bg_removed_object # setting the removed bg img to corresponding obj.img
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


def remove_bg_u2(objects, batch_size=1, model_path='bg_remover_models/u2netp_script_model.pt', weights_path='bg_remover_models/u2netp.pth'):
  parent_dir = os.path.dirname(__file__)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = torch.jit.load(os.path.join(parent_dir, model_path)).to(device)
  model.load_state_dict(torch.load(os.path.join(parent_dir, weights_path)), strict=False)
  model.eval()

  preprocess_u2(objects)
  imgs_concat = Variable(torch.cat([obj.img_tensor for obj in objects])).to(device)

  with torch.no_grad():
    outputs = model(imgs_concat)
    reqd_outputs = outputs[0]
    del outputs
  
    for i,output in enumerate(reqd_outputs):
      output = output.unsqueeze(0)
      mask = torch.squeeze(F.interpolate(output, tuple(reversed(objects[i].img.size)), mode='bilinear'), 0) # since objects[i].img hasnt changed, the size will be the same as the original
      max_val = torch.max(mask)
      min_val = torch.min(mask)
      mask = (mask-min_val)/(max_val-min_val)
      mask = (mask*255.0).permute(1,2,0).cpu().data.numpy().astype(np.uint8).squeeze(2) # squeeze removes the third dim in mask
      pil_mask = Image.fromarray(mask).convert('L')
      orig_object = objects[i].img.copy()
      orig_object.putalpha(pil_mask)
      bg_removed_object = orig_object
      objects[i].img = bg_removed_object # setting the removed bg img to corresponding obj.img
  return objects


def preprocess_u2(objects):
  for obj in objects:
    obj_img = obj.img.resize((512, 512))
    if obj_img.mode=='RGBA' or obj_img.mode=='CMYK':
      obj_img = obj_img.convert('RGB')
    img = np.asarray(obj_img)
    
    tmp_img = np.zeros((*img.shape,))
    np.seterr(all="ignore")
    img = img/np.max(img)
    tmp_img[:,:,0] = (img[:,:,2]-0.406)/0.225
    tmp_img[:,:,1] = (img[:,:,1]-0.456)/0.224
    tmp_img[:,:,2] = (img[:,:,0]-0.485)/0.229
    tmp_img = tmp_img.transpose((2,0,1))
    tmp_img = tmp_img[np.newaxis,:,:,:]
    tmp_img = torch.from_numpy(tmp_img)
    tmp_img = tmp_img.type(torch.FloatTensor)
    obj.img_tensor = tmp_img

objects = [ObjectImage(Image.open(fpath),1) for fpath in ['ignore/trashnet/train/700.jpg', 'ignore/trashnet/train/579.jpg']]
for obj in objects:
  obj.img.show()
remove_bg_u2(objects)
for obj in objects:
  obj.img.show()