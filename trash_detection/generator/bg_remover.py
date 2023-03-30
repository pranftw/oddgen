from PIL import Image
from torchvision.transforms.functional import normalize
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch.nn.functional as F


# implementation references from https://github.com/xuebinqin/DIS/blob/main/IS-Net/Inference.py https://huggingface.co/spaces/doevent/dis-background-removal/blob/main/app.py


class ObjectImage:
  def __init__(self, img, category):
    self.img = img
    self.category = category
    self.bbox = None


class BGRemoverDataset(Dataset):
  def __init__(self, objects):
    self.objects = objects
  
  def __len__(self):
    return len(self.objects)
  
  def __getitem__(self, idx):
    return self.objects[idx].img_tensor


def remove_bg(objects, batch_size, model_path='bg_remover_models/isnet_script_model.pt', weights_path='bg_remover_models/isnet-general-use.pth'):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = torch.jit.load(model_path).to(device)
  model.load_state_dict(torch.load(weights_path))
  model.eval()

  preprocess(objects)
  dataset = BGRemoverDataset(objects)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  
  with torch.no_grad():
    outputs = []
    for data_batch in dataloader:
      output = model(data_batch.to(device))
      outputs+=output[0][0]
    
    for i,output in enumerate(outputs):
      output = torch.unsqueeze(output, 0)
      mask = torch.squeeze(F.upsample(output, tuple(reversed(objects[i].img.size)), mode='bilinear'), 0) # since objects[i].img hasnt changed, the size will be the same as the original
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
  input_size = (1024, 1024) # default input size for isnet model
  for obj in objects:
    obj_img = obj.img.copy()
    if obj_img.mode=='RGBA' or obj_img.mode=='CMYK':
      obj_img = obj_img.convert('RGB')
    img = np.asarray(obj_img)
    if len(img.shape) < 3:
      img = img[:, :, np.newaxis]
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
    img_tensor = F.upsample(torch.unsqueeze(img_tensor,0), input_size, mode="bilinear").type(torch.uint8)
    img = torch.divide(img_tensor,255.0)
    img = normalize(img,[0.5,0.5,0.5],[1.0,1.0,1.0])
    obj.img_tensor = img.squeeze(0)

objects = [ObjectImage(Image.open(fpath),1) for fpath in ['/home/pranav/Documents/py/trash_detection/ignore/trashnet/train/700.jpg', '/home/pranav/Documents/py/trash_detection/ignore/trashnet/train/579.jpg']]
for obj in objects:
  obj.img.show()
remove_bg(objects, 1)
for obj in objects:
  obj.img.show()