import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import normalize


# NOTE: AFTER TESTING, CHANGE OBJECT to INCORPORATE .image
def remove_bg(objects, model_path='bg_remover_models/isnet_scripted_reduction_gpu.pt', weights_path='bg_remover_models/isnet-general-use.pth'):
  bg_removed_objects = []

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = torch.jit.load(model_path).to(device)
  model.load_state_dict(torch.load(weights_path))
  model.eval()

  processed_imgs = preprocess(objects, device)
  imgs_cat = torch.cat([img for img,_ in processed_imgs])
  with torch.no_grad():
    outputs = model(imgs_cat)
  for i, output in enumerate(outputs[0][0]):
    output = torch.unsqueeze(output, 0)
    mask = torch.squeeze(F.upsample(output, processed_imgs[i][1], mode='bilinear'), 0) # processed_imgs[i][1] is the corresponding shape
    max_val = torch.max(mask)
    min_val = torch.min(mask)
    mask = (mask-min_val)/(max_val-min_val)
    mask = (mask*255.0).permute(1,2,0).cpu().data.numpy().astype(np.uint8).squeeze(2) # squeeze removes the third dim in mask
    pil_mask = Image.fromarray(mask).convert('L')
    orig_object = objects[i].copy()
    orig_object.putalpha(pil_mask)
    bg_removed_object = orig_object
    bg_removed_objects.append(bg_removed_object)

def preprocess(objects, device):
  imgs = []
  input_size = (1024, 1024)
  for obj in objects:
    if obj.mode=='RGBA' or obj.mode=='CMYK':
      obj = obj.convert('RGB')
    img = np.asarray(obj)
    if len(img.shape) < 3:
      img = img[:, :, np.newaxis]
    orig_im_shp=img.shape[0:2]
    im_tensor = torch.tensor(img, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
    img = torch.divide(im_tensor,255.0)
    img = normalize(img,[0.5,0.5,0.5],[1.0,1.0,1.0])
    img = img.to(device)
    imgs.append((img, orig_im_shp))
  return imgs

objects = [Image.open(fpath) for fpath in ['ignore/trashnet/train/700.jpg', 'ignore/trashnet/train/579.jpg']]
remove_bg(objects)