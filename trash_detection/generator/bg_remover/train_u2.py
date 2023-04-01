from .remove import load_model
from ..utils import get_annotations
from ..objects import crop
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np


BATCH_SIZE = 4
EPOCHS = 100
MODEL_PATH = 'models/u2netp_script_model.pt'
WEIGHTS_PATH = 'models/u2netp.pth'


class TrainDataset(Dataset):
  def __init__(self, annotations_fpath, num_workers, upsample_size):
    annotations_dict = get_annotations(annotations_fpath, with_segmentation=True)
    self.objects = crop(annotations_dict, num_workers, padding=0)
    self.upsample_size = upsample_size
  
  def __len__(self):
    return len(self.objects)
  
  def __getitem__(self, idx):
    obj = self.objects[idx]
    img = torch.tensor(np.asarray(obj.img), dtype=torch.float32).permute(2,0,1)
    img = F.interpolate(img, self.upsample_size, mode='bilinear').type(torch.uint8)/255.0
    img = normalize(img, [0.406,0.485,0.456], [0.225,0.229,0.224])
    mask_np = np.asarray(obj.mask)
    mask_np = mask_np/np.max(mask_np)
    mask = torch.tensor().permute(2,0,1)
    mask = F.interpolate(mask, self.upsample_size, mode='bilinear')
    return img, mask


def loss_fn(outputs, mask):
  bce_loss = nn.BCELoss(size_average=True)
  total_loss = 0
  for output in outputs:
    loss = bce_loss(output, mask)
    total_loss+=loss
  return total_loss


dataset = TrainDataset('data/annotations.json', 2, (320,320))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

model, device = load_model(MODEL_PATH, WEIGHTS_PATH, strict_weights_loading=False)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


for epoch in range(EPOCHS):
  epoch_loss = 0
  for img, mask in dataloader:
    optim.zero_grad()
    img = Variable(img.type(torch.FloatTensor), requires_grad=False).to(device)
    mask = Variable(img.type(torch.FloatTensor), requires_grad=False).to(device)
    outputs = model(img)
    loss = loss_fn(outputs, mask)
    loss.backward()
    optim.step()
    epoch_loss+=loss.item()
    del outputs, loss
  print(f'epoch: {epoch}, loss: {epoch_loss}')