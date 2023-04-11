'''
	Referenced from:
		https://github.com/xuebinqin/U-2-Net/blob/master/data_loader.py
		https://github.com/xuebinqin/U-2-Net/blob/master/u2net_train.py

	NOTE: run this file directly, not as a part of the module
	NOTE: scikit-image has to be installed using: pip install scikit-image
'''


from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torchvision import transforms, utils
from skimage import io, transform, color
from utils import load_model
import tqdm
import os
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
import random
import math
import argparse


# <-----------------------------DATALOADER------------------------->

class RescaleT(object):
	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		h, w = image.shape[:2]
		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size
		new_h, new_w = int(new_h), int(new_w)
		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)
		return {'imidx':imidx, 'image':img, 'label':lbl}


class RandomCrop(object):
	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self,sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		if random.random() >= 0.5:
			image = image[::-1]
			label = label[::-1]
		h, w = image.shape[:2]
		new_h, new_w = self.output_size
		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)
		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]
		return {'imidx':imidx, 'image':image, 'label':label}


class ToTensorLab(object):
	def __call__(self, sample):
		imidx, image, label = sample['imidx'], sample['image'], sample['label']
		tmpLbl = np.zeros(label.shape)
		max_lbl = np.max(label)
		max_img = np.max(image)
		label = label if (max_lbl<1e-6 or max_lbl==0) else label/max_lbl
		# change the color space flag=0
		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		image = image if (max_img<1e-6 or max_img==0) else image/max_img
		# normalizing
		tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
		tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
		tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
		tmpLbl[:,:,0] = label[:,:,0]
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))
		return {'imidx':torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class TrainDataset(Dataset):
	def __init__(self, img_paths, lbl_paths, num_data, transform=None):
		num_img_paths = len(img_paths)
		num_lbl_paths = len(lbl_paths)
		assert num_img_paths==num_lbl_paths, f'Total number of images and labels should match! Got {num_img_paths} images and {num_lbl_paths} labels.'
		if num_data is not None:
			assert num_data<=num_img_paths, f'number of data points required {num_data} should be lesser than or equal to total number of data {num_img_paths}'
		self.num_data = num_data if num_data is not None else num_img_paths
		self.img_paths, self.lbl_paths = zip(*(random.sample(list(zip(img_paths, lbl_paths)), self.num_data)))
		self.transform = transform

	def __len__(self):
		return len(self.img_paths)
	
	def __getitem__(self, idx):
		image = np.asarray(Image.open(self.img_paths[idx]).convert('RGB')) # CxWxH
		label = np.asarray(Image.open(self.lbl_paths[idx]))[:,:,np.newaxis] # WxH
		imidx = np.array([idx])
		sample = {'imidx':imidx, 'image':image, 'label':label}
		if self.transform:
			sample = self.transform(sample)
		return sample


# <-----------------------------ARGS HANDLING----------------------------->

parser = argparse.ArgumentParser()
parser.add_argument('--imgs_path', type=str, required=True)
parser.add_argument('--lbls_path', type=str, required=True)
parser.add_argument('--num_epochs', type=int, required=True)
parser.add_argument('--save_model_weights_in', type=str, required=True)
parser.add_argument('--batch_size', type=int, nargs='?', default=8)
parser.add_argument('--num_workers', type=int, nargs='?', default=1)
parser.add_argument('--num_data', type=int, nargs='?', default=None)
parser.add_argument('--model_save_freq', type=int, nargs='?', default=None)
parser.add_argument('--model_path', type=str, nargs='?', default='models/u2netp_script_model.pt')

args = parser.parse_args()


# <-----------------------------TRAINING----------------------------->

BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
MODEL_SAVE_FREQ = args.model_save_freq # save after a number of epochs
NUM_EPOCHS = args.num_epochs
MODEL_PATH = args.model_path
SAVE_MODEL_WEIGHTS_IN = args.save_model_weights_in
NUM_DATA = args.num_data # if None, use all the data in the dataset
IMGS_PATH = args.imgs_path
LBLS_PATH = args.lbls_path

try:
	os.mkdir(SAVE_MODEL_WEIGHTS_IN)
except FileExistsError:
	if len(os.listdir(SAVE_MODEL_WEIGHTS_IN))>0:
		raise ValueError(f'Directory SAVE_MODEL_WEIGHTS_IN-{SAVE_MODEL_WEIGHTS_IN} is not empty!')

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
	bce_loss = nn.BCELoss(reduction='mean')
	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)
	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	return loss0, loss


with open(IMGS_PATH) as fp:
	img_paths = fp.read().split('\n')
with open(LBLS_PATH) as fp:
	lbl_paths = fp.read().split('\n')
train_dataset = TrainDataset(
		img_paths=img_paths,
		lbl_paths=lbl_paths,
		num_data=NUM_DATA,
		transform=transforms.Compose([
			RescaleT(320),
			RandomCrop(288),
			ToTensorLab()
		]))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
train_num = NUM_DATA

model, device = load_model(MODEL_PATH, None, strict_weights_loading=False)
optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

model.train()
for epoch in range(NUM_EPOCHS):
	running_loss = 0.0
	running_tar_loss = 0.0
	with tqdm.tqdm(train_dataloader, unit='batch', mininterval=0) as tobj:
		tobj.set_description(f'Epoch {epoch+1}')
		for i, data in enumerate(tobj):
			inputs, labels = data['image'].type(torch.FloatTensor), data['label'].type(torch.FloatTensor)
			inputs_v, labels_v = Variable(inputs.to(device), requires_grad=False), Variable(labels.to(device), requires_grad=False)

			optim.zero_grad()
			d0, d1, d2, d3, d4, d5, d6 = model(inputs_v)
			loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
			loss.backward()
			optim.step()

			running_loss += loss.data.item()
			running_tar_loss += loss2.data.item()
			del d0, d1, d2, d3, d4, d5, d6, loss2, loss
			tobj.set_postfix({'train_loss': f'{(running_loss/(i+1)):.2f}', 'tar': f'{(running_tar_loss/(i+1)):.2f}'})
		if (MODEL_SAVE_FREQ is not None) and (epoch%MODEL_SAVE_FREQ==0):
			torch.save(model.state_dict(), f'{SAVE_MODEL_WEIGHTS_IN}/epoch{epoch}.pth')
torch.save(model.state_dict(), f'{SAVE_MODEL_WEIGHTS_IN}/last.pth')