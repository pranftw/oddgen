from PIL import Image
import numpy as np
import torch


img = Image.open('ShapeNetRendering/04090263/1c4361c083f3abe2cc34b900bb2492e/rendering/15.png')
img_tensor = torch.tensor(np.asarray(img)).permute(2,0,1)
alpha = img_tensor[3]
alpha_img = Image.fromarray(alpha.numpy())
sum_0 = alpha.sum(dim=0)!=0
sum_1 = alpha.sum(dim=1)!=0
sum_0_nz = sum_0.nonzero()
sum_1_nz = sum_1.nonzero()
le,ri = sum_0_nz[0].item(), sum_0_nz[-1].item()
up,lo = sum_1_nz[0].item(), sum_1_nz[-1].item()
cropped_img = img.crop((le,up,ri,lo))
cropped_alpha = alpha_img.crop((le,up,ri,lo))
cropped_img.show()
cropped_alpha.show()