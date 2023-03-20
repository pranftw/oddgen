from PIL import Image


def generate(num_imgs, img_size, object_transformations, objects):
  generated_imgs = [Image.new(mode='RGBA', size=img_size) for _ in range(num_imgs)]
  